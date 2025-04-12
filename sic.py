#!/usr/bin/env python3

import csv
import json
import click
import boto3
import time
import queue
import threading
import backoff
import botocore.exceptions

from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Iterator, Optional, Union
from loguru import logger


def blocks(file: object, size: int = 65536) -> Iterator[str]:
    """Efficiently read a file in blocks.
    
    Args:
        file: Open file handle
        size: Block size in bytes
        
    Yields:
        Blocks of file content
    """
    while True:
        b = file.read(size)
        if not b:
            break
        yield b


def count_file_lines(filepath: Union[str, Path]) -> int:
    """Count lines in a file efficiently.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Number of lines in the file
    """
    with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))


def process_s3_deletion_batches(reader: csv.reader, batch_size: int) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
    """Process the CSV reader and yield batches of objects for deletion.
    
    Args:
        reader: CSV reader object with columns [bucket_name, prefix, object_version, ...]
        batch_size: Maximum number of objects per batch
        
    Yields:
        Tuples of (bucket_name, batch_of_objects) for S3 deletion
    """
    # Use a dictionary to group objects by bucket
    bucket_batches: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    
    for row in reader:
        try:
            if len(row) >= 3:  # Ensure we have at least bucket, prefix, and version
                bucket_name, key, version_id = row[:3]
                
                # Skip if any of the required fields are empty
                if not bucket_name or not key:
                    logger.warning(f"Skipping row with empty bucket or key: {row}")
                    continue
                
                # Add the object to its bucket's batch
                bucket_batches[bucket_name].append({"Key": key, "VersionId": version_id})
                
                # If we have a full batch for any bucket, yield it
                for b, objects in list(bucket_batches.items()):
                    if len(objects) >= batch_size:
                        yield b, objects[:batch_size]
                        # Keep remaining objects for the next batch
                        bucket_batches[b] = objects[batch_size:]
            else:
                logger.warning(f"Skipping invalid row (not enough columns): {row}")
        except Exception as e:
            logger.error(f"Error processing row {row}: {e}")
    
    # Yield any remaining batches
    for bucket, objects in bucket_batches.items():
        if objects:
            yield bucket, objects


def delete_s3_objects_batch(bucket: str, objects: List[Dict[str, str]], throttle: float = 0) -> Tuple[int, List[Dict[str, str]]]:
    """Delete a batch of S3 objects with exponential backoff for rate limiting.
    
    Args:
        bucket: S3 bucket name
        objects: List of objects to delete (each with Key and VersionId)
        throttle: Seconds to wait between API calls
        
    Returns:
        Tuple of (success_count, failed_objects)
    """
    s3 = boto3.client('s3')
    failed_objects = []

    # Define the backoff condition for when a prefix has been deleted
    def is_non_retryable_exception(exception):
        if isinstance(exception, botocore.exceptions.ClientError):
            error_code = exception.response.get('Error', {}).get('Code')
            # Check for various "not found" error codes that indicate a prefix is already deleted
            if error_code in ['NoSuchKey', 'NoSuchBucket', 'NoSuchVersion', 'NotFound', '404']:
                logger.warning(f"Skipping objects in bucket {bucket} as they appear to be already deleted: {error_code}")
                # For these errors, we'll treat this as a "success" in that we don't need to delete them again
                # but we'll log it differently
                return True
        return False

    # Backoff decorator for the delete operation
    @backoff.on_exception(
        backoff.expo,
        botocore.exceptions.ClientError,
        max_time=300,  # Maximum of 5 minutes total retry time
        giveup=is_non_retryable_exception,  # Don't retry if resource is already gone
        on_backoff=lambda details: logger.warning(
            f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries "
            f"calling S3 delete_objects for bucket {bucket}"
        ),
        on_giveup=lambda details: logger.error(
            f"Giving up on S3 delete_objects for bucket {bucket} after {details['tries']} tries"
        ),
        base=2,  # Exponential backoff starting at 2 seconds
        factor=5,  # Start with a minimum of 2 seconds delay
        jitter=backoff.full_jitter,
    )
    def delete_with_backoff():
        try:
            response = s3.delete_objects(
                Bucket=bucket, 
                Delete={'Objects': objects, 'Quiet': False}
            )
            
            if 'Errors' in response and response['Errors']:
                for error in response['Errors']:
                    error_code = error.get('Code', 'Unknown')
                    
                    # Check if the error indicates the object is already deleted
                    if error_code in ['NoSuchKey', 'NoSuchBucket', 'NoSuchVersion', 'NotFound']:
                        logger.debug(f"Object already deleted: {error.get('Key', 'Unknown')}")
                    else:
                        failed_objects.append({
                            'Key': error.get('Key', 'Unknown'),
                            'VersionId': error.get('VersionId', 'Unknown'),
                            'Code': error_code,
                            'Message': error.get('Message', 'Unknown')
                        })
                    
            if throttle > 0:
                time.sleep(throttle)
                
            return len(objects) - len(failed_objects)
        
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            
            # Handle case where prefix has already been deleted
            if error_code in ['NoSuchKey', 'NoSuchBucket', 'NoSuchVersion', 'NotFound', '404']:
                logger.info(f"Prefix appears to be already deleted in bucket {bucket}: {error_code}")
                # Return success since we don't need to delete these objects
                return len(objects)
            
            # Re-raise for the backoff decorator to handle
            raise
    
    try:
        success_count = delete_with_backoff()
        return success_count, failed_objects
    except Exception as e:
        logger.error(f"Failed batch delete in bucket {bucket} after all retries: {e}")
        # Return all objects as failed in case of unhandled exception
        return 0, objects



def extract_all_directory_levels(prefix: str, max_depth: int = -1) -> List[str]:
    """
    Extract directory paths at all levels from top to deeper levels.
    Returns a list of directory paths with trailing slashes.
    Level 1 is the top-most directory, each subsequent level goes one directory deeper.

    Args:
        prefix: The S3 object key/prefix
        max_depth: Maximum directory depth to consider (-1 for no limit)

    Returns:
        List of directory paths from top level to specified depth
    """
    # Split the path and remove empty parts
    parts = [p for p in prefix.split("/") if p]

    # Check if the last part is a filename (contains a period) and remove it
    if parts and "." in parts[-1]:
        parts = parts[:-1]

    if not parts:
        return []

    # Determine the maximum number of levels to extract
    if max_depth == -1:
        # No limit
        levels_to_extract = len(parts)
    else:
        # Limit to max_depth or the actual number of parts, whichever is smaller
        levels_to_extract = min(max_depth, len(parts))

    # Generate directory levels from top to bottom
    levels = []

    # Start with the top level and go deeper up to the max_depth
    for i in range(1, levels_to_extract + 1):
        dir_path = "/".join(parts[:i]) + "/"
        levels.append(dir_path)

    return levels


def process_input_file(file_path: Path, max_depth: int) -> Dict[int, Dict[str, Tuple[int, int]]]:
    """
    Process a single CSV or Parquet file and return the aggregated sizes and file counts.

    Args:
        file_path: Path to the CSV or Parquet file
        max_depth: Maximum directory depth to consider

    Returns:
        Dictionary of {level: {directory: (size_bytes, file_count)}}
    """
    # Dictionary to store aggregated sizes for each level
    # For each directory, we store a tuple of (total_size, file_count)
    level_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    try:
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            with open(file_path, "r", newline="") as f:
                # Use csv module to handle quoted values
                reader = csv.reader(f, quoting=csv.QUOTE_ALL)

                for row in reader:
                    try:
                        if len(row) >= 3:  # Ensure row has at least 3 columns
                            prefix = row[1]
                            size_bytes = int(row[2])

                            # Extract directory paths at all levels (top to bottom)
                            dir_levels = extract_all_directory_levels(prefix, max_depth)

                            # Detect if this is a file (not a directory)
                            is_file = "." in prefix.split("/")[-1] if prefix else False

                            # Aggregate sizes and count files for each level
                            for level, dir_path in enumerate(dir_levels, 1):
                                # Update size
                                level_data[level][dir_path][0] += size_bytes

                                # Update file count if this is a file
                                if is_file:
                                    level_data[level][dir_path][1] += 1
                        else:
                            raise ValueError(f"Invalid number of rows {len(row)} in {file_path}: {row}")
                    except (IndexError, ValueError):
                        logger.error(f"Error processing row in CSV file {file_path}: {row}")
                        pass
        elif file_extension == '.parquet':
            try:
                import pyarrow.parquet as pq
            except ImportError:
                logger.error("pandas and pyarrow are required for Parquet support. Please install them with 'pip install pandas pyarrow'")
                return {}
            
            # Read the parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            # Ensure the dataframe has the required columns
            if len(df.columns) < 3:
                logger.error(f"Parquet file {file_path} does not have enough columns (expected at least 3)")
                return {}
            
            # Process each row in the dataframe
            for _, row in df.iterrows():
                try:
                    prefix = row['key']  # Second column contains the prefix/key
                    size_bytes = int(row['size'])  # Third column contains the size
                    
                    # Extract directory paths at all levels (top to bottom)
                    dir_levels = extract_all_directory_levels(prefix, max_depth)
                    
                    # Detect if this is a file (not a directory)
                    is_file = "." in prefix.split("/")[-1] if prefix else False
                    
                    # Aggregate sizes and count files for each level
                    for level, dir_path in enumerate(dir_levels, 1):
                        # Update size
                        level_data[level][dir_path][0] += size_bytes
                        
                        # Update file count if this is a file
                        if is_file:
                            level_data[level][dir_path][1] += 1
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"Error processing row in Parquet file {file_path}: {e}")
                    pass
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

    # Convert the defaultdict values to tuples for better serialization
    result = {}
    for level, dirs in level_data.items():
        result[level] = {dir_path: tuple(data) for dir_path, data in dirs.items()}

    return result


def process_files_in_batches(
    input_files: List[Path], max_depth: int, n_jobs: int, batch_size: int = 10
) -> List[Dict[int, Dict[str, Tuple[int, int]]]]:
    """
    Process files in batches with a progress bar.

    Args:
        input_files: List of CSV and Parquet files to process
        max_depth: Maximum directory depth to consider
        n_jobs: Number of parallel jobs
        batch_size: Size of batches for parallel processing

    Returns:
        List of results from all files
    """
    results = []

    with tqdm(total=len(input_files), desc="Processing input files") as pbar:
        for i in range(0, len(input_files), batch_size):
            batch_files = input_files[i : i + batch_size]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_input_file)(file_path, max_depth) for file_path in batch_files
            )
            results.extend(batch_results)
            pbar.update(len(batch_files))

    return results


def combine_results(
    results_list: List[Dict[int, Dict[str, Tuple[int, int]]]],
) -> Dict[int, Dict[str, Tuple[int, int]]]:
    """
    Combine results from multiple processes.

    Args:
        results_list: List of dictionaries with level data from each process

    Returns:
        Combined dictionary of {level: {directory: (size_bytes, file_count)}}
    """
    combined = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for result in results_list:
        for level, dirs in result.items():
            for dir_path, (size, count) in dirs.items():
                combined[level][dir_path][0] += size  # Add size
                combined[level][dir_path][1] += count  # Add count

    # Convert the defaultdict values to tuples
    result = {}
    for level, dirs in combined.items():
        result[level] = {dir_path: tuple(data) for dir_path, data in dirs.items()}

    return result


def process_input_files(input_dir: Path, output_dir: Path, max_depth: int, n_jobs: int) -> None:
    """
    Process all CSV files in the input directory and generate aggregated reports.
    Uses parallel processing to speed up file processing.

    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save output files
        max_depth: Maximum directory depth to consider (-1 for no limit)
        n_jobs: Number of parallel jobs to run (-1 for all available cores)
    """
    # Find all CSV files in the input directory
    csv_files = list(input_dir.glob("*.csv"))
    parequet_files = list(input_dir.glob("*.parquet"))
    input_files = csv_files + parequet_files

    if not input_files:
        logger.warning(f"No input files found in {input_dir}")
        return

    logger.info(f"Found {len(input_files)} files to process")

    # Process files in parallel with tqdm progress bar
    results = process_files_in_batches(input_files, max_depth, n_jobs)

    # Combine results from all processes
    logger.info("Combining results from all files...")
    level_data = combine_results(results)

    # Determine the maximum level
    max_level = max(level_data.keys()) if level_data else 0

    # Convert sizes to GB and write to output files
    logger.info("Writing output files...")
    for level in tqdm(range(1, max_level + 1), desc="Writing level reports"):
        if level not in level_data:
            continue

        output_file = output_dir / f"level_{level}_sizes.csv"

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Directory", "Size (GB)", "File Count"])

            # Sort by size (descending)
            sorted_dirs = sorted(
                level_data[level].items(),
                key=lambda x: x[1][0],  # Sort by size (first element of tuple)
                reverse=True,
            )

            for dir_path, (size_bytes, file_count) in sorted_dirs:
                size_gb = size_bytes / (1024**3)  # Convert bytes to GB
                writer.writerow([dir_path, f"{size_gb:.2f}", file_count])

    logger.success(f"Reports have been written to {output_dir}")

    if max_depth == -1:
        logger.info(f"Generated {max_level} level reports, from top level directories (level 1) to deeper structures")
    else:
        logger.info(
            f"Generated {max_level} level reports (max depth: {max_depth}), from top level directories (level 1) to deeper structures"
        )


def process_batch(
    curr_bucket: str, 
    batch: List[Dict[str, str]], 
    throttle: float = 0
) -> Tuple[str, int, List[Dict[str, str]]]:
    """Process a single batch of objects from one bucket.
    
    Args:
        curr_bucket: Bucket name
        batch: List of objects to delete
        throttle: Seconds to wait between API calls
        
    Returns:
        Tuple of (bucket_name, success_count, failed_objects)
    """
    success, failed = delete_s3_objects_batch(curr_bucket, batch, throttle)
    return curr_bucket, success, failed


def partition_prefix(bucket: str, prefix: str, delimiter: str = '/', max_prefixes: int = 1000) -> List[str]:
    """
    Partition a prefix into multiple prefixes for parallel processing.
    
    Args:
        bucket: S3 bucket name
        prefix: Base prefix to partition
        delimiter: Delimiter to use for partitioning
        max_prefixes: Maximum number of prefixes to return
        
    Returns:
        List of prefixes to process in parallel
    """
    s3 = boto3.client('s3')
    
    # If prefix doesn't end with delimiter, get common prefixes under it
    if prefix and not prefix.endswith(delimiter):
        prefix = prefix + delimiter if prefix else ""
    
    try:
        # Get common prefixes to parallelize the work
        paginator = s3.get_paginator('list_objects_v2')
        common_prefixes = []
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
            for cp in page.get('CommonPrefixes', []):
                common_prefixes.append(cp['Prefix'])
                
                # If we've collected enough prefixes, stop
                if len(common_prefixes) >= max_prefixes:
                    break
            
            # Also break the outer loop if we have enough prefixes
            if len(common_prefixes) >= max_prefixes:
                break
        
        # If we found some common prefixes, return them
        if common_prefixes:
            return common_prefixes
    except Exception as e:
        logger.warning(f"Error partitioning prefix {prefix}: {e}")
    
    # If no common prefixes or an error occurred, return the original prefix
    return [prefix]


def process_prefix_chunk(bucket: str, prefix: str, marker_queue: queue.Queue, batch_size: int = 10000) -> int:
    """
    Process a single prefix chunk and put delete markers into a queue.
    
    Args:
        bucket: S3 bucket name
        prefix: Prefix to process
        marker_queue: Queue to put results in
        batch_size: Number of markers to batch before putting in queue
        
    Returns:
        Count of delete markers found
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_object_versions')
    count = 0
    batch = []
    
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            delete_markers = page.get('DeleteMarkers', [])
            for marker in delete_markers:
                batch.append((
                    bucket,
                    marker['Key'],
                    marker['VersionId'],
                    marker['IsLatest']
                ))
                count += 1
                
                # If batch is full, put it in queue and reset
                if len(batch) >= batch_size:
                    marker_queue.put(batch)
                    batch = []
        
        # Put any remaining items
        if batch:
            marker_queue.put(batch)
            
        return count
    except Exception as e:
        logger.error(f"Error processing prefix {prefix}: {e}")
        if batch:
            marker_queue.put(batch)
        return count


def find_and_write_delete_markers(bucket: str, prefix: str, output_file: Path, 
                                 jobs: int = 20, batch_size: int = 10000, 
                                 delimiter: str = '/') -> None:
    """
    Find all objects with delete markers in an S3 bucket and write them to a CSV file.
    
    Args:
        bucket: S3 bucket name
        prefix: Prefix to limit the search
        output_file: Path to output CSV file
        jobs: Number of parallel jobs
        batch_size: Number of markers to batch before writing
        delimiter: Delimiter to use for prefix partitioning
    """
    # Create a queue for communication
    marker_queue = queue.Queue(maxsize=jobs * 2)  # Buffer some batches
    stop_event = threading.Event()
    total_markers = [0]  # Use a list for mutable reference
    
    # Partition the prefix for parallel processing
    logger.info(f"Partitioning prefix '{prefix}' for parallel processing")
    prefixes = partition_prefix(bucket, prefix, delimiter, max_prefixes=jobs*10)
    logger.info(f"Found {len(prefixes)} prefixes to process")
    
    # Writer thread function
    def writer_thread():
        markers_written = 0
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bucket', 'key', 'version_id', 'is_latest'])
            
            with tqdm(desc="Writing delete markers", unit="markers") as pbar:
                while not (stop_event.is_set() and marker_queue.empty()):
                    try:
                        batch = marker_queue.get(timeout=1)
                        if batch is None:  # Sentinel value
                            break
                            
                        writer.writerows(batch)
                        batch_size = len(batch)
                        markers_written += batch_size
                        pbar.update(batch_size)
                        marker_queue.task_done()
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error writing to file: {e}")
                
                total_markers[0] = markers_written
    
    # Start the writer thread
    writer = threading.Thread(target=writer_thread)
    writer.daemon = True
    writer.start()
    
    # Process prefixes in parallel with progress tracking
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        # Submit all prefix chunks to executor
        futures = {
            executor.submit(
                process_prefix_chunk, 
                bucket, 
                chunk_prefix, 
                marker_queue, 
                batch_size
            ): chunk_prefix for chunk_prefix in prefixes
        }
        
        # Track progress with tqdm
        with tqdm(total=len(prefixes), desc="Processing prefixes", unit="prefix") as pbar:
            for future in as_completed(futures):
                prefix_val = futures[future]
                try:
                    result = future.result()
                    logger.debug(f"Prefix '{prefix_val}': Found {result} delete markers")
                except Exception as e:
                    logger.error(f"Error processing prefix '{prefix_val}': {e}")
                
                pbar.update(1)
    
    # Signal the writer to stop by putting a sentinel value
    marker_queue.put(None)
    
    # Wait for writer to finish
    writer.join()
    
    logger.success(f"Completed finding delete markers. Total found: {total_markers[0]}")
    logger.info(f"Results written to {output_file}")


def find_and_write_all_versions(bucket: str, prefix: str, output_file: Path, 
                            jobs: int = 20, batch_size: int = 10000, 
                            delimiter: str = '/', include_delete_markers: bool = True,
                            include_versions: bool = True, latest_only: bool = False) -> None:
    """
    Find all object versions and/or delete markers in an S3 bucket and write them to a CSV file.
    
    Args:
        bucket: S3 bucket name
        prefix: Prefix to limit the search
        output_file: Path to output CSV file
        jobs: Number of parallel jobs
        batch_size: Number of items to batch before writing
        delimiter: Delimiter to use for prefix partitioning
        include_delete_markers: Whether to include delete markers
        include_versions: Whether to include non-delete-marker versions
        latest_only: Whether to include only the latest versions/markers
    """
    # Create a queue for communication
    item_queue = queue.Queue(maxsize=jobs * 2)  # Buffer some batches
    stop_event = threading.Event()
    total_items = [0]  # Use a list for mutable reference
    
    # Partition the prefix for parallel processing
    logger.info(f"Partitioning prefix '{prefix}' for parallel processing")
    prefixes = partition_prefix(bucket, prefix, delimiter, max_prefixes=jobs*10)
    logger.info(f"Found {len(prefixes)} prefixes to process")
    
    # Writer thread function
    def writer_thread():
        items_written = 0
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bucket', 'key', 'version_id', 'is_delete_marker', 'is_latest'])
            
            with tqdm(desc="Writing object versions", unit="items") as pbar:
                while not (stop_event.is_set() and item_queue.empty()):
                    try:
                        batch = item_queue.get(timeout=1)
                        if batch is None:  # Sentinel value
                            break
                            
                        writer.writerows(batch)
                        batch_size = len(batch)
                        items_written += batch_size
                        pbar.update(batch_size)
                        item_queue.task_done()
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error writing to file: {e}")
                
                total_items[0] = items_written
    
    # Start the writer thread
    writer = threading.Thread(target=writer_thread)
    writer.daemon = True
    writer.start()
    
    # Process a single prefix chunk
    def process_prefix_chunk(bucket: str, prefix: str, item_queue: queue.Queue, batch_size: int = 10000) -> int:
        """
        Process a single prefix chunk and put items into a queue.
        
        Returns:
            Count of items found
        """
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_object_versions')
        count = 0
        batch = []
        
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                # Process delete markers
                if include_delete_markers:
                    delete_markers = page.get('DeleteMarkers', [])
                    for marker in delete_markers:
                        # Skip non-latest versions if latest_only is True
                        if latest_only and not marker.get('IsLatest', False):
                            continue
                            
                        batch.append((
                            bucket,
                            marker['Key'],
                            marker['VersionId'],
                            True,  # is_delete_marker
                            marker.get('IsLatest', False)
                        ))
                        count += 1
                        
                        # If batch is full, put it in queue and reset
                        if len(batch) >= batch_size:
                            item_queue.put(batch)
                            batch = []
                
                # Process regular versions
                if include_versions:
                    versions = page.get('Versions', [])
                    for version in versions:
                        # Skip delete markers (they were handled above)
                        if version.get('IsDeleteMarker', False):
                            continue
                            
                        # Skip non-latest versions if latest_only is True
                        if latest_only and not version.get('IsLatest', False):
                            continue
                            
                        batch.append((
                            bucket,
                            version['Key'],
                            version['VersionId'],
                            False,  # is_delete_marker
                            version.get('IsLatest', False)
                        ))
                        count += 1
                        
                        # If batch is full, put it in queue and reset
                        if len(batch) >= batch_size:
                            item_queue.put(batch)
                            batch = []
            
            # Put any remaining items
            if batch:
                item_queue.put(batch)
                
            return count
        except Exception as e:
            logger.error(f"Error processing prefix {prefix}: {e}")
            if batch:
                item_queue.put(batch)
            return count
    
    # Process prefixes in parallel with progress tracking
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        # Submit all prefix chunks to executor
        futures = {
            executor.submit(
                process_prefix_chunk, 
                bucket, 
                chunk_prefix, 
                item_queue, 
                batch_size
            ): chunk_prefix for chunk_prefix in prefixes
        }
        
        # Track progress with tqdm
        with tqdm(total=len(prefixes), desc="Processing prefixes", unit="prefix") as pbar:
            for future in as_completed(futures):
                prefix_val = futures[future]
                try:
                    result = future.result()
                    logger.debug(f"Prefix '{prefix_val}': Found {result} items")
                except Exception as e:
                    logger.error(f"Error processing prefix '{prefix_val}': {e}")
                
                pbar.update(1)
    
    # Signal the writer to stop
    stop_event.set()
    item_queue.put(None)
    
    # Wait for writer to finish
    writer.join()
    
    if include_delete_markers and include_versions:
        logger.success(f"Completed finding all object versions and delete markers. Total found: {total_items[0]}")
    elif include_delete_markers:
        logger.success(f"Completed finding delete markers. Total found: {total_items[0]}")
    elif include_versions:
        logger.success(f"Completed finding object versions. Total found: {total_items[0]}")
    
    logger.info(f"Results written to {output_file}")


def export_to_excel(csv_dir: Path, output_file: Path, max_rows_per_sheet: int = 1000000,
                   min_size_gb: float = 1.0, min_objects: int = 10000) -> None:
    """
    Create an Excel spreadsheet from multiple CSV files in a directory.
    Each CSV file becomes a separate sheet in the Excel workbook.
    Sheets are ordered by the level number in the CSV filename.
    If a sheet would have more than max_rows_per_sheet rows, it is split into multiple sheets.
    
    Rows with size smaller than min_size_gb (in GB) and with fewer objects than min_objects
    will be skipped.

    Args:
        csv_dir: Directory containing the CSV files
        output_file: Path where the Excel file will be saved
        max_rows_per_sheet: Maximum number of rows per sheet (default: 1,000,000)
        min_size_gb: Minimum size in GB to include a row (default: 1.0)
        min_objects: Minimum number of objects to include a row (default: 10,000)
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for Excel export. Please install it with 'pip install pandas'")
        return

    # Find all CSV files in the directory
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to export to Excel")
    
    # Sort files by level number
    def extract_level(filename):
        # Example: "level_1_sizes.csv" -> 1
        parts = filename.name.split('_')
        if len(parts) >= 2 and parts[0] == "level":
            try:
                return int(parts[1])
            except ValueError:
                return float('inf')  # Place invalid formats at the end
        return float('inf')
    
    csv_files.sort(key=extract_level)
    
    # Create Excel writer
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for csv_file in tqdm(csv_files, desc="Exporting to Excel"):
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Filter rows based on size and object count
                if 'Size (GB)' in df.columns and 'File Count' in df.columns:
                    original_row_count = len(df)
                    df = df[(df['Size (GB)'] >= min_size_gb) | (df['File Count'] >= min_objects)]
                    filtered_row_count = len(df)
                    
                    if filtered_row_count < original_row_count:
                        logger.info(f"Filtered {original_row_count - filtered_row_count} rows from {csv_file.name} " 
                                   f"(min size: {min_size_gb} GB, min objects: {min_objects})")
                
                # Skip if no rows left after filtering
                if len(df) == 0:
                    logger.warning(f"Skipping {csv_file.name} - no rows left after filtering")
                    continue
                
                # Use level number as sheet name (e.g., "Level 1")
                level_num = extract_level(csv_file)
                if level_num != float('inf'):
                    base_sheet_name = f"Level {level_num}"
                else:
                    # Use filename (without extension) as fallback
                    base_sheet_name = csv_file.stem
                
                # Check if we need to split the sheet due to row limit
                row_count = len(df)
                if row_count > max_rows_per_sheet:
                    logger.info(f"Splitting sheet '{base_sheet_name}' into multiple sheets due to large size ({row_count:,} rows)")
                    
                    # Calculate number of sheets needed
                    num_sheets = (row_count + max_rows_per_sheet - 1) // max_rows_per_sheet
                    
                    # Split the dataframe into chunks and create a sheet for each
                    for i in range(num_sheets):
                        start_idx = i * max_rows_per_sheet
                        end_idx = min((i + 1) * max_rows_per_sheet, row_count)
                        
                        # Create sheet name with row range
                        range_sheet_name = f"{base_sheet_name} ({start_idx+1:,}-{end_idx:,})"
                        
                        # Truncate sheet name if too long (Excel has a 31 character limit)
                        if len(range_sheet_name) > 31:
                            # Use more compact format for large numbers
                            range_sheet_name = f"{base_sheet_name} ({(start_idx+1)//1000}K-{end_idx//1000}K)"
                            if len(range_sheet_name) > 31:
                                range_sheet_name = range_sheet_name[:31]
                        
                        # Extract the slice of dataframe for this sheet
                        df_slice = df.iloc[start_idx:end_idx]
                        
                        # Write to Excel
                        df_slice.to_excel(writer, sheet_name=range_sheet_name, index=False)
                        
                        # Auto-adjust column widths
                        worksheet = writer.sheets[range_sheet_name]
                        for idx, col in enumerate(df_slice.columns):
                            # Find the maximum length in the column
                            max_len = max(
                                df_slice[col].astype(str).apply(len).max(),  # max length of values
                                len(str(col))  # length of column name
                            ) + 2  # Add a little extra space
                            
                            # Set the column width
                            worksheet.column_dimensions[chr(65 + idx)].width = max_len
                else:
                    # Standard case - write the dataframe to a single sheet
                    df.to_excel(writer, sheet_name=base_sheet_name, index=False)
                    
                    # Auto-adjust column widths based on content
                    worksheet = writer.sheets[base_sheet_name]
                    for idx, col in enumerate(df.columns):
                        # Find the maximum length in the column
                        max_len = max(
                            df[col].astype(str).apply(len).max(),  # max length of values
                            len(str(col))  # length of column name
                        ) + 2  # Add a little extra space
                        
                        # Set the column width
                        worksheet.column_dimensions[chr(65 + idx)].width = max_len
        
        logger.success(f"Excel file has been created at {output_file}")
    
    except Exception as e:
        logger.error(f"Error creating Excel file: {e}")


def load_inventory_json(json_file_path):
    """Load and parse the inventory JSON file."""
    with open(json_file_path, 'r') as file:
        inventory_data = json.load(file)
    return inventory_data

def extract_filename(key):
    """Extract just the filename from a path."""
    return Path(key).name

def download_s3_files(inventory_data, download_dir):
    """
    Download files from S3 based on inventory data using a flat structure.
    
    Args:
        inventory_data: Parsed JSON inventory data
        download_dir: Directory to save downloaded files (Path object)
    """
    # Create download directory if it doesn't exist
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract destination bucket from ARN
    dest_bucket_arn = inventory_data["destinationBucket"]
    dest_bucket = dest_bucket_arn.split(":::")[-1]
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # Track progress
    total_files = len(inventory_data["files"])
    downloaded_files = 0
    failed_files = 0
    
    logger.info(f"Starting download of {total_files} files from bucket {dest_bucket}")
    logger.info(f"Saving to directory: {download_dir.absolute()}")
    
    # Download each file
    for i, file_info in enumerate(inventory_data["files"], 1):
        inventory_key = file_info["key"]
        size_bytes = file_info["size"]
        
        # Extract just the filename for a flat structure
        filename = extract_filename(inventory_key)
        
        # Create the local file path (flat structure)
        local_filepath = download_dir / filename
        
        logger.info(f"[{i}/{total_files}] Downloading {inventory_key} ({size_bytes/1024/1024:.2f} MB)...")
        
        try:
            # Download the file
            s3_client.download_file(dest_bucket, inventory_key, str(local_filepath))
            
            downloaded_files += 1
            logger.success(f"Successfully downloaded: {filename}")
            
        except botocore.exceptions.ClientError as e:
            logger.error(f"ERROR downloading {inventory_key}: {e}")
            logger.error(f"Attempted to get object '{inventory_key}' from bucket '{dest_bucket}'")
            failed_files += 1
    
    # Summary
    logger.info("\nDownload Summary:")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successfully downloaded: {downloaded_files}")
    logger.info(f"Failed: {failed_files}")
    
    if downloaded_files == total_files:
        logger.success("All files downloaded successfully!")
    else:
        logger.warning("Some files failed to download.")


@click.group()
def cli() -> None:
    """
    S3 Storage Analysis Tool - Process and visualize S3 inventory data
    """
    pass


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing CSV and Parquet files",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to save output files",
)
@click.option(
    "--max-depth",
    default=-1,
    show_default=True,
    type=int,
    help="Maximum directory depth to process. Use -1 for no limit",
)
@click.option(
    "--n-jobs",
    default=-1,
    show_default=True,
    type=int,
    help="Number of parallel jobs to run. Use -1 for all available cores",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
def analyze(input_dir: Path, output_dir: Path, max_depth: int, n_jobs: int, log_level: str) -> None:
    """
    Process AWS S3 Inventory CSV or Parquet files and aggregate storage by directory levels.

    This command reads AWS S3 Inventory CSV or Parquet files and aggregates storage sizes at
    directory levels, where level 1 is the top-most directory (e.g., slr0/, slr1/, slr2/)
    and each subsequent level goes one directory deeper, outputting results in GB.

    Parallel processing is used to speed up file analysis.
    """
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),  # Use tqdm.write to avoid breaking progress bars
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    logger.info(f"Starting S3 Storage Analysis with {n_jobs} workers")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Ensured output directory exists: {output_dir}")

    # Process CSV and Parquet files
    process_input_files(input_dir, output_dir, max_depth, n_jobs)

    logger.info("Analysis complete!")


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing CSV files to export to Excel",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path where the Excel file will be saved",
)
@click.option(
    "--max-rows",
    default=1000000,
    show_default=True,
    type=int,
    help="Maximum number of rows per sheet. Sheets with more rows will be split",
)
@click.option(
    "--min-size-gb",
    default=1.0,
    show_default=True,
    type=float,
    help="Skip rows with size smaller than this value (in GB)",
)
@click.option(
    "--min-objects",
    default=10000,
    show_default=True,
    type=int,
    help="Skip rows with fewer objects than this value",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
def excel_export(input_dir: Path, output_file: Path, max_rows: int, min_size_gb: float, min_objects: int, log_level: str) -> None:
    """
    Export CSV reports to a single Excel file with multiple sheets.
    
    This command takes CSV files generated by the analysis tool and combines them
    into a single Excel workbook, with each CSV file becoming a separate sheet.
    Sheets are ordered by level number.
    
    If a sheet would have more than the specified maximum number of rows (default: 1,000,000),
    it will be split into multiple sheets with appropriate range indicators.
    
    You can filter out small directories by setting minimum size (--min-size-gb) and/or
    minimum objects count (--min-objects) thresholds. Rows not meeting both criteria
    will be excluded from the Excel file.
    """
    # Configure logger level
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),  # Use tqdm.write to avoid breaking progress bars
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    
    logger.info(f"Starting export to Excel (max {max_rows:,} rows per sheet, min size: {min_size_gb} GB, min objects: {min_objects})")
    
    # Create parent directory of output file if it doesn't exist
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Export CSV files to Excel
    export_to_excel(input_dir, output_file, max_rows_per_sheet=max_rows, min_size_gb=min_size_gb, min_objects=min_objects)
    
    logger.info("Export complete!")

# @cli.command()
# @click.argument("csv_input", type=click.Path(exists=True, path_type=Path))
# @click.option("--jobs", "-j", default=30, show_default=True, type=int, help="Number of parallel jobs.")
# @click.option("--batch-size", default=1000, show_default=True, type=int, help="Batch size for deletion (max 1000).")
# @click.option("--throttle", default=0, show_default=True, type=float, help="Seconds to wait between batch API calls to avoid rate limiting.")
# @click.option("--skip-header/--no-skip-header", default=True, show_default=True, help="Skip the first line of CSV (header row).")
# @click.option("--dry-run", is_flag=True, help="Simulate deletion without actually deleting objects.")
# @click.option("--no-confirm", is_flag=True, help="Skip confirmation prompt and proceed with deletion.")
# @click.option(
#     "--log-level",
#     default="INFO",
#     show_default=True,
#     type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
#     help="Set the logging level.",
# )
# @click.option("--error-log", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), 
#               help="Path to save error log for failed deletions.")
# def delete(
#     csv_input: Path, 
#     jobs: int, 
#     batch_size: int, 
#     throttle: float, 
#     skip_header: bool, 
#     dry_run: bool, 
#     no_confirm: bool, 
#     log_level: str, 
#     error_log: Optional[Path]
# ) -> None:
#     """
#     Permanently delete objects from S3 buckets using a CSV input file.
    
#     The CSV file should contain one object per line with at least these columns:
#     bucket_name, prefix, object_version
    
#     Additional columns are allowed but will be ignored. The CSV file is typically
#     generated by AWS S3 Inventory and contains objects from a single bucket.
    
#     WARNING: This operation is IRREVERSIBLE. Use --dry-run first to verify
#     which objects will be deleted, and consider creating a bucket lifecycle
#     rule instead for safer management of object retention.
#     """
#     # Configure logger
#     logger.remove()
#     logger.add(
#         sink=lambda msg: tqdm.write(msg, end=""),
#         format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
#         level=log_level,
#     )
    
#     # Validate batch size
#     if batch_size > 1000:
#         logger.warning("Batch size cannot exceed 1000. Setting to 1000.")
#         batch_size = 1000
    
#     # Count lines efficiently
#     logger.info(f"Counting objects in {csv_input}...")
#     total_lines = count_file_lines(csv_input)

#     total_batches = total_lines // batch_size
#     logger.info(f"Total objects: {total_lines}, Batch size: {batch_size}, Total batches: {total_batches}")
    
#     if skip_header and total_lines > 0:
#         total_lines -= 1
    
#     # Skip further processing if there are no lines to process
#     if total_lines <= 0:
#         logger.warning("No objects found in the input file.")
#         return
    
#     # Start processing
#     logger.info(f"Found {total_lines} objects to process")
    
#     # Confirmation step
#     if dry_run:
#         logger.info(f"DRY RUN MODE: Would process {total_lines} objects")
#     else:
#         if not no_confirm:
#             if not click.confirm(f"Are you sure you want to permanently delete objects from the S3 paths in {csv_input}?", 
#                                default=True):
#                 logger.info("Deletion cancelled.")
#                 return
#         logger.info(f"Starting deletion of objects from {csv_input} using {jobs} parallel jobs.")
    
#     # Process CSV file in batches
#     all_failed_objects = []
#     bucket_object_counts = defaultdict(int)
    
#     with open(csv_input, "r") as file:
#         reader = csv.reader(file)
        
#         # Skip header if requested
#         if skip_header:
#             next(reader, None)
        
#         # First pass to count objects by bucket for better progress reporting
#         if dry_run:
#             # Sample the first few entries to show what would be deleted
#             logger.info("Dry run - showing sample of objects that would be deleted:")
#             sample_count = 0
#             bucket_samples = defaultdict(int)
            
#             for curr_bucket, batch in process_s3_deletion_batches(reader, batch_size):

#                 if sample_count <= 10:
#                     for obj in batch:
#                         logger.info(f"Would delete: s3://{curr_bucket}/{obj['Key']} (Version: {obj['VersionId']})")
#                         sample_count += 1
#                         if sample_count >= 10:
#                             break

#                     # Add to the sample count
#                     batch_size_actual = len(batch)
#                     bucket_samples[curr_bucket] += batch_size_actual
#                     sample_count += batch_size_actual
                        
#                 # Stop once we've shown enough samples
#                 if sample_count >= 10:
#                     break
                    
#             # Show summary of what would be deleted
#             total_objects = sum(bucket_samples.values())
            
#             logger.info("\nSummary of objects that would be deleted:")
#             for b, count in bucket_samples.items():
#                 bucket_percent = (count / total_objects) * 100 if total_objects > 0 else 0
#                 logger.info(f"  Bucket {b}: {count:,} objects ({bucket_percent:.1f}% of total)")
                
#             logger.info("\nDry run complete. Use --no-dry-run to perform actual deletion.")
                
#         else:
#             # Rewind file to start for actual processing
#             file.seek(0)
#             if skip_header:
#                 next(reader, None)
                
#             # Process in parallel using joblib
#             logger.info("Processing objects for deletion...")
            
#             # Use Parallel to process batches
#             batch_results = Parallel(n_jobs=jobs, backend="threading")(
#                 delayed(process_batch)(curr_bucket, batch, throttle)
#                 for curr_bucket, batch in tqdm(
#                     process_s3_deletion_batches(reader, batch_size),
#                     desc="Processing S3 Batches",
#                     mininterval=1,
#                     total=total_batches,
#                 )
#             )
            
#             # Process results
#             for curr_bucket, success_count, failed_objects in batch_results:
#                 bucket_object_counts[curr_bucket] += success_count
#                 all_failed_objects.extend([(curr_bucket, obj) for obj in failed_objects])
            
#             # Log results
#             total_deleted = sum(bucket_object_counts.values())
#             logger.info("\nDeletion Summary:")
#             logger.info(f"Total processed: {total_lines}, Successfully deleted: {total_deleted}, Failed: {len(all_failed_objects)}")
            
#             # Show per-bucket summary
#             for b, count in bucket_object_counts.items():
#                 logger.info(f"  Bucket {b}: {count} objects deleted")
            
#             # Save error log if requested
#             if error_log and all_failed_objects:
#                 with open(error_log, "w", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow(["Bucket", "Key", "VersionId", "Error Code", "Error Message"])
#                     for curr_bucket, obj in all_failed_objects:
#                         writer.writerow([
#                             curr_bucket,
#                             obj.get("Key", ""),
#                             obj.get("VersionId", ""),
#                             obj.get("Code", ""),
#                             obj.get("Message", "")
#                         ])
#                 logger.info(f"Error log saved to {error_log}")
                
#     logger.info("Operation complete.")


@cli.command()
@click.option("--bucket", required=True, help="S3 bucket to scan for delete markers")
@click.option("--prefix", default="", show_default=True, help="Optional prefix to limit the search scope")
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output CSV file path")
@click.option("--jobs", "-j", default=20, show_default=True, type=int, help="Number of parallel jobs")
@click.option("--batch-size", default=10000, show_default=True, type=int, help="Number of markers to buffer before writing")
@click.option("--delimiter", default="/", show_default=True, help="Delimiter to use for prefix partitioning")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
def find_deleted(
    bucket: str,
    prefix: str,
    output: Path,
    jobs: int,
    batch_size: int,
    delimiter: str,
    log_level: str
) -> None:
    """
    Find all objects with delete markers in an S3 bucket and write them to a CSV file.
    
    This command efficiently searches for delete markers in a versioned S3 bucket and
    writes the results to a CSV file with the format: bucket,key,version_id,is_latest
    
    For buckets with hundreds of millions of objects, this command uses:
    
    1. Intelligent prefix partitioning to enable parallel processing
    2. Multi-threaded S3 API calls for maximum throughput
    3. Buffered writing to optimize I/O performance
    4. Progress tracking for both prefix processing and output writing
    
    The output file can be used with the 'delete' command to permanently remove the
    delete markers, making the objects visible again or cleaning up the version history.
    """
    # Configure logger
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    
    logger.info(f"Starting to find delete markers in bucket '{bucket}' with prefix '{prefix}'")
    
    # Create parent directory of output file if it doesn't exist
    output.parent.mkdir(exist_ok=True, parents=True)
    
    # Run the optimized search and write function
    find_and_write_delete_markers(bucket, prefix, output, jobs, batch_size, delimiter)
    
    logger.info("Operation complete!")


@cli.command()
@click.option("--bucket", required=True, help="S3 bucket to scan")
@click.option("--prefix", default="", show_default=True, help="Optional prefix to limit the search scope")
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output CSV file path")
@click.option("--jobs", "-j", default=20, show_default=True, type=int, help="Number of parallel jobs")
@click.option("--batch-size", default=10000, show_default=True, type=int, help="Number of items to buffer before writing")
@click.option("--delimiter", default="/", show_default=True, help="Delimiter to use for prefix partitioning")
@click.option(
    "--include-delete-markers/--exclude-delete-markers", 
    default=True, 
    show_default=True, 
    help="Include delete markers in the output"
)
@click.option(
    "--include-versions/--exclude-versions", 
    default=True, 
    show_default=True, 
    help="Include non-delete-marker versions in the output"
)
@click.option(
    "--latest-only/--all-versions", 
    default=False, 
    show_default=True, 
    help="Include only the latest version of each object"
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
def find_versions(
    bucket: str,
    prefix: str,
    output: Path,
    jobs: int,
    batch_size: int,
    delimiter: str,
    include_delete_markers: bool,
    include_versions: bool,
    latest_only: bool,
    log_level: str
) -> None:
    """
    Find all object versions and/or delete markers in an S3 bucket and write them to a CSV file.
    
    This command efficiently searches for object versions and delete markers in a versioned S3 bucket
    and writes the results to a CSV file with the format:
    bucket,key,version_id,is_delete_marker,is_latest
    
    For buckets with hundreds of millions of objects, this command uses:
    
    1. Intelligent prefix partitioning to enable parallel processing
    2. Multi-threaded S3 API calls for maximum throughput
    3. Buffered writing to optimize I/O performance
    4. Progress tracking for both prefix processing and output writing
    
    The output file can be used with the 'delete' command to permanently remove objects,
    delete markers, or specific versions.
    
    Examples:
    
    # Find all object versions and delete markers
    sic.py find-versions --bucket my-bucket --output all_versions.csv
    
    # Find only delete markers (same as find-deleted)
    sic.py find-versions --bucket my-bucket --output delete_markers.csv --exclude-versions
    
    # Find only the latest version or delete marker of each object
    sic.py find-versions --bucket my-bucket --output latest.csv --latest-only
    
    # Find all versions (excluding delete markers)
    sic.py find-versions --bucket my-bucket --output versions.csv --exclude-delete-markers
    """
    # Configure logger
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    
    # Show what we're going to do
    actions = []
    if include_delete_markers and include_versions:
        actions.append("all object versions and delete markers")
    elif include_delete_markers:
        actions.append("delete markers")
    elif include_versions:
        actions.append("object versions")
    
    if latest_only:
        actions.append("(latest versions only)")
    
    logger.info(f"Starting to find {' '.join(actions)} in bucket '{bucket}' with prefix '{prefix}'")
    
    # Create parent directory of output file if it doesn't exist
    output.parent.mkdir(exist_ok=True, parents=True)
    
    # Run the search and write function
    find_and_write_all_versions(
        bucket, 
        prefix, 
        output, 
        jobs, 
        batch_size, 
        delimiter,
        include_delete_markers,
        include_versions,
        latest_only
    )
    
    logger.info("Operation complete!")


@cli.command()
@click.option('--json', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to the inventory JSON file')
@click.option('--dir', type=click.Path(path_type=Path), 
              help='Directory to save downloaded files')
def download_from_manifest(json, dir):
    """Download files from AWS S3 using an inventory JSON file with a flat output structure."""
    try:
        # Load inventory data
        inventory_data = load_inventory_json(json)
        
        # Download files
        download_s3_files(inventory_data, dir)
    except Exception as e:
        logger.error(f"Error: {e}")


@cli.command()
@click.option("--bucket", required=True, help="S3 bucket containing the prefix to list")
@click.option("--prefix", required=True, help="Prefix to list (without leading/trailing slash)")
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output CSV file path")
@click.option("--include-versions", is_flag=True, default=False, help="Include all versions if bucket has versioning enabled")
@click.option("--include-size", is_flag=True, default=False, help="Include object size information")
@click.option("--max-keys", default=1000, show_default=True, type=int, help="Number of keys to fetch per API call")
@click.option("--storage-class", 
    type=click.Choice(["STANDARD", "REDUCED_REDUNDANCY", "STANDARD_IA", "ONEZONE_IA", 
    "INTELLIGENT_TIERING", "GLACIER", "DEEP_ARCHIVE", "GLACIER_IR"]),
    help="Filter objects by storage class")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
def list_prefix(
    bucket: str,
    prefix: str,
    output: Path,
    include_versions: bool,
    include_size: bool,
    max_keys: int,
    storage_class: Optional[str],
    log_level: str,
) -> None:
    """
    List all objects under a specified prefix in an S3 bucket and save to CSV.
    
    This command lists all objects under the given prefix and saves them to a CSV file.
    If --include-versions is specified, it will list all versions of each object 
    (only relevant when bucket versioning is enabled).
    
    Use --storage-class to filter objects by their storage class.
    
    The CSV file can be used with the 'delete-prefix' command to delete specific objects.
    
    Examples:
    
    # List current versions of all objects under a prefix
    sic.py list-prefix --bucket my-bucket --prefix folder/to/list --output objects.csv
    
    # List all versions of objects under a prefix
    sic.py list-prefix --bucket my-bucket --prefix folder/to/list --output objects.csv --include-versions
    
    # List objects with size information
    sic.py list-prefix --bucket my-bucket --prefix folder/to/list --output objects.csv --include-size
    
    # List only STANDARD storage class objects
    sic.py list-prefix --bucket my-bucket --prefix folder/to/list --output objects.csv --storage-class STANDARD
    """
    # Configure logger
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    
    # Normalize prefix (ensure no leading slash, but add trailing slash if not empty)
    if prefix.startswith('/'):
        prefix = prefix[1:]
    if prefix and not prefix.endswith('/') and not prefix.endswith('*'):
        prefix = prefix + '/'
        
    # Create parent directory of output file if it doesn't exist
    output.parent.mkdir(exist_ok=True, parents=True)
    
    s3 = boto3.client('s3')
    
    logger.info(f"Listing objects under prefix '{prefix}' in bucket '{bucket}'...")
    
    try:
        # Define CSV headers based on options
        if include_versions:
            if include_size:
                headers = ['bucket', 'key', 'version_id', 'is_latest', 'is_delete_marker', 'size_bytes', 'last_modified', 'storage_class']
            else:
                headers = ['bucket', 'key', 'version_id', 'is_latest', 'is_delete_marker', 'storage_class']
        else:
            if include_size:
                headers = ['bucket', 'key', 'size_bytes', 'last_modified', 'storage_class']
            else:
                headers = ['bucket', 'key', 'storage_class']
        
        # Write to CSV file
        object_count = 0
        skipped_count = 0
        
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # Use appropriate pagination based on whether we're listing versions or not
            if include_versions:
                # List all versions of objects
                paginator = s3.get_paginator('list_object_versions')
                
                for page in tqdm(paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys), 
                                desc="Listing object versions", unit="batch"):
                    # Process versions
                    for version in page.get('Versions', []):
                        # Check storage class if filter is specified
                        curr_storage_class = version.get('StorageClass', 'STANDARD')
                        if storage_class and curr_storage_class != storage_class:
                            skipped_count += 1
                            continue
                            
                        row = [
                            bucket,
                            version['Key'],
                            version['VersionId'],
                            version.get('IsLatest', False),
                            False  # not a delete marker
                        ]
                        
                        if include_size:
                            row.extend([
                                version.get('Size', 0),
                                version.get('LastModified', '').isoformat() if hasattr(version.get('LastModified', ''), 'isoformat') else version.get('LastModified', '')
                            ])
                        
                        # Always include storage class
                        row.append(curr_storage_class)
                        
                        writer.writerow(row)
                        object_count += 1
                    
                    # Process delete markers
                    for marker in page.get('DeleteMarkers', []):
                        # Delete markers don't have a storage class, skip if filtering by storage class
                        if storage_class:
                            skipped_count += 1
                            continue
                            
                        row = [
                            bucket,
                            marker['Key'],
                            marker['VersionId'],
                            marker.get('IsLatest', False),
                            True  # is a delete marker
                        ]
                        
                        if include_size:
                            # Delete markers don't have size, but we keep the CSV structure consistent
                            row.extend([
                                0,  # size is 0 for delete markers
                                marker.get('LastModified', '').isoformat() if hasattr(marker.get('LastModified', ''), 'isoformat') else marker.get('LastModified', '')
                            ])
                        
                        # Delete markers don't have a storage class    
                        row.append("N/A")
                        
                        writer.writerow(row)
                        object_count += 1
            else:
                # List only current versions of objects
                paginator = s3.get_paginator('list_objects_v2')
                
                for page in tqdm(paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys),
                                desc="Listing objects", unit="batch"):
                    for obj in page.get('Contents', []):
                        # Check storage class if filter is specified
                        curr_storage_class = obj.get('StorageClass', 'STANDARD')
                        if storage_class and curr_storage_class != storage_class:
                            skipped_count += 1
                            continue
                            
                        row = [
                            bucket,
                            obj['Key']
                        ]
                        
                        if include_size:
                            row.extend([
                                obj.get('Size', 0),
                                obj.get('LastModified', '').isoformat() if hasattr(obj.get('LastModified', ''), 'isoformat') else obj.get('LastModified', '')
                            ])
                        
                        # Always include storage class
                        row.append(curr_storage_class)
                        
                        writer.writerow(row)
                        object_count += 1
        
        version_str = "versions and delete markers" if include_versions else "objects"
        storage_class_str = f" with storage class '{storage_class}'" if storage_class else ""
        
        logger.success(f"Listed {object_count} {version_str}{storage_class_str} under prefix '{prefix}' in bucket '{bucket}'")
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} objects that didn't match the storage class filter")
        logger.info(f"Results saved to {output}")
        
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        error_message = e.response.get('Error', {}).get('Message')
        logger.error(f"AWS Error: {error_code} - {error_message}")
    except Exception as e:
        logger.error(f"Error: {e}")


@cli.command()
@click.option("--input-file", required=True, type=click.Path(exists=True, path_type=Path), help="Input CSV file with objects to delete")
@click.option("--batch-size", default=1000, show_default=True, type=int, help="Batch size for deletion (max 1000)")
@click.option("--throttle", default=0, show_default=True, type=float, help="Seconds to wait between batch API calls to avoid rate limiting")
@click.option("--dry-run", is_flag=True, help="Simulate deletion without actually deleting objects")
@click.option("--no-confirm", is_flag=True, help="Skip confirmation prompt and proceed with deletion")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
@click.option("--error-log", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), 
              help="Path to save error log for failed deletions")
def delete_prefix(
    input_file: Path,
    batch_size: int,
    throttle: float,
    dry_run: bool,
    no_confirm: bool,
    log_level: str,
    error_log: Optional[Path],
) -> None:
    """
    Delete objects listed in a CSV file (generated by list-prefix command).
    
    This command reads a CSV file containing object information and deletes
    the objects in batches. The CSV file should be in the format produced
    by the 'list-prefix' command.
    
    WARNING: This operation is IRREVERSIBLE. Use --dry-run first to verify
    which objects will be deleted, and consider creating a bucket lifecycle
    rule instead for safer management of object retention.
    
    Examples:
    
    # Dry run to see what would be deleted
    sic.py delete-prefix --input-file objects.csv --dry-run
    
    # Delete objects listed in the CSV file
    sic.py delete-prefix --input-file objects.csv
    
    # Delete with custom batch size and throttling
    sic.py delete-prefix --input-file objects.csv --batch-size 500 --throttle 0.5
    """
    # Configure logger
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    
    # Validate batch size
    if batch_size > 1000:
        logger.warning("Batch size cannot exceed 1000. Setting to 1000.")
        batch_size = 1000
    
    # Read the input file to count objects and determine format
    logger.info(f"Reading objects from {input_file}...")
    
    try:
        with open(input_file, 'r', newline='') as f:
            reader = csv.reader(f)
            
            # Read header to determine file format
            headers = next(reader, [])
            
            # Check if file is valid
            if 'bucket' not in headers or 'key' not in headers:
                logger.error("Invalid input file format. File must have 'bucket' and 'key' columns.")
                return
                
            # Determine if file contains version information
            has_versions = 'version_id' in headers
            
            # Count total objects for progress tracking
            rows = list(reader)
            total_objects = len(rows)
            
            if total_objects == 0:
                logger.warning("No objects found in the input file.")
                return
                
            # Get bucket name(s) and sample some objects for display
            buckets = set()
            sample_objects = []
            
            for i, row in enumerate(rows[:5]):
                bucket_idx = headers.index('bucket')
                key_idx = headers.index('key')
                
                if len(row) > max(bucket_idx, key_idx):
                    buckets.add(row[bucket_idx])
                    sample_objects.append((row[bucket_idx], row[key_idx]))
            
            # Log information about objects to be deleted
            if has_versions:
                logger.info(f"Found {total_objects} object versions to delete from {len(buckets)} bucket(s)")
            else:
                logger.info(f"Found {total_objects} objects to delete from {len(buckets)} bucket(s)")
                
            # Show sample of objects
            if sample_objects:
                logger.info("\nSample of objects that would be deleted:")
                for bucket, key in sample_objects:
                    logger.info(f"  s3://{bucket}/{key}")
                
                if len(sample_objects) < total_objects:
                    logger.info(f"  ... and {total_objects - len(sample_objects)} more")
            
            # Dry run - stop here
            if dry_run:
                logger.info("\nDRY RUN - no objects will be deleted.")
                return
                
            # Confirmation step
            if not no_confirm:
                version_str = "object versions" if has_versions else "objects"
                confirm_message = f"Are you sure you want to permanently delete {total_objects} {version_str}?"
                if not click.confirm(confirm_message, default=False):
                    logger.info("Deletion cancelled.")
                    return
            
            # Initialize for tracking results
            total_deleted = 0
            total_failed = 0
            all_failed_objects = []
            
            # Group objects by bucket for more efficient deletion
            bucket_objects = defaultdict(list)
            
            # Re-read the file for processing
            with open(input_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                bucket_idx = headers.index('bucket')
                key_idx = headers.index('key')
                version_id_idx = headers.index('version_id') if has_versions else -1
                
                for row in reader:
                    if len(row) > max(bucket_idx, key_idx):
                        bucket = row[bucket_idx]
                        key = row[key_idx]
                        
                        obj = {'Key': key}
                        
                        # Add version ID if available
                        if has_versions and version_id_idx >= 0 and len(row) > version_id_idx:
                            version_id = row[version_id_idx]
                            if version_id and version_id.lower() != 'null':
                                obj['VersionId'] = version_id
                        
                        bucket_objects[bucket].append(obj)
            
            # Process deletions in batches for each bucket
            with tqdm(total=total_objects, desc="Deleting objects", unit="obj") as progress:
                for bucket, objects in bucket_objects.items():
                    # Process in batches
                    for i in range(0, len(objects), batch_size):
                        batch = objects[i:i+batch_size]
                        
                        try:
                            success, failed = delete_s3_objects_batch(bucket, batch, throttle)
                            total_deleted += success
                            total_failed += len(failed)
                            
                            # Add failed objects to the list with bucket info
                            for obj in failed:
                                all_failed_objects.append((bucket, obj))
                                
                            # Update progress
                            progress.update(len(batch))
                            
                        except Exception as e:
                            logger.error(f"Error deleting batch from bucket {bucket}: {e}")
                            # Assume all objects in the batch failed
                            total_failed += len(batch)
                            progress.update(len(batch))
                
            # Log summary
            logger.info("\nDeletion Summary:")
            logger.info(f"Total objects processed: {total_objects}")
            logger.info(f"Successfully deleted: {total_deleted}")
            
            if total_failed > 0:
                logger.warning(f"Failed to delete: {total_failed}")
                
                # Save error log if requested
                if error_log and all_failed_objects:
                    with open(error_log, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Bucket", "Key", "VersionId", "Error Code", "Error Message"])
                        for bucket, obj in all_failed_objects:
                            writer.writerow([
                                bucket,
                                obj.get("Key", ""),
                                obj.get("VersionId", ""),
                                obj.get("Code", ""),
                                obj.get("Message", "")
                            ])
                    logger.info(f"Error log saved to {error_log}")
            
            logger.info("Operation complete.")
    
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        error_message = e.response.get('Error', {}).get('Message')
        logger.error(f"AWS Error: {error_code} - {error_message}")
    except Exception as e:
        logger.error(f"Error: {e}")


@cli.command()
@click.option("--bucket", required=True, help="S3 bucket containing objects to move")
@click.option("--prefix", default="", help="Prefix to limit the objects to move")
@click.option("--storage-class", required=True, type=click.Choice([
    "STANDARD", "REDUCED_REDUNDANCY", "STANDARD_IA", "ONEZONE_IA", 
    "INTELLIGENT_TIERING", "GLACIER", "DEEP_ARCHIVE", "GLACIER_IR"]), 
    help="Target storage class")
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), 
    help="Optional CSV file with objects to move (format from list-prefix command)")
@click.option("--batch-size", default=1000, show_default=True, type=int, help="Batch size for operations (max 1000)")
@click.option("--min-size", type=int, default=0, help="Minimum object size in bytes to move")
@click.option("--max-size", type=int, default=None, help="Maximum object size in bytes to move")
@click.option("--exclude-prefix", multiple=True, help="Prefixes to exclude (can be used multiple times)")
@click.option("--throttle", default=0, show_default=True, type=float, 
    help="Seconds to wait between batch API calls to avoid rate limiting")
@click.option("--handle-versions", type=click.Choice(["keep", "delete"], case_sensitive=False),
    default="keep", show_default=True, help="How to handle old versions in versioned buckets")
@click.option("--dry-run", is_flag=True, help="Simulate the operation without actually moving objects")
@click.option("--no-confirm", is_flag=True, help="Skip confirmation prompt and proceed")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
@click.option("--output-file", type=click.Path(dir_okay=False, path_type=Path), 
    help="Path to save results of the operation")
def move_storage_class(
    bucket: str,
    prefix: str,
    storage_class: str,
    input_file: Optional[Path],
    batch_size: int,
    min_size: int,
    max_size: Optional[int],
    exclude_prefix: List[str],
    throttle: float,
    handle_versions: str,
    dry_run: bool,
    no_confirm: bool,
    log_level: str,
    output_file: Optional[Path],
) -> None:
    """
    Move objects to a different storage class while keeping prefixes unchanged.
    
    This command changes the storage class of objects either:
    1. All objects under a prefix in a bucket, or
    2. Specific objects listed in a CSV file
    
    For versioned buckets, you can choose how to handle old versions using the --handle-versions option:
    - keep: Keep the old versions (default)
    - delete: Delete the old versions after changing storage class (saves storage costs)
    
    Objects keep their original keys/prefixes but are moved to the specified storage class.
    Filters allow excluding certain objects based on size or prefix patterns.
    
    Examples:
    
    # Move all objects under a prefix to GLACIER storage class and delete old versions
    sic.py move-storage-class --bucket my-bucket --prefix logs/2023/ --storage-class GLACIER --handle-versions delete
    
    # Move objects from a CSV file to STANDARD_IA storage class, keeping old versions
    sic.py move-storage-class --bucket my-bucket --input-file objects.csv --storage-class STANDARD_IA --handle-versions keep
    
    # Move objects larger than 128MB to DEEP_ARCHIVE, only advising on lifecycle rules
    sic.py move-storage-class --bucket my-bucket --prefix data/ --storage-class DEEP_ARCHIVE --min-size 134217728 --handle-versions advise
    """
    # Configure logger
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    
    # Validate batch size
    if batch_size > 1000:
        logger.warning("Batch size cannot exceed 1000. Setting to 1000.")
        batch_size = 1000
    
    # Normalize prefix (ensure no leading slash)
    if prefix and prefix.startswith('/'):
        prefix = prefix[1:]
    
    # Create S3 client
    s3 = boto3.client('s3')
    
    # Check if output file path is valid if specified
    if output_file:
        output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Function to check if object should be excluded based on prefix
    def should_exclude(key):
        for excl in exclude_prefix:
            if key.startswith(excl):
                return True
        return False
    
    # Function to move a batch of objects to the new storage class
    @backoff.on_exception(
        backoff.expo,
        botocore.exceptions.ClientError,
        max_time=300,  # Maximum of 5 minutes total retry time
        giveup=lambda e: getattr(e, 'response', {}).get('Error', {}).get('Code') 
                in ['NoSuchKey', 'NoSuchBucket', 'NotFound', '404'],
        on_backoff=lambda details: logger.warning(
            f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries "
            f"calling S3 copy_object for bucket {bucket}"
        ),
        on_giveup=lambda details: logger.error(
            f"Giving up on S3 copy_object for bucket {bucket} after {details['tries']} tries"
        ),
        base=2,
        factor=5,
        jitter=backoff.full_jitter,
    )
    def move_batch(objects):
        """
        Move a batch of objects to the new storage class.
        
        Args:
            objects: List of objects to move, each with Key and (optionally) Size
            
        Returns:
            Tuple of (success_count, list_of_failed)
        """
        success_count = 0
        failed_objects = []
        
        # Process each object in the batch
        for obj in objects:
            key = obj['Key']
            
            # Skip if should be excluded based on exclude_prefix
            if should_exclude(key):
                logger.debug(f"Skipping excluded object: {key}")
                continue
            
            # Skip if doesn't meet size criteria
            size = obj.get('Size')  # May be None if not provided
            if size is not None:
                if size < min_size:
                    logger.debug(f"Skipping {key} - too small ({size} bytes < {min_size} bytes)")
                    continue
                if max_size is not None and size > max_size:
                    logger.debug(f"Skipping {key} - too large ({size} bytes > {max_size} bytes)")
                    continue
            
            # Skip in dry run mode
            if dry_run:
                logger.debug(f"Would move: {key} to {storage_class}")
                success_count += 1
                continue
            
            try:
                # Copy the object to itself with new storage class
                s3.copy_object(
                    Bucket=bucket,
                    CopySource={'Bucket': bucket, 'Key': key},
                    Key=key,
                    StorageClass=storage_class,
                    MetadataDirective='COPY'  # Keep all metadata
                )
                success_count += 1
                
                # Add small delay if throttling is enabled
                if throttle > 0:
                    time.sleep(throttle)
            
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                error_message = e.response.get('Error', {}).get('Message')
                
                # Check if object is already in target storage class
                if error_code == 'InvalidStorageClass':
                    logger.debug(f"Object {key} is already in {storage_class} storage class")
                    success_count += 1  # Count as success since it's already in the target state
                else:
                    logger.warning(f"Failed to move {key}: {error_code} - {error_message}")
                    failed_objects.append({
                        'Key': key,
                        'Error': f"{error_code}: {error_message}"
                    })
            
            except Exception as e:
                logger.warning(f"Error moving {key}: {e}")
                failed_objects.append({
                    'Key': key,
                    'Error': str(e)
                })
        
        return success_count, failed_objects
    
    try:
        # Variables to track results
        objects_to_move = []
        total_processed = 0
        total_success = 0
        total_failed = 0
        total_skipped = 0
        
        # If input file is provided, read objects from CSV
        if input_file:
            logger.info(f"Reading objects from {input_file}...")
            
            with open(input_file, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                
                # Check if file is valid
                if 'bucket' not in headers or 'key' not in headers:
                    logger.error("Invalid input file format. File must have 'bucket' and 'key' columns.")
                    return
                
                # Determine indexes for relevant columns
                bucket_idx = headers.index('bucket')
                key_idx = headers.index('key')
                size_idx = headers.index('size_bytes') if 'size_bytes' in headers else -1
                
                # Read object data
                objects = []
                for row in reader:
                    if len(row) > max(bucket_idx, key_idx):
                        row_bucket = row[bucket_idx]
                        
                        # Skip objects from other buckets
                        if row_bucket != bucket:
                            logger.debug(f"Skipping object from different bucket: {row_bucket}/{row[key_idx]}")
                            total_skipped += 1
                            continue
                        
                        obj = {'Key': row[key_idx]}
                        
                        # Add size if available
                        if size_idx >= 0 and len(row) > size_idx:
                            try:
                                obj['Size'] = int(row[size_idx])
                            except (ValueError, TypeError):
                                pass  # Ignore if size can't be parsed
                        
                        objects.append(obj)
                
                total_processed = len(objects)
                objects_to_move = objects
                
                logger.info(f"Found {total_processed} objects in the input file")
        
        # If no input file but prefix is provided, list objects from S3
        elif prefix is not None:
            logger.info(f"Listing objects under prefix '{prefix}' in bucket '{bucket}'...")
            
            # List objects from S3
            paginator = s3.get_paginator('list_objects_v2')
            
            # Sample some keys for display
            sample_keys = []
            objects = []
            
            for page in tqdm(paginator.paginate(Bucket=bucket, Prefix=prefix),
                            desc="Listing objects", unit="batch"):
                for obj in page.get('Contents', []):
                    objects.append({
                        'Key': obj['Key'],
                        'Size': obj['Size'],
                        'StorageClass': obj.get('StorageClass', 'STANDARD')
                    })
                    
                    # Add to sample if we haven't reached 5 yet
                    if len(sample_keys) < 5:
                        sample_keys.append(obj['Key'])
            
            total_processed = len(objects)
            objects_to_move = objects
            
            logger.info(f"Found {total_processed} objects under prefix '{prefix}' in bucket '{bucket}'")
        
        else:
            logger.error("Either --prefix or --input-file must be specified")
            return
        
        # Check if there are objects to process
        if not objects_to_move:
            logger.warning("No objects found to move")
            return
        
        # Show sample of objects if available
        if len(objects_to_move) > 0:
            sample_size = min(5, len(objects_to_move))
            logger.info("\nSample of objects that would be moved to storage class: " + storage_class)
            for i in range(sample_size):
                obj = objects_to_move[i]
                size_str = f" ({obj['Size']/1024/1024:.2f} MB)" if 'Size' in obj else ""
                logger.info(f"  {obj['Key']}{size_str}")
            
            if len(objects_to_move) > sample_size:
                logger.info(f"  ... and {len(objects_to_move) - sample_size} more")
        
        # In dry run mode, just simulate
        if dry_run:
            logger.info(f"\nDRY RUN: Would move {len(objects_to_move)} objects to {storage_class} storage class")
            return
        
        # Confirm before proceeding
        if not no_confirm:
            if not click.confirm(f"Are you sure you want to move {len(objects_to_move)} objects to {storage_class} storage class?", 
                             default=False):
                logger.info("Operation cancelled.")
                return
        
        # Process objects in batches
        logger.info(f"Moving objects to {storage_class} storage class...")
        
        # Prepare for batched processing and results tracking
        all_failed_objects = []
        
        # Process in batches
        with tqdm(total=len(objects_to_move), desc=f"Moving to {storage_class}", unit="obj") as progress:
            for i in range(0, len(objects_to_move), batch_size):
                batch = objects_to_move[i:i+batch_size]
                
                # Process the batch
                success, failed = move_batch(batch)
                
                # Update counters
                total_success += success
                total_failed += len(failed)
                
                # Add failed objects to the list
                all_failed_objects.extend(failed)
                
                # Update progress
                progress.update(len(batch))
        
        # Generate summary
        logger.info("\nStorage Class Migration Summary:")
        logger.info(f"Total objects processed: {total_processed}")
        logger.info(f"Successfully moved to {storage_class}: {total_success}")
        
        if total_skipped > 0:
            logger.info(f"Skipped: {total_skipped}")
        
        if total_failed > 0:
            logger.warning(f"Failed: {total_failed}")
        
        # Write output file if specified
        if output_file and all_failed_objects:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Key', 'Error'])
                for obj in all_failed_objects:
                    writer.writerow([obj['Key'], obj.get('Error', 'Unknown error')])
            logger.info(f"Failed objects list saved to {output_file}")
        
        logger.info("Operation complete.")
    
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        error_message = e.response.get('Error', {}).get('Message')
        logger.error(f"AWS Error: {error_code} - {error_message}")
    except Exception as e:
        logger.error(f"Error: {e}")
  

if __name__ == "__main__":
    cli()