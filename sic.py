#!/usr/bin/env python3

import csv
import click
import boto3
import time
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from joblib import Parallel, delayed
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
        if not b: break
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
    """Delete a batch of S3 objects.
    
    Args:
        bucket: S3 bucket name
        objects: List of objects to delete (each with Key and VersionId)
        throttle: Seconds to wait between API calls
        
    Returns:
        Tuple of (success_count, failed_objects)
    """
    s3 = boto3.client('s3')
    failed_objects = []
    
    try:
        response = s3.delete_objects(
            Bucket=bucket, 
            Delete={'Objects': objects, 'Quiet': False}
        )
        
        # Track any errors that occurred
        if 'Errors' in response and response['Errors']:
            for error in response['Errors']:
                failed_objects.append({
                    'Key': error.get('Key', 'Unknown'),
                    'VersionId': error.get('VersionId', 'Unknown'),
                    'Code': error.get('Code', 'Unknown'),
                    'Message': error.get('Message', 'Unknown')
                })
                
        # Apply throttling if requested
        if throttle > 0:
            time.sleep(throttle)
            
        return len(objects) - len(failed_objects), failed_objects
    
    except Exception as e:
        logger.error(f"Failed batch delete in bucket {bucket}: {e}")
        # Return all objects as failed in case of exception
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


def process_csv_file(csv_file: Path, max_depth: int) -> Dict[int, Dict[str, Tuple[int, int]]]:
    """
    Process a single CSV file and return the aggregated sizes and file counts.

    Args:
        csv_file: Path to the CSV file
        max_depth: Maximum directory depth to consider

    Returns:
        Dictionary of {level: {directory: (size_bytes, file_count)}}
    """
    # Dictionary to store aggregated sizes for each level
    # For each directory, we store a tuple of (total_size, file_count)
    level_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    try:
        with open(csv_file, "r", newline="") as f:
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
                        raise ValueError(f"Invalid number of rows {len(row)} in {csv_file}: {row}")
                except (IndexError, ValueError):
                    logger.error(f"Error processing row in CSV file {csv_file}: {row}")
                    pass
    except Exception as e:
        logger.error(f"Error processing file {csv_file}: {e}")

    # Convert the defaultdict values to tuples for better serialization
    result = {}
    for level, dirs in level_data.items():
        result[level] = {dir_path: tuple(data) for dir_path, data in dirs.items()}

    return result


def process_files_in_batches(
    csv_files: List[Path], max_depth: int, n_jobs: int, batch_size: int = 10
) -> List[Dict[int, Dict[str, Tuple[int, int]]]]:
    """
    Process files in batches with a progress bar.

    Args:
        csv_files: List of CSV files to process
        max_depth: Maximum directory depth to consider
        n_jobs: Number of parallel jobs
        batch_size: Size of batches for parallel processing

    Returns:
        List of results from all files
    """
    results = []

    with tqdm(total=len(csv_files), desc="Processing CSV files") as pbar:
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i : i + batch_size]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_csv_file)(csv_file, max_depth) for csv_file in batch_files
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


def process_csv_files(input_dir: Path, output_dir: Path, max_depth: int, n_jobs: int) -> None:
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

    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files to process")

    # Process files in parallel with tqdm progress bar
    results = process_files_in_batches(csv_files, max_depth, n_jobs)

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


def export_to_excel(csv_dir: Path, output_file: Path, max_rows_per_sheet: int = 1000000) -> None:
    """
    Create an Excel spreadsheet from multiple CSV files in a directory.
    Each CSV file becomes a separate sheet in the Excel workbook.
    Sheets are ordered by the level number in the CSV filename.
    If a sheet would have more than max_rows_per_sheet rows, it is split into multiple sheets.

    Args:
        csv_dir: Directory containing the CSV files
        output_file: Path where the Excel file will be saved
        max_rows_per_sheet: Maximum number of rows per sheet (default: 1,000,000)
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
    help="Directory containing CSV files",
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
    Process AWS S3 Inventory CSV files and aggregate storage by directory levels.

    This command reads AWS S3 Inventory CSV files and aggregates storage sizes at
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

    # Process CSV files
    process_csv_files(input_dir, output_dir, max_depth, n_jobs)

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
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
def excel_export(input_dir: Path, output_file: Path, max_rows: int, log_level: str) -> None:
    """
    Export CSV reports to a single Excel file with multiple sheets.
    
    This command takes CSV files generated by the analysis tool and combines them
    into a single Excel workbook, with each CSV file becoming a separate sheet.
    Sheets are ordered by level number.
    
    If a sheet would have more than the specified maximum number of rows (default: 1,000,000),
    it will be split into multiple sheets with appropriate range indicators.
    """
    # Configure logger level
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),  # Use tqdm.write to avoid breaking progress bars
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    
    logger.info(f"Starting export to Excel (max {max_rows:,} rows per sheet)")
    
    # Create parent directory of output file if it doesn't exist
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Export CSV files to Excel
    export_to_excel(input_dir, output_file, max_rows_per_sheet=max_rows)
    
    logger.info("Export complete!")


@cli.command()
@click.argument("csv_input", type=click.Path(exists=True, path_type=Path))
@click.option("--jobs", "-j", default=30, show_default=True, type=int, help="Number of parallel jobs.")
@click.option("--batch-size", default=1000, show_default=True, type=int, help="Batch size for deletion (max 1000).")
@click.option("--throttle", default=0, show_default=True, type=float, help="Seconds to wait between batch API calls to avoid rate limiting.")
@click.option("--skip-header/--no-skip-header", default=True, show_default=True, help="Skip the first line of CSV (header row).")
@click.option("--dry-run", is_flag=True, help="Simulate deletion without actually deleting objects.")
@click.option("--no-confirm", is_flag=True, help="Skip confirmation prompt and proceed with deletion.")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level.",
)
@click.option("--error-log", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), 
              help="Path to save error log for failed deletions.")
def delete(
    csv_input: Path, 
    jobs: int, 
    batch_size: int, 
    throttle: float, 
    skip_header: bool, 
    dry_run: bool, 
    no_confirm: bool, 
    log_level: str, 
    error_log: Optional[Path]
) -> None:
    """
    Permanently delete objects from S3 buckets using a CSV input file.
    
    The CSV file should contain one object per line with at least these columns:
    bucket_name, prefix, object_version
    
    Additional columns are allowed but will be ignored. The CSV file is typically
    generated by AWS S3 Inventory and contains objects from a single bucket.
    
    WARNING: This operation is IRREVERSIBLE. Use --dry-run first to verify
    which objects will be deleted, and consider creating a bucket lifecycle
    rule instead for safer management of object retention.
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
    
    # Count lines efficiently
    logger.info(f"Counting objects in {csv_input}...")
    total_lines = count_file_lines(csv_input)
    
    if skip_header and total_lines > 0:
        total_lines -= 1
    
    # Skip further processing if there are no lines to process
    if total_lines <= 0:
        logger.warning("No objects found in the input file.")
        return
    
    # Start processing
    logger.info(f"Found {total_lines} objects to process")
    
    # Confirmation step
    if dry_run:
        logger.info(f"DRY RUN MODE: Would process {total_lines} objects")
    else:
        if not no_confirm:
            if not click.confirm(f"Are you sure you want to permanently delete objects from the S3 paths in {csv_input}?", 
                               default=True):
                logger.info("Deletion cancelled.")
                return
        logger.info(f"Starting deletion of objects from {csv_input} using {jobs} parallel jobs.")
    
    # Process CSV file in batches
    all_failed_objects = []
    bucket_object_counts = defaultdict(int)
    
    with open(csv_input, "r") as file:
        reader = csv.reader(file)
        
        # Skip header if requested
        if skip_header:
            next(reader, None)
        
        # First pass to count objects by bucket for better progress reporting
        if dry_run:
            # Sample the first few entries to show what would be deleted
            logger.info("Dry run - showing sample of objects that would be deleted:")
            sample_count = 0
            bucket_samples = defaultdict(int)
            
            for curr_bucket, batch in process_s3_deletion_batches(reader, batch_size):
                # Add to the sample count
                batch_size_actual = len(batch)
                bucket_samples[curr_bucket] += batch_size_actual
                sample_count += batch_size_actual
                
                # Show sample objects (up to 10 total)
                if sample_count <= 10:
                    for obj in batch:
                        logger.info(f"Would delete: s3://{curr_bucket}/{obj['Key']} (Version: {obj['VersionId']})")
                        
                # Stop once we've shown enough samples
                if sample_count >= 10:
                    break
                    
            # Show summary of what would be deleted
            logger.info("\nSummary of objects that would be deleted:")
            for b, count in bucket_samples.items():
                bucket_percent = (count / sample_count) * 100 if sample_count > 0 else 0
                logger.info(f"  Bucket {b}: {count} objects shown ({bucket_percent:.1f}% of sample)")
                
            logger.info("\nDry run complete. Use --no-dry-run to perform actual deletion.")
                
        else:
            # Rewind file to start for actual processing
            file.seek(0)
            if skip_header:
                next(reader, None)
                
            # Process in parallel using joblib
            logger.info("Processing objects for deletion...")
            
            # Use Parallel to process batches
            batch_results = Parallel(n_jobs=jobs, backend="threading")(
                delayed(process_batch)(curr_bucket, batch, throttle)
                for curr_bucket, batch in tqdm(
                    process_s3_deletion_batches(reader, batch_size),
                    desc="Processing S3 Batches",
                    mininterval=1
                )
            )
            
            # Process results
            for curr_bucket, success_count, failed_objects in batch_results:
                bucket_object_counts[curr_bucket] += success_count
                all_failed_objects.extend([(curr_bucket, obj) for obj in failed_objects])
            
            # Log results
            total_deleted = sum(bucket_object_counts.values())
            logger.info("\nDeletion Summary:")
            logger.info(f"Total processed: {total_lines}, Successfully deleted: {total_deleted}, Failed: {len(all_failed_objects)}")
            
            # Show per-bucket summary
            for b, count in bucket_object_counts.items():
                logger.info(f"  Bucket {b}: {count} objects deleted")
            
            # Save error log if requested
            if error_log and all_failed_objects:
                with open(error_log, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Bucket", "Key", "VersionId", "Error Code", "Error Message"])
                    for curr_bucket, obj in all_failed_objects:
                        writer.writerow([
                            curr_bucket,
                            obj.get("Key", ""),
                            obj.get("VersionId", ""),
                            obj.get("Code", ""),
                            obj.get("Message", "")
                        ])
                logger.info(f"Error log saved to {error_log}")
                
    logger.info("Operation complete.")


if __name__ == "__main__":
    cli()