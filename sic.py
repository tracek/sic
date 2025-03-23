#!/usr/bin/env python3

import csv
import click
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, DefaultDict, Optional, Union, Any, Iterator
from loguru import logger


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
    parts = [p for p in prefix.split('/') if p]
    
    # Check if the last part is a filename (contains a period) and remove it
    if parts and '.' in parts[-1]:
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
        dir_path = '/'.join(parts[:i]) + '/'
        levels.append(dir_path)
    
    return levels

def process_csv_file(csv_file: Path, max_depth: int) -> Dict[int, Dict[str, int]]:
    """
    Process a single CSV file and return the aggregated sizes.
    
    Args:
        csv_file: Path to the CSV file
        max_depth: Maximum directory depth to consider
        
    Returns:
        Dictionary of {level: {directory: size}}
    """
    # Dictionary to store aggregated sizes for each level
    level_sizes = defaultdict(lambda: defaultdict(int))
    
    try:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_ALL)
            
            for row in reader:
                try:
                    if len(row) >= 3:  # Ensure row has at least 3 columns
                        bucket = row[0]
                        prefix = row[1]
                        size_bytes = int(row[2])
                        
                        # Extract directory paths at all levels (top to bottom)
                        dir_levels = extract_all_directory_levels(prefix, max_depth)
                        
                        # Aggregate sizes for each level
                        for level, dir_path in enumerate(dir_levels, 1):
                            level_sizes[level][dir_path] += size_bytes
                except (IndexError, ValueError) as e:
                    # Silent error handling for individual rows
                    pass
    except Exception as e:
        logger.error(f"Error processing file {csv_file}: {e}")
    
    return dict(level_sizes)  # Convert defaultdict to regular dict for better serialization

def process_files_in_batches(
    csv_files: List[Path], 
    max_depth: int, 
    n_jobs: int, 
    batch_size: int = 10
) -> List[Dict[int, Dict[str, int]]]:
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
            batch_files = csv_files[i:i + batch_size]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_csv_file)(csv_file, max_depth) for csv_file in batch_files
            )
            results.extend(batch_results)
            pbar.update(len(batch_files))
    
    return results

def combine_results(
    results_list: List[Dict[int, Dict[str, int]]]
) -> Dict[int, Dict[str, int]]:
    """
    Combine results from multiple processes.
    
    Args:
        results_list: List of dictionaries with level sizes from each process
        
    Returns:
        Combined dictionary of {level: {directory: size}}
    """
    combined = defaultdict(lambda: defaultdict(int))
    
    for result in results_list:
        for level, dirs in result.items():
            for dir_path, size in dirs.items():
                combined[level][dir_path] += size
    
    return dict(combined)

def process_csv_files(
    input_dir: Path, 
    output_dir: Path, 
    max_depth: int, 
    n_jobs: int
) -> None:
    """
    Process all CSV files in the input directory and generate aggregated reports.
    Uses parallel processing to speed up file processing.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save output files
        max_depth: Maximum directory depth to consider (-1 for no limit)
        n_jobs: Number of parallel jobs to run (-1 for all available cores)
    """
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process files in parallel with tqdm progress bar
    results = process_files_in_batches(csv_files, max_depth, n_jobs)
    
    # Combine results from all processes
    logger.info("Combining results from all files...")
    level_sizes = combine_results(results)
    
    # Determine the maximum level
    max_level = max(level_sizes.keys()) if level_sizes else 0
    
    # Convert sizes to GB and write to output files
    logger.info("Writing output files...")
    for level in tqdm(range(1, max_level + 1), desc="Writing level reports"):
        if level not in level_sizes:
            continue
            
        output_file = output_dir / f"level_{level}_sizes.csv"
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Directory', 'Size (GB)'])
            
            # Sort by size (descending)
            sorted_dirs = sorted(
                level_sizes[level].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for dir_path, size_bytes in sorted_dirs:
                size_gb = size_bytes / (1024 ** 3)  # Convert bytes to GB
                writer.writerow([dir_path, f"{size_gb:.2f}"])
    
    logger.success(f"Reports have been written to {output_dir}")
    
    if max_depth == -1:
        logger.info(f"Generated {max_level} level reports, from top level directories (level 1) to deeper structures")
    else:
        logger.info(f"Generated {max_level} level reports (max depth: {max_depth}), from top level directories (level 1) to deeper structures")

@click.command()
@click.option('--input-dir', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), 
              help='Directory containing CSV files')
@click.option('--output-dir', required=True, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), 
              help='Directory to save output files')
@click.option('--max-depth', default=-1, type=int, 
              help='Maximum directory depth to process. Use -1 for no limit (default)')
@click.option('--n-jobs', default=-1, type=int,
              help='Number of parallel jobs to run. Use -1 for all available cores (default)')
@click.option('--log-level', default="INFO", type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
              help='Set the logging level (default: INFO)')
def main(
    input_dir: Path, 
    output_dir: Path, 
    max_depth: int, 
    n_jobs: int, 
    log_level: str
) -> None:
    """
    Process AWS S3 Inventory CSV files and aggregate storage by directory levels.
    
    This script reads AWS S3 Inventory CSV files and aggregates storage sizes at
    directory levels, where level 1 is the top-most directory
    and each subsequent level goes one directory deeper, outputting results in GB.
    
    Parallel processing is used to speed up file analysis.
    """
    # Configure logger level
    logger.remove()
    logger.add(
        sink=lambda msg: tqdm.write(msg, end=""),  # Use tqdm.write to avoid breaking progress bars
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level
    )
    
    logger.info(f"Starting S3 Storage Analysis with {n_jobs} workers")
    
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Ensured output directory exists: {output_dir}")
    
    # Process CSV files
    process_csv_files(input_dir, output_dir, max_depth, n_jobs)
    
    logger.info("Analysis complete!")

if __name__ == '__main__':
    main()