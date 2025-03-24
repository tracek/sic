#!/usr/bin/env python3

import csv
import click
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List, Tuple
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
def cli():
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
    type=int,
    help="Maximum directory depth to process. Use -1 for no limit (default)",
)
@click.option(
    "--n-jobs",
    default=-1,
    type=int,
    help="Number of parallel jobs to run. Use -1 for all available cores (default)",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level (default: INFO)",
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
    type=int,
    help="Maximum number of rows per sheet (default: 1,000,000). Sheets with more rows will be split.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level (default: INFO)",
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



if __name__ == "__main__":
    cli()
