# S3 Inventory Command-line (SIC)

A command-line tool for managing and analyzing AWS S3 storage at scale.

## Overview

SIC is a Python command-line utility for working with AWS S3 storage, designed to handle large-scale operations efficiently. It helps with storage analysis, object management, and migrating between storage classes.

## Features

- **Storage Analysis**: Process S3 inventory files to aggregate and visualize storage usage by directory levels
- **Delete Marker Management**: Find and manage delete markers in versioned buckets
- **Version Management**: List and manipulate object versions
- **Storage Class Migration**: Move objects between storage classes
- **Prefix Operations**: List and delete objects under specific prefixes
- **Optimized for Scale**: Uses parallel processing, throttling, and batching to handle millions of objects

## Command Reference

```
Usage: sic.py [OPTIONS] COMMAND [ARGS]...

  S3 Storage Analysis Tool - Process and visualize S3 inventory data

Options:
  --help  Show this message and exit.

Commands:
  analyze                 Process AWS S3 Inventory CSV or Parquet files...
  delete-prefix           Delete objects listed in a CSV file (generated...
  download-from-manifest  Download files from AWS S3 using an inventory...
  excel-export            Export CSV reports to a single Excel file with...
  find-deleted            Find all objects with delete markers in an S3...
  find-versions           Find all object versions and/or delete markers...
  list-prefix             List all objects under a specified prefix in an...
  move-storage-class      Move objects to a different storage class while...
```

### Analyze S3 Storage

```bash
python sic.py analyze --input-dir /path/to/inventory --output-dir /path/to/results --max-depth 3
```

Processes S3 inventory files to aggregate storage by directory level, creating size reports for each level.

### Export to Excel

```bash
python sic.py excel-export --input-dir /path/to/reports --output-file summary.xlsx
```

Creates a formatted Excel file from CSV reports, with filtering for size and object count.

### Find Delete Markers

```bash
python sic.py find-deleted --bucket mybucket --prefix data/ --output markers.csv
```

Finds all objects with delete markers in a bucket, useful for restoring accidentally deleted objects.

### Find Object Versions

```bash
python sic.py find-versions --bucket mybucket --prefix logs/ --output versions.csv
```

Lists all versions of objects in a versioned bucket, with options to filter by version type.

### List Objects in a Prefix

```bash
python sic.py list-prefix --bucket mybucket --prefix data/ --output objects.csv --include-size
```

Lists all objects under a prefix, with options to include size information and filter by storage class.

### Delete Objects by Prefix

```bash
python sic.py delete-prefix --input-file objects.csv --dry-run
```

Deletes objects listed in a CSV file, with batch processing and error logging.

### Move Objects to Different Storage Class

```bash
python sic.py move-storage-class --bucket mybucket --prefix logs/2023/ --storage-class GLACIER
```

Changes the storage class of objects while maintaining their original prefixes.

### Download from S3 Inventory Manifest

```bash
python sic.py download-from-manifest --json inventory-manifest.json --dir /download/path
```

Downloads files listed in an S3 inventory manifest JSON file.

## Examples

### Analyze S3 Storage and Export to Excel

```bash
# Process inventory files and create reports
python sic.py analyze --input-dir /inventory --output-dir /reports --max-depth 3

# Export reports to Excel for sharing
python sic.py excel-export --input-dir /reports --output-file storage_report.xlsx --min-size-gb 1.0
```

### Find and Remove Delete Markers

```bash
# Find delete markers
python sic.py find-deleted --bucket mybucket --output delete_markers.csv

# Delete the markers (after review)
python sic.py delete-prefix --input-file delete_markers.csv
```

### Move Old Data to Glacier Storage

```bash
# List objects
python sic.py list-prefix --bucket mybucket --prefix logs/2022/ --output old_logs.csv --include-size

# Move to Glacier
python sic.py move-storage-class --bucket mybucket --input-file old_logs.csv --storage-class GLACIER
```

## Performance Considerations

- For large buckets (100M+ objects), use the `--batch-size` and `--throttle` options to avoid API rate limits
- Use `--jobs` parameter to control parallelism based on your system's capabilities
- For inventory analysis, processing Parquet files is faster than CSV

## Safety Features

- `--dry-run` option for simulating operations without making changes
- Confirmation prompts before destructive operations
- Error logging for failed operations
- Exponential backoff for API rate limits
