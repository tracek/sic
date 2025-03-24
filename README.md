# S3 inventory calculator

A Python CLI tool for analyzing AWS S3 inventory reports and generating size reports by directory level.

## Features

- Processes AWS S3 inventory CSV files to aggregate storage usage by directory levels
- Supports parallel processing for faster analysis of large inventory files
- Generates CSV reports for each directory level showing sizes and file counts
- Exports analysis results to Excel with auto-formatted sheets
- Handles large datasets by automatically splitting sheets that exceed Excel's row limits

## Usage

### Analyzing S3 Inventory Files

```bash
python s3_storage_analysis.py analyze \
  --input-dir /path/to/inventory/csv/files \
  --output-dir /path/to/output/directory \
  --max-depth 3
```

This will:
1. Process all CSV files in the input directory
2. Generate level-based reports in the output directory
3. Create files named `level_1_sizes.csv`, `level_2_sizes.csv`, etc.

### Excel Export

After generating CSV reports, you can export them to a single Excel file:

```bash
python s3_storage_analysis.py excel-export \
  --input-dir /path/to/csv/reports \
  --output-file storage_report.xlsx
```

For large datasets that might exceed Excel's row limits:

```bash
python s3_storage_analysis.py excel-export \
  --input-dir /path/to/csv/reports \
  --output-file storage_report.xlsx \
  --max-rows 500000
```

## Command Options

### Analyze Command

```
--input-dir      Directory containing S3 inventory CSV files (required)
--output-dir     Directory to save the CSV reports (required)
--max-depth      Maximum directory depth to analyze (-1 for unlimited, default: -1)
--n-jobs         Number of parallel jobs (-1 for all cores, default: -1)
--log-level      Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, default: INFO)
```

### Excel Export Command

```
--input-dir      Directory containing generated CSV reports (required)
--output-file    Path to save the Excel file (required)
--max-rows       Maximum rows per sheet (default: 1,000,000)
--log-level      Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, default: INFO)
```

## Output Format

### CSV Reports

Each level report contains:
- Directory path
- Size in GB
- File count

Example `level_1_sizes.csv`:
```
Directory,Size (GB),File Count
data/,1250.45,12500
logs/,850.22,9800
backups/,650.75,4200
```

### Excel Export

The Excel file contains:
- One sheet per CSV report (named "Level 1", "Level 2", etc.)
- Auto-adjusted column widths
- If a sheet exceeds the row limit, it's split into multiple sheets (e.g., "Level 1 (1-1000000)", "Level 1 (1000001-1500000)")

## Performance Tips

- Processing large inventory files can be memory-intensive. Adjust `--n-jobs` if memory is limited.
- For analyzing very deep directory structures, use `--max-depth` to limit analysis.
- The Excel export feature works best for reports under a few million rows total.

## License

[MIT License](LICENSE)
