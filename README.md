# Label Studio Auto-Annotation Tool

This tool automatically processes images in a Label Studio project, detecting cars and license plates using Plate Recognizer Snapshot SDK, and uploads the annotations back to Label Studio with **concurrent processing** for improved performance.

## Features

- **🚀 Concurrent Processing**: Uses ThreadPoolExecutor to process multiple images simultaneously
- Automatic detection of cars and license plates in images using Plate Recognizer
- Support for images stored in Label Studio or S3
- Pagination support for large projects
- Skip already annotated tasks
- Configurable number of concurrent workers
- Real-time progress tracking and statistics
- Environment variable support for secure credential management

## Prerequisites

- Python 3.8+
- Label Studio account
- Plate Recognizer Snapshot SDK (Docker) running locally
- API tokens for both Label Studio and Plate Recognizer

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your API tokens:
   - Create `token.txt` with your Label Studio API token
   - Create `plate_recognizer_token.txt` with your Plate Recognizer API token
4. Ensure Plate Recognizer Snapshot SDK is running on `http://localhost:8080`

## Usage

### Basic Usage

```
python Script/process_annotations.py --project-id YOUR_PROJECT_ID
```

### Advanced Usage with Custom Workers

```
python Script/process_annotations.py --project-id YOUR_PROJECT_ID --max-workers 15
```

### Command Line Arguments

- `--project-id`: (Required) Your Label Studio project ID
- `--max-workers`: (Optional) Maximum number of concurrent workers (default: 10)

## Performance Optimization

The script now uses **ThreadPoolExecutor** to process multiple images concurrently:

- **Sequential Processing**: Old version processed one image at a time
- **Concurrent Processing**: New version processes multiple images simultaneously
- **Speed Improvement**: Typically 3-10x faster depending on your system and API limits
- **Configurable Workers**: Adjust `--max-workers` based on your system capabilities

### Recommended Worker Counts

- **Conservative**: 5-8 workers (good for slower systems or API rate limits)
- **Balanced**: 10-15 workers (default, good for most systems)
- **Aggressive**: 15-20 workers (for fast systems with high API limits)

## How It Works

1. **Concurrent Fetching**: The script connects to Label Studio API and fetches all tasks
2. **Parallel Processing**: Multiple worker threads process images simultaneously:
   - Download images from Label Studio or S3
   - Send to Plate Recognizer Snapshot SDK for detection
   - Convert results to Label Studio format
   - Upload annotations back to Label Studio
3. **Progress Tracking**: Real-time progress updates with completion statistics
4. **Error Handling**: Robust error handling with detailed logging per worker

## Testing Concurrent Processing

You can test the performance improvement with the included test script:

```bash
python Script/test_concurrent.py
```

This will demonstrate the difference between sequential and concurrent processing.

## Environment Variables

You can set the following environment variables in your `.env` file:

- `LABEL_STUDIO_API_TOKEN`: Your Label Studio API token
- `LABEL_STUDIO_URL`: Your Label Studio URL (default: https://app.humansignal.com)
- `PLATE_RECOGNIZER_URL`: Your Plate Recognizer SDK URL (default: http://localhost:8080/v1/plate-reader/)

## Troubleshooting

### Rate Limiting

If you encounter API rate limits:

- Reduce the `--max-workers` value
- The script includes built-in delays to prevent overwhelming the API

### Memory Usage

For very large projects:

- Monitor memory usage during processing
- Consider reducing `--max-workers` if memory becomes an issue

### API Errors

- Check that Plate Recognizer Snapshot SDK is running
- Verify your API tokens are correct
- Ensure network connectivity to both services

## Using resolve_uri=true

The script uses the `resolve_uri=true` parameter when fetching tasks from Label Studio. This makes Label Studio return direct URLs to images instead of S3 URIs, allowing the script to access images without needing S3 credentials.
