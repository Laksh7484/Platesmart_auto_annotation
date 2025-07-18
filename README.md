# Label Studio Auto-Annotation Tool

This tool automatically processes images in a Label Studio project, detecting cars and license plates using YOLOv8 models, and uploads the annotations back to Label Studio.

## Features

- Automatic detection of cars and license plates in images
- Support for images stored in Label Studio or S3
- Pagination support for large projects
- Skip already annotated tasks
- Environment variable support for secure credential management
- Debug mode for troubleshooting

## Prerequisites

- Python 3.8+
- Label Studio account
- YOLOv8 models for car and license plate detection

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your credentials:
   ```
   cp .env.example .env
   ```

## Usage

```
python process_annotations.py --project-id YOUR_PROJECT_ID [--debug] [--skip-s3]
```

### Command Line Arguments

- `--project-id`: (Required) Your Label Studio project ID
- `--debug`: (Optional) Enable debug logging
- `--skip-s3`: (Optional) Skip S3 initialization even if credentials are provided

## Environment Variables

You can set the following environment variables in your `.env` file:

- `LABEL_STUDIO_API_TOKEN`: Your Label Studio API token
- `LABEL_STUDIO_URL`: Your Label Studio URL (default: https://app.humansignal.com)
- `AWS_ACCESS_KEY`: Your AWS access key for S3
- `AWS_SECRET_KEY`: Your AWS secret key for S3
- `AWS_REGION`: Your AWS region (default: us-east-1)
- `S3_BUCKET`: Your S3 bucket name
- `DEBUG`: Set to "true" to enable debug logging

## How It Works

1. The script connects to Label Studio API and fetches tasks from your project
2. For each task without existing annotations:
   - It downloads the image (from Label Studio or S3)
   - Runs object detection using YOLOv8 models
   - Creates annotations in Label Studio format
   - Uploads the annotations back to Label Studio

## Using resolve_uri=true

The script now uses the `resolve_uri=true` parameter when fetching tasks from Label Studio. This makes Label Studio return direct URLs to images instead of S3 URIs, allowing the script to access images without needing S3 credentials.
