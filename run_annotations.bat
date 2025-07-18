@echo off
setlocal enabledelayedexpansion

echo ===================================
echo License Plate Annotation Processor
echo ===================================
echo.

REM Check if Python environment is activated
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Please make sure you have set up the Python environment.
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Prompt for project ID
set /p PROJECT_ID=Enter Label Studio Project ID: 

REM Validate input is a number
echo !PROJECT_ID!| findstr /r "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo Error: Project ID must be a positive number.
    pause
    exit /b 1
)

echo.
echo Starting annotation process for project ID: !PROJECT_ID!
echo.

REM Run the Python script with the provided project ID
.venv\Scripts\python.exe process_annotations.py --project-id !PROJECT_ID!

REM Check if the script executed successfully
if errorlevel 1 (
    echo.
    echo There was an error running the script.
    echo.
    echo Note: If you see a 404 error for a page during task fetching, this is normal
    echo when reaching the end of available tasks and doesn't indicate a critical failure.
    pause
    exit /b 1
)

echo.
echo Process completed successfully!
echo.
echo Press any key to exit...
pause >nul