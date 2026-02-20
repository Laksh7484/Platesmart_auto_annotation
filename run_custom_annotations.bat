@echo off
setlocal enabledelayedexpansion

echo ===================================
echo Custom Model Annotation Processor
echo ===================================
echo This will ONLY use your custom-trained
echo YOLO model for car and plate detection.
echo.

REM Check if Python environment is activated
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Please make sure you have set up the Python environment.
    pause
    exit /b 1
)

REM Check if custom model exists
if not exist "Models\custom_car_plate.pt" (
    echo Error: Models\custom_car_plate.pt not found!
    echo Please make sure your trained model is in the Models folder.
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
echo Starting annotation process using ONLY the custom model...
echo Project ID: !PROJECT_ID!
echo.

REM Run the Python script with the provided project ID
.venv\Scripts\python.exe Script\custom_annotate.py --project-id !PROJECT_ID!

REM Check if the script executed successfully
if errorlevel 1 (
    echo.
    echo There was an error running the script.
    pause
    exit /b 1
)

echo.
echo Process completed successfully!
echo.
echo Press any key to exit...
pause >nul
