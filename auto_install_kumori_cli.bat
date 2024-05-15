@echo off
setlocal enabledelayedexpansion

:: Get start timestamp
for /f "tokens=1-4 delims=:.," %%a in ('echo %time%') do (
    set start_h=%%a
    set start_m=%%b
    set start_s=%%c
    set start_ms=%%d
)

echo ======================================================
echo Setting up the Kumori CLI Engine environment...
echo ======================================================
echo.
echo Starting script @: %time% 
echo FYI: This script could take 5-10 mins to complete, based on your system specs, but worry not, should be verbose...
echo.

:: Get initial free space using PowerShell
echo ------------------------------------------------------
echo Initial Disk Space Available (in GB):
echo ------------------------------------------------------
powershell -command "Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -eq 'C' } | Select-Object @{Name='Used (GB)'; Expression={[math]::Round($_.Used/1GB, 2)}}, @{Name='Free (GB)'; Expression={[math]::Round($_.Free/1GB, 2)}}"
echo.

echo ------------------------------------------------------
echo Step 1: Checking for Python installation...
echo ------------------------------------------------------
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in the PATH. Please install Python 3.8 or newer.
    pause
    exit /b
)
echo.

echo ------------------------------------------------------
echo Step 2: Checking for Git installation...
echo ------------------------------------------------------
git --version
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in the PATH.
    echo Git is essential for cloning the project repository to your local machine.
    echo If you do not have Git installed, please follow these instructions to install it:
    echo.
    echo Installing Git:
    echo   1. Download the latest Git for Windows installer from Git's official website: https://git-scm.com/download/win
    echo   2. You can also download it directly from this link: https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-32-bit.exe
    echo   3. Run the installer and follow the installation prompts. Accepting the default options is recommended for most users.
    echo.
    echo After installing Git, please run this script again.
    pause
    exit /b
)
echo Git is installed.
echo.

echo ------------------------------------------------------
echo Step 3: Checking and cleaning up existing directories...
echo ------------------------------------------------------
if exist kumori_venv (
    echo Deleting existing kumori_venv directory...
    rmdir /S /Q kumori_venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to delete kumori_venv directory.
        pause
        exit /b
    )
)
if exist kumori_cli_engine (
    echo Deleting existing kumori_cli_engine directory...
    rmdir /S /Q kumori_cli_engine
    if %errorlevel% neq 0 (
        echo ERROR: Failed to delete kumori_cli_engine directory.
        pause
        exit /b
    )
)
echo Existing directories cleaned up.
echo.

echo ------------------------------------------------------
echo Step 4: Creating Python virtual environment...
echo ------------------------------------------------------
python -m venv kumori_venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create Python virtual environment.
    pause
    exit /b
)
echo.

echo ------------------------------------------------------
echo Step 5: Activating the virtual environment...
echo ------------------------------------------------------
call .\kumori_venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate the virtual environment.
    pause
    exit /b
)
echo Virtual environment 'kumori_venv' activated.
echo.

echo ------------------------------------------------------
echo Step 6: Installing gdown package and other dependencies...
echo ------------------------------------------------------
pip install gdown
if %errorlevel% neq 0 (
    echo ERROR: Failed to install gdown package.
    pause
    exit /b
)
echo gdown package installed successfully.
echo.

echo ------------------------------------------------------
echo Step 7: Cloning Kumori CLI Engine repository...
echo ------------------------------------------------------
git clone https://github.com/tillo13/kumori_cli_engine.git
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone the Kumori CLI Engine repository.
    pause
    exit /b
)
cd kumori_cli_engine
echo Repository cloned and navigated to 'kumori_cli_engine' directory.
echo.

echo ------------------------------------------------------
echo Step 8: Installing dependencies from requirements.txt... This may take a minute or two...
echo ------------------------------------------------------
pip install -r requirements.txt -v
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b
)
echo.

echo ------------------------------------------------------
echo Step 9: Uninstalling existing PyTorch packages...
echo ------------------------------------------------------
pip uninstall torch torchvision torchaudio -y
if %errorlevel% neq 0 (
    echo ERROR: Failed to uninstall existing PyTorch packages.
    pause
    exit /b
)
echo.

echo ------------------------------------------------------
echo Step 10: Installing PyTorch with CUDA support for RTX 3060... This may take a minute or two...
echo ------------------------------------------------------
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch with CUDA support.
    pause
    exit /b
)
echo.

echo ------------------------------------------------------
echo Step 11: Downloading and extracting models_and_facial_landmarks_model.zip...
echo ------------------------------------------------------

:: Create the temporary Python script to download and extract the ZIP file
if exist temp_download_and_extract.py del temp_download_and_extract.py

echo import gdown > temp_download_and_extract.py
echo import zipfile >> temp_download_and_extract.py
echo import os >> temp_download_and_extract.py
echo. >> temp_download_and_extract.py
echo url = 'https://drive.google.com/uc?id=1jdfyvxHMvAN7OJMW3zGn0IZZ107OxmqN' >> temp_download_and_extract.py
echo zip_filename = 'models_and_facial_landmarks_model.zip' >> temp_download_and_extract.py
echo gdown.download(url, zip_filename, quiet=False) >> temp_download_and_extract.py
echo unzip_dir = os.path.dirname(os.path.abspath(zip_filename)) >> temp_download_and_extract.py
echo with zipfile.ZipFile(zip_filename, 'r') as zip_ref: >> temp_download_and_extract.py
echo     zip_ref.extractall(unzip_dir) >> temp_download_and_extract.py
echo print(f"File downloaded and extracted to {unzip_dir}") >> temp_download_and_extract.py

:: Run the temporary Python script
python temp_download_and_extract.py

if %errorlevel% neq 0 (
    echo ERROR: Failed to download and extract the files.
    del temp_download_and_extract.py
    pause
    exit /b
)
echo The additional files have been downloaded and extracted successfully.
echo.

:: Delete the temporary Python script
del temp_download_and_extract.py

:: Get end timestamp
for /f "tokens=1-4 delims=:.," %%a in ('echo %time%') do (
    set end_h=%%a
    set end_m=%%b
    set end_s=%%c
    set end_ms=%%d
)

:: Calculate elapsed time
set /a start_total_ms=start_h*3600000 + start_m*60000 + start_s*1000 + start_ms
set /a end_total_ms=end_h*3600000 + end_m*60000 + end_s*1000 + end_ms
set /a elapsed_ms=end_total_ms - start_total_ms

:: Convert elapsed time to human-readable format
set /a elapsed_s=elapsed_ms / 1000
set /a elapsed_ms=elapsed_ms %% 1000
set /a elapsed_m=elapsed_s / 60
set /a elapsed_s=elapsed_s %% 60
set /a elapsed_h=elapsed_m / 60
set /a elapsed_m=elapsed_m %% 60

echo.
echo Time to install: %elapsed_h% hours, %elapsed_m% minutes, %elapsed_s% seconds, %elapsed_ms% milliseconds

echo.
echo ------------------------------------------------------
echo Final Disk Space Available (in GB):
echo ------------------------------------------------------
powershell -command "Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -eq 'C' } | Select-Object @{Name='Used (GB)'; Expression={[math]::Round($_.Used/1GB, 2)}}, @{Name='Free (GB)'; Expression={[math]::Round($_.Free/1GB, 2)}}"

echo ------------------------------------------------------
echo IMPORTANT: Custom PyTorch Installation for Different GPUs
echo ------------------------------------------------------
echo We configured PyTorch with CUDA 11.8, which supports a variety of GPUs including NVIDIA RTX 3060.
echo If you have a different GPU, please follow these steps:
echo 1. Visit https://pytorch.org to get the appropriate installation command for your GPU.
echo 2. Uninstall the current PyTorch packages using the following command:
echo    pip uninstall torch torchvision torchaudio -y
echo 3. Install the PyTorch version specified by the command from PyTorch's website.
echo.

echo ======================================================
echo Setup complete!
echo ======================================================
echo All necessary files have been downloaded and extracted.
echo The correct version of PyTorch must be ensured for your specific GPU.
echo.

echo Please follow these steps to start using Kumori CLI Engine:
echo 1. Activate the virtual environment with: .\kumori_venv\Scripts\activate
echo 2. Navigate to the kumori_cli_engine directory: cd kumori_cli_engine
echo 3. Run the Kumori CLI Engine: python .\kumori_cli.py
echo 4. See the images you've created in the generated_images folder!
echo.

echo NOTE: Now you're ready to roll!  Re-running this will remove/re-install from scratch if you have troubles.
echo Check out the configs.py file to make changes to your personal preferences!
echo Additionally, for more information, refer to the README.md file located here: https://github.com/tillo13/kumori_cli_engine/blob/main/README.md
echo.

pause
endlocal