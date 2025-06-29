# PowerShell script to run the Automatic Dataset Downloader Tool
# Run this script with: powershell -ExecutionPolicy Bypass -File run_downloader.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Automatic Dataset Downloader Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if required packages are installed
Write-Host "Checking required packages..." -ForegroundColor Yellow
try {
    python -c "import requests, tqdm" 2>$null
    Write-Host "✅ Required packages are installed" -ForegroundColor Green
} catch {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    try {
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install packages"
        }
        Write-Host "✅ Packages installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ ERROR: Failed to install required packages" -ForegroundColor Red
        Write-Host "Press any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
}

Write-Host ""
Write-Host "Starting dataset downloader..." -ForegroundColor Green
Write-Host ""

# Run the enhanced downloader
try {
    python enhanced_downloader.py
} catch {
    Write-Host "❌ Error running the downloader: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 