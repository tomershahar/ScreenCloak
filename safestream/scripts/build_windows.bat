@echo off
setlocal enabledelayedexpansion

set VERSION=1.0.0
set APP_NAME=ScreenCloak
set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"

echo === ScreenCloak Windows Build ===
echo Version: %VERSION%
echo.

echo ^> Cleaning previous build...
if exist build rmdir /s /q build
if exist "dist\%APP_NAME%" rmdir /s /q "dist\%APP_NAME%"

echo ^> Running PyInstaller...
python -m PyInstaller ScreenCloak-Windows.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller failed
    exit /b 1
)
echo    dist\%APP_NAME%\ created

echo ^> Building installer with Inno Setup...
if not exist %ISCC% (
    echo ERROR: Inno Setup 6 not found at %ISCC%
    echo Install from: https://jrsoftware.org/isdl.php
    exit /b 1
)
%ISCC% scripts\screencloak.iss
if errorlevel 1 (
    echo ERROR: Inno Setup failed
    exit /b 1
)

echo.
echo === Build complete ===
echo   Installer: dist\%APP_NAME%-%VERSION%-Setup.exe
