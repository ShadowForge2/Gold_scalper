@echo off
cd /d "%~dp0android"
call gradlew.bat assembleDebug %*
if %errorlevel% neq 0 exit /b %errorlevel%
set "DSTDIR=%~dp0build\app\outputs\flutter-apk"
if not exist "%DSTDIR%" mkdir "%DSTDIR%"
copy /y "C:\GoldScalperBuild\app\outputs\flutter-apk\app-debug.apk" "%DSTDIR%\app-debug.apk" >nul
echo APK ready: build\app\outputs\flutter-apk\app-debug.apk
