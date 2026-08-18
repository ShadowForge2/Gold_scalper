@echo off
cd /d "%~dp0android"
call gradlew.bat assembleRelease %*
if %errorlevel% neq 0 exit /b %errorlevel%
set "DSTDIR=%~dp0build\app\outputs\flutter-apk"
if not exist "%DSTDIR%" mkdir "%DSTDIR%"
copy /y "C:\GoldScalperBuild\app\outputs\flutter-apk\app-release.apk" "%DSTDIR%\app-release.apk" >nul
echo APK ready: build\app\outputs\flutter-apk\app-release.apk
