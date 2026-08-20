@echo off
setlocal

REM === JDK 17 ===
set "JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
set "PATH=%JAVA_HOME%\bin;%PATH%"

REM === Flutter SDK ===
set "FLUTTER_ROOT=C:\flutter"
set "PATH=%FLUTTER_ROOT%\bin;%PATH%"

REM === Android SDK ===
set "ANDROID_HOME=C:\Users\Sir AK HEE\AppData\Local\Android\Sdk"
set "PATH=%ANDROID_HOME%\platform-tools;%PATH%"

REM === Gradle memory: 1GB max (safe for 3GB laptop) ===
set "GRADLE_OPTS=-Xmx1024m -XX:MaxMetaspaceSize=512m"

echo ============================================
echo   QuantoraFX APK Builder (3GB RAM Safe)
echo ============================================
echo JAVA_HOME=%JAVA_HOME%
echo FLUTTER_ROOT=%FLUTTER_ROOT%
echo ANDROID_HOME=%ANDROID_HOME%
echo.

REM === Verify tools ===
echo [1/5] Checking Java...
java -version 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Java not found at %JAVA_HOME%
    pause
    exit /b 1
)

echo.
echo [2/5] Checking Flutter...
flutter --version 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Flutter not found at %FLUTTER_ROOT%
    pause
    exit /b 1
)

echo.
echo [3/5] Flutter doctor...
flutter doctor --android-licenses 2>nul
flutter doctor -v 2>&1
echo.

echo [4/5] Getting dependencies...
cd /d "%~dp0"
flutter pub get
if %errorlevel% neq 0 (
    echo ERROR: flutter pub get failed
    pause
    exit /b 1
)

echo.
echo [5/5] Building release APK (no-daemon, single-thread)...
cd /d "%~dp0android"
call gradlew.bat assembleRelease --no-daemon --no-parallel -Dorg.gradle.jvmargs="-Xmx1024m -XX:MaxMetaspaceSize=512m"
if %errorlevel% neq 0 (
    echo.
    echo Build FAILED. Common fixes for low RAM:
    echo   1. Close all other programs to free memory
    echo   2. Increase pagefile/swap size in Windows settings
    echo   3. Try debug build instead: gradlew.bat assembleDebug --no-daemon
    pause
    exit /b %errorlevel%
)

set "DSTDIR=%~dp0build\app\outputs\flutter-apk"
if not exist "%DSTDIR%" mkdir "%DSTDIR%"
copy /y "C:\GoldScalperBuild\app\outputs\flutter-apk\app-release.apk" "%DSTDIR%\app-release.apk" >nul 2>&1

echo.
echo ============================================
echo   BUILD SUCCESS!
echo   APK: %DSTDIR%\app-release.apk
echo ============================================
pause
