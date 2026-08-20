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

REM === Gradle memory: 1GB max ===
set "GRADLE_OPTS=-Xmx1024m -XX:MaxMetaspaceSize=512m"

echo ============================================
echo   QuantoraFX DEBUG APK Builder (Light)
echo ============================================

cd /d "%~dp0"
flutter pub get

cd /d "%~dp0android"
call gradlew.bat assembleDebug --no-daemon --no-parallel -Dorg.gradle.jvmargs="-Xmx1024m -XX:MaxMetaspaceSize=512m"

if %errorlevel% neq 0 (
    echo Build FAILED
    pause
    exit /b %errorlevel%
)

echo.
echo DEBUG APK: build\app\outputs\flutter-apk\app-debug.apk
echo (Debug APK is larger but builds faster and uses less RAM)
pause
