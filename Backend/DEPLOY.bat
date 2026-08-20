@echo off
setlocal
title QuantoraFX Deploy
color 0A

echo ============================================
echo   QuantoraFX — Safe Deploy
echo ============================================
echo.

:: Navigate to project root (one level up from Backend/)
cd /d "%~dp0.."

set PRIMARY=https://gold-scalper-qyhg.onrender.com
set BACKUP=https://gold-scalper.onrender.com
set FORCE=0

if "%1"=="--force" set FORCE=1
if "%1"=="-f" set FORCE=1

echo [1/4] Checking deploy safety...
echo.

:: Try primary first, fallback to backup
set URL=%PRIMARY%/api/deploy/check
echo   Trying %URL% ...
curl -s -m 15 "%URL%" > "%TEMP%\deploy_check.json" 2>nul

:: Check if we got a valid response
findstr /C:"safe" "%TEMP%\deploy_check.json" >nul 2>&1
if errorlevel 1 (
    echo   Primary unreachable, trying backup...
    set URL=%BACKUP%/api/deploy/check
    curl -s -m 15 "%URL%" > "%TEMP%\deploy_check.json" 2>nul
)

:: Parse the response
type "%TEMP%\deploy_check.json"
echo.

:: Check if safe=true
findstr /C:"\"safe\": true" "%TEMP%\deploy_check.json" >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo   [OK] No open trades — safe to deploy!
    echo.
    goto :PUSH
)

:: Check if safe=false
findstr /C:"\"safe\": false" "%TEMP%\deploy_check.json" >nul 2>&1
if %errorlevel%==0 (
    if "%FORCE%"=="1" (
        echo.
        echo   [FORCE] Open trades detected, but --force flag set. Deploying anyway.
        echo.
        goto :PUSH
    )
    echo.
    echo   [BLOCKED] Open trades detected! Deploy cancelled.
    echo   Use --force to override (risky — positions may orphan).
    echo.
    echo   Options:
    echo     1. Wait for trades to close, then retry
    echo     2. Run: DEPLOY.bat --force
    echo.
    goto :END
)

:: Unparseable response — server might be down, safe to deploy
echo.
echo   [WARN] Could not parse response. Server may be down — proceeding.
echo.
goto :PUSH

:PUSH
echo [2/4] Staging changes...
git add -A

echo [3/4] Committing...
git commit -m "deploy: auto-push %date% %time%"

echo [4/4] Pushing to origin...
git push origin main

echo.
echo ============================================
echo   Deploy pushed successfully!
echo   Render will auto-deploy in ~2-3 minutes.
echo ============================================
goto :END

:END
del "%TEMP%\deploy_check.json" 2>nul
endlocal
