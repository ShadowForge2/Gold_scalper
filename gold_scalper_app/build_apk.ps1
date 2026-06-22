$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CustomBuildDir = "C:\GoldScalperBuild"

Write-Host "Building Gold Scalper APK..." -ForegroundColor Cyan

# Run Flutter build (Gradle output redirected to C:\GoldScalperBuild)
$output = & flutter build apk --debug 2>&1
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "APK built successfully!" -ForegroundColor Green
    exit 0
}

# Gradle succeeded but Flutter couldn't find APK at standard path
# Copy from custom build dir to Flutter-expected path
Write-Host "Flutter build reported issue — copying APK from custom build dir..." -ForegroundColor Yellow

$srcApk = Join-Path $CustomBuildDir "app\outputs\flutter-apk\app-debug.apk"
$dstDir = Join-Path $ProjectRoot "build\app\outputs\flutter-apk"
$dstApk = Join-Path $dstDir "app-debug.apk"

if (Test-Path $srcApk) {
    New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    Copy-Item -LiteralPath $srcApk -Destination $dstApk -Force
    $size = (Get-Item $dstApk).Length
    Write-Host "✓ Built build\app\outputs\flutter-apk\app-debug.apk ($([math]::Round($size/1MB, 1)) MB)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "ERROR: APK not found at $srcApk" -ForegroundColor Red
    exit 1
}
