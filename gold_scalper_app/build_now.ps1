$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;C:\flutter\bin;$env:PATH"
$env:GRADLE_OPTS = "-Xmx1536m -XX:MaxMetaspaceSize=512m"
$env:ANDROID_HOME = "C:\Users\Sir AK HEE\AppData\Local\Android\Sdk"

Set-Location "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app"

Write-Host "Starting build at $(Get-Date)..."
flutter build apk --debug --no-pub --android-skip-build-dependency-validation 2>&1 | Tee-Object -FilePath "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app\build_output.txt"

Write-Host "Build finished at $(Get-Date)"
Write-Host "Exit code: $LASTEXITCODE"

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS! APK built."
} else {
    Write-Host "BUILD FAILED. Check build_output.txt"
}
