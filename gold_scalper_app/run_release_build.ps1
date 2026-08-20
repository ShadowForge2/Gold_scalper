$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
$env:PATH = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot\bin;C:\flutter\bin;" + $env:PATH
$env:GRADLE_OPTS = "-Xmx1536m -XX:MaxMetaspaceSize=512m"
$env:ANDROID_HOME = "C:\Users\Sir AK HEE\AppData\Local\Android\Sdk"

Set-Location "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app"

$logFile = "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app\build_release_log.txt"

"Release build started at $(Get-Date)" | Out-File $logFile -Encoding UTF8

flutter build apk --release --no-pub --android-skip-build-dependency-validation 2>&1 | Out-File $logFile -Append

"Release build finished at $(Get-Date) with exit code $LASTEXITCODE" | Out-File $logFile -Append
