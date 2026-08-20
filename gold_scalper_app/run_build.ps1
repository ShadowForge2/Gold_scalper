$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
$env:PATH = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot\bin;C:\flutter\bin;" + $env:PATH
$env:GRADLE_OPTS = "-Xmx1536m -XX:MaxMetaspaceSize=512m"
$env:ANDROID_HOME = "C:\Users\Sir AK HEE\AppData\Local\Android\Sdk"

Set-Location "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app"

"Build started at $(Get-Date)" | Out-File "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app\build_log_new.txt"

flutter build apk --debug --no-pub --android-skip-build-dependency-validation 2>&1 | Out-File "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app\build_log_new.txt" -Append

"Build finished at $(Get-Date) with exit code $LASTEXITCODE" | Out-File "C:\Users\Sir AK HEE\Desktop\MY PROJECTS\QuantoraFX (scalping machine)\gold_scalper_app\build_log_new.txt" -Append
