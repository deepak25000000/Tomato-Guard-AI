# TomatoGuard app (Flutter, Android)

Research prototype. Offline on-device TFLite inference. No INTERNET permission in release manifest.

## Build
```bash
export JAVA_HOME="$HOME/toolchains/jdk-17"
export ANDROID_HOME="$HOME/toolchains/android-sdk"
export PATH="$JAVA_HOME/bin:$HOME/toolchains/flutter/bin:$PATH"
cd tomatoguard/app
flutter pub get
flutter analyze
flutter test
flutter build apk --release
```

## Assets
Generated from build artifacts (no hand-typed numbers):
```bash
.venv/bin/python build/scripts/gen_app_assets.py
```
Copies model, ui_strings, model_card into `tomatoguard/app/assets/`.

## Benchmark
In-app: About -> Run benchmark (20 warm-up + 100 timed). Record real-phone results in `benchmarks/device_results_template.csv` (emulator = "emulator", never used as phone result).
