#!/usr/bin/env bash
# Phase 3 preflight: install the missing mobile toolchain WITHOUT root.
# Everything lands in ~/toolchains (disk, never /tmp tmpfs). Restart-proof: each
# step skips if its final artifact exists. Logs progress to stdout (captured by
# build/toolchain_setup.log). Versions are pinned/reported at the end.
set -uo pipefail
TOOLS="$HOME/toolchains"
mkdir -p "$TOOLS"

echo "=== [1/4] JDK 17 (Temurin) ==="
if [ -x "$TOOLS/jdk-17/bin/java" ]; then
  echo "[skip] jdk-17 present"
else
  curl -fL --retry 3 -o "$TOOLS/jdk17.tar.gz" \
    "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse" \
    || { echo "FATAL: JDK download failed"; exit 11; }
  mkdir -p "$TOOLS/jdk-17"
  tar -xzf "$TOOLS/jdk17.tar.gz" -C "$TOOLS/jdk-17" --strip-components=1 || { echo "FATAL: JDK extract failed"; exit 12; }
  rm -f "$TOOLS/jdk17.tar.gz"
fi
export JAVA_HOME="$TOOLS/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
java -version 2>&1 | head -1

echo "=== [2/4] Android command-line tools ==="
if [ -x "$TOOLS/android-sdk/cmdline-tools/latest/bin/sdkmanager" ]; then
  echo "[skip] cmdline-tools present"
else
  curl -fL --retry 3 -o "$TOOLS/cmdline-tools.zip" \
    "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip" \
    || { echo "FATAL: cmdline-tools download failed"; exit 21; }
  mkdir -p "$TOOLS/android-sdk/cmdline-tools"
  unzip -q "$TOOLS/cmdline-tools.zip" -d "$TOOLS/android-sdk/cmdline-tools" || { echo "FATAL: unzip failed (is unzip installed?)"; exit 22; }
  mv "$TOOLS/android-sdk/cmdline-tools/cmdline-tools" "$TOOLS/android-sdk/cmdline-tools/latest"
  rm -f "$TOOLS/cmdline-tools.zip"
fi
export ANDROID_HOME="$TOOLS/android-sdk"
SDKMANAGER="$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager"

echo "=== [3/4] Android SDK packages ==="
if [ -d "$ANDROID_HOME/platforms/android-35" ] && [ -d "$ANDROID_HOME/build-tools/35.0.0" ] && [ -x "$ANDROID_HOME/platform-tools/adb" ]; then
  echo "[skip] platform-35 + build-tools-35.0.0 + adb present"
else
  yes | "$SDKMANAGER" --licenses >/dev/null 2>&1 || true
  "$SDKMANAGER" "platform-tools" "platforms;android-35" "build-tools;35.0.0" || { echo "FATAL: sdkmanager install failed"; exit 23; }
fi

echo "=== [4/4] Flutter SDK (stable) ==="
if [ -x "$TOOLS/flutter/bin/flutter" ]; then
  echo "[skip] flutter present"
else
  # pin: resolve the current stable release URL from Google's releases index
  FLUTTER_URL=$("$TOOLS/../.venv/bin/python" 2>/dev/null -c "" || true)
  URL=$(curl -fsSL "https://storage.googleapis.com/flutter_infra_release/releases/releases_linux.json" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); h=d['current_release']['stable']; r=[x for x in d['releases'] if x['hash']==h][0]; print('https://storage.googleapis.com/flutter_infra_release/releases/' + r['archive'])") \
    || { echo "FATAL: cannot resolve flutter stable URL"; exit 31; }
  echo "flutter archive: $URL"
  curl -fL --retry 3 -o "$TOOLS/flutter.tar.xz" "$URL" || { echo "FATAL: flutter download failed"; exit 32; }
  tar -xJf "$TOOLS/flutter.tar.xz" -C "$TOOLS" || { echo "FATAL: flutter extract failed"; exit 33; }
  rm -f "$TOOLS/flutter.tar.xz"
fi

echo "=== versions ==="
"$TOOLS/jdk-17/bin/java" -version 2>&1 | head -1
"$TOOLS/flutter/bin/flutter" --version 2>&1 | head -3
"$SDKMANAGER" --version 2>&1 | tail -1
echo "ANDROID_HOME=$ANDROID_HOME"
echo "TOOLCHAIN_SETUP_DONE"
