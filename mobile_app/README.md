# fundus_screener

A fully offline Android app for a fundus/retinal photo screening research
study. It captures fundus photos using a phone camera paired with a
DIYretCAM lens rig, runs on-device multi-disease detection (diabetic
retinopathy, glaucoma, cataract, AMD, hypertensive retinopathy) against a
TFLite backbone plus a Dart-native classification head, and shows
Grad-CAM heatmaps so a reviewer can see which regions of the image drove
each prediction. No network access is required or used at any point in
the capture/inference/save flow.

## Running tests on macOS

`flutter test` (e.g. `test/inference/backbone_engine_test.dart`) uses the
`tflite_flutter` plugin, which on macOS fails with an error like this on a
fresh machine/clone:

```
Invalid argument(s): Failed to load dynamic library
'/<flutter-sdk>/bin/cache/artifacts/engine/resources/libtensorflowlite_c-mac.dylib':
dlopen(...) tried: '.../resources/libtensorflowlite_c-mac.dylib' (no such file), ...
  dart:ffi                                                 new DynamicLibrary.open
  package:tflite_flutter/src/bindings/bindings.dart 32:27  _dylib.<fn>
  ...
  package:tflite_flutter/src/interpreter.dart 126:24       Interpreter.fromAsset
```

**Why this happens:** `tflite_flutter`'s macOS native binding
(`lib/src/bindings/bindings.dart`) loads the TFLite C library relative to
the running executable, assuming it's inside a built `.app` bundle
(`Contents/MacOS/<exe>` next to `Contents/Resources/<dylib>`). But
`flutter test` on desktop runs the `flutter_tester` binary directly out of
the Flutter SDK's engine cache
(`<flutter-sdk>/bin/cache/artifacts/engine/darwin-x64/flutter_tester`),
which has no such `resources/` sibling directory — so the dylib the plugin
ships in its own package is never found. This only affects `flutter test`
on macOS; it does **not** affect the real Android app, which loads the
Android `.so` normally via `Interpreter.fromAsset` and the Android asset
bundle.

**One-time fix per machine:** copy the plugin's bundled dylib to the exact
path the binding expects, inside your Flutter SDK's cache (adjust
`tflite_flutter-<version>` and `$FLUTTER_HOME` as needed — check
`pubspec.lock` for the resolved `tflite_flutter` version):

```bash
FLUTTER_HOME="$(dirname "$(dirname "$(which flutter)")")"
TFLITE_VERSION="0.12.1"  # match the tflite_flutter entry in pubspec.lock

mkdir -p "$FLUTTER_HOME/bin/cache/artifacts/engine/resources"
cp "$HOME/.pub-cache/hosted/pub.dev/tflite_flutter-$TFLITE_VERSION/macos/libtensorflowlite_c-mac.dylib" \
   "$FLUTTER_HOME/bin/cache/artifacts/engine/resources/libtensorflowlite_c-mac.dylib"
```

After this, `flutter test` will find the dylib and load the TFLite
interpreter correctly. This step is not automated by `flutter pub get` and
must be repeated whenever the Flutter SDK cache is reset (e.g. reinstalling
Flutter, or a fresh CI runner image).
