import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'inference/backbone_engine.dart';
import 'inference/feature_map.dart';
import 'inference/head_weights.dart';
import 'inference/head_math.dart';
import 'inference/preprocess_config.dart';
import 'inference/preprocessing.dart';
import 'models/session_record.dart';
import 'screens/capture_screen.dart';
import 'screens/model_error_screen.dart';
import 'screens/plugin_camera_source.dart';
import 'screens/result_screen.dart';
import 'screens/session_id_screen.dart';
import 'storage/session_log.dart';

class ModelBundle {
  final BackboneEngine engine;
  final HeadWeights weights;
  final PreprocessConfig preprocessConfig;

  ModelBundle(this.engine, this.weights, this.preprocessConfig);
}

/// Loads the on-device model bundle (TFLite backbone + head weights +
/// preprocessing config) from the app's asset bundle.
///
/// All three assets are loaded through Flutter's real asset-bundle
/// machinery (`rootBundle`, which `BackboneEngine.load` also uses under the
/// hood via `Interpreter.fromAsset`), not raw filesystem reads — assets
/// bundled into an Android APK are not present at these paths on the
/// device's filesystem, only through the AssetBundle/platform-asset-channel
/// API. `flutter test` serves the same declared-in-pubspec asset paths
/// through that same API, so this behaves identically in tests and on a
/// real device.
///
/// Throws a descriptive [Exception] (always mentioning "model") if any
/// asset is missing or unparseable, so callers can distinguish a startup
/// model-load failure from any other error.
Future<ModelBundle> loadModelBundle(
    {String modelDirOverride = 'assets/model'}) async {
  try {
    final engine = BackboneEngine();
    await engine.load('$modelDirOverride/backbone.tflite');

    final headJsonString =
        await rootBundle.loadString('$modelDirOverride/head_weights.json');
    final weights =
        HeadWeights.fromJson(jsonDecode(headJsonString) as Map<String, dynamic>);

    final preprocessJsonString = await rootBundle
        .loadString('$modelDirOverride/preprocess_config.json');
    final preprocessConfig = PreprocessConfig.fromJson(
        jsonDecode(preprocessJsonString) as Map<String, dynamic>);

    return ModelBundle(engine, weights, preprocessConfig);
  } catch (e) {
    throw Exception('failed to load on-device model bundle: $e');
  }
}

void main() {
  runApp(const FundusScreenerApp());
}

class FundusScreenerApp extends StatelessWidget {
  const FundusScreenerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fundus Screener',
      home: FutureBuilder<ModelBundle>(
        future: loadModelBundle(),
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Scaffold(
                body: Center(child: CircularProgressIndicator()));
          }
          if (snapshot.hasError) {
            return ModelErrorScreen(message: snapshot.error.toString());
          }
          return _SessionFlow(modelBundle: snapshot.data!);
        },
      ),
    );
  }
}

/// Owns the session ID, the capture-screen camera source, and the session
/// log for one screening session: SessionIdScreen -> repeated
/// (CaptureScreen -> preprocess/backbone/head -> ResultScreen -> Save)
/// cycles, all against the same session ID.
class _SessionFlow extends StatefulWidget {
  final ModelBundle modelBundle;
  const _SessionFlow({required this.modelBundle});

  @override
  State<_SessionFlow> createState() => _SessionFlowState();
}

class _SessionFlowState extends State<_SessionFlow> {
  String? _sessionId;
  SessionLog? _sessionLog;

  // Memoized so the camera is only opened once per session (a fresh
  // FutureBuilder.future on every build would reopen the camera controller
  // on each rebuild), and the resolved source is kept so it can be
  // disposed when this widget goes away instead of leaking the platform
  // camera resource.
  Future<PluginCameraSource>? _cameraSourceFuture;
  PluginCameraSource? _cameraSource;

  Future<PluginCameraSource> _ensureCameraSource() {
    return _cameraSourceFuture ??= PluginCameraSource.create().then((source) {
      _cameraSource = source;
      return source;
    });
  }

  Future<void> _ensureSessionLog() async {
    if (_sessionLog != null) return;
    final dir = await getApplicationDocumentsDirectory();
    _sessionLog = SessionLog(dir.path);
  }

  Future<void> _onCaptured(Uint8List imageBytes, BuildContext context) async {
    // Never silent: a corrupt JPEG from the camera, a native TFLite runtime
    // error, or any other failure in this synchronous pipeline must not
    // leave the user stuck on CaptureScreen with no feedback — surface it
    // and let them retry the capture instead of losing the exception into
    // an unawaited Future (CaptureScreen fires onCaptured without awaiting
    // it).
    img.Image original;
    Float32List input;
    FeatureMap features;
    HeadForwardResult forward;
    try {
      original = img.decodeImage(imageBytes)!;
      input = preprocessImage(original, widget.modelBundle.preprocessConfig);
      features = widget.modelBundle.engine.run(input);
      forward = headForward(features, widget.modelBundle.weights);
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not process that capture: $e')),
      );
      return;
    }

    if (!context.mounted) return;
    // CaptureScreen stays mounted underneath this pushed route (Navigator
    // push, not replace), so its torch would otherwise stay lit -- aimed at
    // the subject's eye through the DIYretCAM rig -- for the entire result
    // review / Grad-CAM inspection, not just while framing the shot. Turn
    // it off before navigating away and back on only once we're about to
    // show CaptureScreen again for the next capture.
    await _cameraSource?.setTorch(false);
    if (!context.mounted) return;
    final shouldSave = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (ctx) => ResultScreen(
        originalImage: original,
        features: features,
        weights: widget.modelBundle.weights,
        forward: forward,
        onDiseaseSelected: (_) {},
        onSave: () => Navigator.of(ctx).pop(true),
        onRetake: () => Navigator.of(ctx).pop(false),
      ),
    ));
    if (context.mounted) {
      await _cameraSource?.setTorch(true);
    }

    // Only an explicit Save commits anything: a discarded/retaken capture
    // (shouldSave == false, or the user backing out of the result screen)
    // must leave no record and no orphan file. Since the image bytes only
    // exist in memory until this point, "do nothing" already satisfies
    // that guarantee; SessionLog.commitRecord is the sole write path.
    if (shouldSave == true) {
      await _ensureSessionLog();
      final fileName = 'capture_${DateTime.now().microsecondsSinceEpoch}.jpg';
      await _sessionLog!.commitRecord(
        SessionRecord(
          sessionId: _sessionId!,
          timestamp: DateTime.now().toUtc(),
          imageFileName: fileName,
          predictions: forward.probabilities,
          mode: 'local',
          deviceTag: 's25ultra_diyretcam',
        ),
        // Save the original camera JPEG bytes, not a re-encode of the
        // decoded img.Image: re-encoding risks quality loss and drops EXIF
        // orientation data on the exact artifact an ophthalmologist will
        // grade. `original` (the decoded img.Image) is still used above for
        // Grad-CAM overlay rendering/preprocessing -- only the file actually
        // written to SessionLog should be the untouched original bytes.
        imageBytes,
      );
    }
  }

  @override
  void dispose() {
    _cameraSource?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_sessionId == null) {
      return SessionIdScreen(onSubmit: (id) => setState(() => _sessionId = id));
    }

    return FutureBuilder<PluginCameraSource>(
      future: _ensureCameraSource(),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Scaffold(
              body: Center(child: CircularProgressIndicator()));
        }
        return CaptureScreen(
          cameraSource: snapshot.data!,
          onCaptured: (bytes) => _onCaptured(bytes, context),
        );
      },
    );
  }
}
