import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/main.dart';

void main() {
  // Loading the tflite backbone goes through Interpreter.fromAsset, which
  // reads via Flutter's rootBundle/platform-asset-channel machinery. That
  // requires a binding to be initialized even for a plain (non-widget)
  // test, matching the convention already used in
  // test/inference/backbone_engine_test.dart.
  TestWidgetsFlutterBinding.ensureInitialized();

  test('loadModelBundle throws a descriptive error for a missing asset path',
      () async {
    expect(
      () => loadModelBundle(modelDirOverride: 'assets/does_not_exist'),
      throwsA(isA<Exception>().having(
          (e) => e.toString(), 'message', contains('model'))),
    );
  });

  test('loadModelBundle succeeds against the real bundled assets', () async {
    final bundle = await loadModelBundle();

    // Sanity-check that every piece actually parsed/loaded, not just that
    // no exception was thrown: a genuinely wired-up bundle should have a
    // usable interpreter, a positive image size, and at least one disease
    // label with matching classifier weights.
    expect(bundle.preprocessConfig.imageSize, greaterThan(0));
    expect(bundle.weights.labels, isNotEmpty);
    expect(bundle.weights.classifierBias.length, bundle.weights.labels.length);

    bundle.engine.close();
  });
}
