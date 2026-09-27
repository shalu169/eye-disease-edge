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
}
