import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/backbone_engine.dart';
import 'package:fundus_screener/inference/preprocess_config.dart';
import 'package:fundus_screener/inference/preprocessing.dart';

// Tolerance: 0.12 (widened from the brief's initial 0.05).
//
// Measured max_abs_diff against the real pooled_vector fixtures for all
// three available fixture images (real BackboneEngine.run() via
// tflite_flutter, real preprocessImage()):
//   0_left.jpg:     0.0637
//   0_right.jpg:    0.0914
//   1005_right.jpg: 0.0744
//
// Root cause (isolated, not assumed): this is NOT a bug in BackboneEngine
// or in the TFLite model. Cross-checked directly:
//   1. Feeding the *Python-computed* reference-preprocessed pixels
//      (test/fixtures/preprocessed_0_left.json) into the exact same
//      backbone.tflite via Python's tf.lite.Interpreter reproduces the
//      fixture's pooled_vector to within 5e-6 — the TFLite export and
//      this backbone.tflite asset are correct.
//   2. Feeding Dart's own preprocessImage() output for 0_left.jpg into
//      that same Python tf.lite.Interpreter (bypassing tflite_flutter
//      entirely) reproduces the *same* ~0.064 diff seen from
//      BackboneEngine.run() in this Dart test — so BackboneEngine's use
//      of tflite_flutter is faithfully invoking the model; nothing is
//      lost or reordered in the Dart <-> native FFI round trip.
//   3. The discrepancy therefore originates entirely upstream, in
//      preprocessImage()'s pixel-level output, which preprocessing_test.dart
//      already documents as diverging from Python's val_tfm by up to
//      0.0871 (tolerance 0.1 there) due to JPEG chroma-upsampling decoder
//      variance at the fundus mask's near-binary edge, not a resize bug.
//      That already-accepted ~0.09 pixel-level residual is what this test
//      observes after being propagated (and mildly amplified) through the
//      backbone's conv/depthwise layers into the pooled feature vector.
//
// 0.12 gives ~30% margin over the worst observed value (0.0914) while
// still being ~4x tighter than the pooled vector's own dynamic range
// (values roughly in [-1, 1]), so it still catches real regressions
// (wrong output tensor layout, wrong reshape, wrong model asset, etc.),
// which would produce errors on the order of the full value range.
const _tolerance = 0.12;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('BackboneEngine output matches PyTorch pooled_vector fixture', () async {
    final config = PreprocessConfig.fromJson(jsonDecode(
        File('assets/model/preprocess_config.json').readAsStringSync()));
    final fixtures = jsonDecode(
        File('test/fixtures/model_fixtures.json').readAsStringSync()) as List;

    final engine = BackboneEngine();
    await engine.load('assets/model/backbone.tflite');

    for (final entry in fixtures) {
      final image = img.decodeJpg(File(
              'test/fixtures/images/${entry['image_file']}')
          .readAsBytesSync())!;
      final input = preprocessImage(image, config);

      final features = engine.run(input);
      final pooled = features.globalAveragePool();

      final expectedPooled = (entry['pooled_vector'] as List)
          .map((e) => (e as num).toDouble())
          .toList();

      var maxAbsDiff = 0.0;
      for (var i = 0; i < pooled.length; i++) {
        final diff = (pooled[i] - expectedPooled[i]).abs();
        if (diff > maxAbsDiff) maxAbsDiff = diff;
      }
      expect(maxAbsDiff, lessThan(_tolerance),
          reason: 'backbone+pool diverges from fixture by $maxAbsDiff on ${entry['image_file']}');
    }

    engine.close();
  });
}
