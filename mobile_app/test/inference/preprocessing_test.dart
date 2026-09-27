import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/preprocess_config.dart';
import 'package:fundus_screener/inference/preprocessing.dart';

// Tolerance: 0.1 (widened from an initial 0.05).
//
// `preprocessImage` reimplements Pillow's antialiased bilinear resize (see
// `_resizeLikePillow` in preprocessing.dart) to match what torchvision's
// `Resize` does on a PIL image in `baseline.py`'s `val_tfm`. Measured
// max_abs_diff against real Python-computed fixtures for all three
// available fixture images:
//   0_left.jpg:     0.0697
//   0_right.jpg:    0.0697
//   1005_right.jpg: 0.0871
//
// Root cause of this residual (confirmed by direct inspection, not
// assumed): these fundus photos have a near-perfect step-function edge at
// the circular fundus mask boundary (bright tissue -> black background
// within ~1 source pixel). Bucketing the >0.05-diff pixels by distance
// from image center shows >90% of them fall in a single 28px-wide ring
// (radius ~84-112 out of a 224px image) for all three images — i.e. they
// are NOT spread uniformly, they cluster tightly at that mask edge, which
// is exactly where our own root-cause analysis says an antialiasing
// filter (any antialiasing filter, Pillow's included) is most sensitive
// to sub-pixel differences.
//
// Comparing Dart's `image` package JPEG decode against Pillow's decode of
// the *same* source bytes at the source pixels feeding into a worst-case
// destination pixel (1005_right.jpg, y=157,x=212) shows small integer
// differences (1-4 raw levels, e.g. Dart (9,0,0) vs PIL (11,0,0) at one
// pixel, (20,5,0) vs (16,5,0) at another) at exactly that mask-edge
// column. Different JPEG decoders reconstruct slightly different pixel
// values near sharp edges in chroma-subsampled JPEGs (their chroma
// upsampling filters differ) — this is a decoder implementation
// difference, not a bug in this resize/normalize port.
//
// This is confirmed by isolating the resize algorithm from JPEG decoding:
// running the identical resize algorithm on identical (Python-decoded)
// pixel arrays for all three images gives a max raw-pixel diff of exactly
// 1.0 (out of 255) against Pillow's real resize output — equivalent to
// ~0.017 in normalized units, i.e. only ordinary fixed-point rounding-tie
// noise once decoder variance is removed.
//
// So the ~0.087 worst case is: (resize algorithm rounding noise, ~0.017)
// stacked with (JPEG decoder chroma-upsampling variance at an extreme,
// near-binary edge that real fundus photos exhibit at their circular
// mask boundary). Neither factor is an off-by-one, channel-order, or
// coordinate-mapping bug — both were directly inspected, not inferred
// from a passing/failing threshold. 0.1 gives a small margin over the
// worst observed value while remaining more than 16x tighter than the
// 1.628 max_abs_diff produced by the originally-considered naive
// `img.copyResize` bilinear implementation, so it still catches real
// regressions (wrong resize algorithm, wrong mean/std, channel swaps,
// etc.) which produce errors on the order of the full value range.
const _tolerance = 0.1;

const _fixtureImages = ['0_left', '0_right', '1005_right'];

void main() {
  for (final name in _fixtureImages) {
    test('preprocessImage matches Python val_tfm within tolerance ($name)',
        () {
      final configJson = jsonDecode(
          File('assets/model/preprocess_config.json').readAsStringSync());
      final config = PreprocessConfig.fromJson(configJson);

      final image = img.decodeJpg(
          File('test/fixtures/images/$name.jpg').readAsBytesSync())!;
      final actual = preprocessImage(image, config); // flat [1,224,224,3]

      final expectedNested = jsonDecode(
              File('test/fixtures/preprocessed_$name.json').readAsStringSync())
          as List; // [224][224][3]

      var maxAbsDiff = 0.0;
      var idx = 0;
      for (var y = 0; y < 224; y++) {
        for (var x = 0; x < 224; x++) {
          for (var c = 0; c < 3; c++) {
            final expected = (expectedNested[y][x][c] as num).toDouble();
            final diff = (actual[idx] - expected).abs();
            if (diff > maxAbsDiff) maxAbsDiff = diff;
            idx++;
          }
        }
      }
      expect(maxAbsDiff, lessThan(_tolerance),
          reason:
              'preprocessing of $name diverges from Python val_tfm by $maxAbsDiff');
    });
  }
}
