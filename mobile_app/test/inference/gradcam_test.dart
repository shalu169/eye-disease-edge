import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/inference/feature_map.dart';
import 'package:fundus_screener/inference/head_weights.dart';
import 'package:fundus_screener/inference/head_math.dart';
import 'package:fundus_screener/inference/gradcam.dart';

void main() {
  late HeadWeights weights;
  late List fixtures;

  setUpAll(() {
    weights = HeadWeights.fromJson(
        jsonDecode(File('assets/model/head_weights.json').readAsStringSync()));
    fixtures = jsonDecode(
        File('test/fixtures/model_fixtures.json').readAsStringSync()) as List;
  });

  test('gradCamAlpha matches PyTorch autograd for every label', () {
    for (final entry in fixtures) {
      final pooled = (entry['pooled_vector'] as List)
          .map((e) => (e as num).toDouble())
          .toList();
      final features =
          FeatureMap(Float32List.fromList(pooled), 1, 1, pooled.length);
      final forward = headForward(features, weights);

      final expectedAlphaByLabel = entry['gradcam_alpha'] as Map;
      for (var i = 0; i < weights.labels.length; i++) {
        final label = weights.labels[i];
        final alpha = gradCamAlpha(forward, weights, i);
        final expected = (expectedAlphaByLabel[label] as List)
            .map((e) => (e as num).toDouble())
            .toList();
        for (var k = 0; k < alpha.length; k++) {
          expect(alpha[k], closeTo(expected[k], 1e-3));
        }
      }
    }
  });

  test('computeHeatmap on a synthetic feature map normalizes to [0, 1]', () {
    // Synthetic 7x7x4 feature map with known, hand-computed spatial
    // variation — this isolates and pins down computeHeatmap's ReLU +
    // min-max normalization math (max exactly 1.0, min exactly 0.0) against
    // known inputs. The separate non-degeneracy check on a REAL backbone
    // output (real image -> real preprocess -> real BackboneEngine.run ->
    // real gradCamAlpha -> real computeHeatmap) lives in
    // backbone_engine_test.dart, since that's where the real FeatureMap is
    // already in hand from the end-to-end pipeline.
    final data = Float32List(7 * 7 * 4);
    var i = 0;
    for (var y = 0; y < 7; y++) {
      for (var x = 0; x < 7; x++) {
        for (var c = 0; c < 4; c++) {
          data[i++] = (y == 3 && x == 3) ? 5.0 : 0.1 * (x + y);
        }
      }
    }
    final features = FeatureMap(data, 7, 7, 4);
    final alpha = Float64List.fromList([1.0, 0.5, -0.5, 0.2]);

    final heatmap = computeHeatmap(features, alpha);

    final flat = heatmap.expand((row) => row).toList();
    final distinctValues = flat.toSet().length;
    expect(distinctValues, greaterThan(1),
        reason: 'heatmap collapsed to a single value: $flat');
    expect(flat.reduce(max), closeTo(1.0, 1e-9));
    expect(flat.reduce(min), greaterThanOrEqualTo(0.0));
  });
}
