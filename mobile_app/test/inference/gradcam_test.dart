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

  test('computeHeatmap on a real feature map is not degenerate', () {
    // Synthetic 7x7x4 feature map with real spatial variation, not a
    // fixture-only check — guards against a heatmap that's technically
    // fixture-correct but collapses to a flat map on real inputs.
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
