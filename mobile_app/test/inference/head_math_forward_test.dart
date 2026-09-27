import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/inference/feature_map.dart';
import 'package:fundus_screener/inference/head_weights.dart';
import 'package:fundus_screener/inference/head_math.dart';

void main() {
  test('headForward reproduces PyTorch logits and probabilities', () {
    final weights = HeadWeights.fromJson(
        jsonDecode(File('assets/model/head_weights.json').readAsStringSync()));
    final fixtures = jsonDecode(
        File('test/fixtures/model_fixtures.json').readAsStringSync()) as List;

    for (final entry in fixtures) {
      final pooledExpected =
          (entry['pooled_vector'] as List).map((e) => (e as num).toDouble()).toList();

      // Build a FeatureMap whose spatial average equals pooledExpected exactly:
      // a 1x1 spatial feature map IS its own average, so this isolates head
      // math from the separate backbone/pooling numerics already covered by
      // Task 3's parity test.
      final data = Float32List.fromList(pooledExpected);
      final features = FeatureMap(data, 1, 1, pooledExpected.length);

      final result = headForward(features, weights);

      final expectedLogits =
          (entry['logits'] as List).map((e) => (e as num).toDouble()).toList();
      final expectedProbs =
          (entry['probs'] as List).map((e) => (e as num).toDouble()).toList();

      for (var i = 0; i < weights.labels.length; i++) {
        expect(result.logits[i], closeTo(expectedLogits[i], 1e-3));
        expect(result.probabilities[weights.labels[i]]!,
            closeTo(expectedProbs[i], 1e-3));
      }
    }
  });
}
