import 'dart:math';
import 'dart:typed_data';
import 'feature_map.dart';
import 'head_weights.dart';
import 'head_math.dart';

/// d(logit[classIndex]) / d(pooled), via chain rule through the two-layer
/// head: logit = classifier(hardswish(conv_head(pooled))).
Float64List gradCamAlpha(
    HeadForwardResult forward, HeadWeights weights, int classIndex) {
  final classifierRow = weights.classifierWeight[classIndex]; // [1024]
  final numHidden = classifierRow.length;
  final numPooled = forward.pooled.length;

  // d(logit)/d(activated) = classifierRow, then chain through hardswish'.
  final dLogitDPreAct = Float64List(numHidden);
  for (var m = 0; m < numHidden; m++) {
    dLogitDPreAct[m] =
        classifierRow[m] * hardswishDerivative(forward.preActivation[m]);
  }

  // d(logit)/d(pooled)[k] = sum_m dLogitDPreAct[m] * convHeadWeight[m][k]
  final alpha = Float64List(numPooled);
  for (var m = 0; m < numHidden; m++) {
    final row = weights.convHeadWeight[m];
    final grad = dLogitDPreAct[m];
    for (var k = 0; k < numPooled; k++) {
      alpha[k] += grad * row[k];
    }
  }
  return alpha;
}

List<List<double>> computeHeatmap(FeatureMap features, Float64List alpha) {
  final raw = List.generate(
      features.height, (_) => List<double>.filled(features.width, 0.0));

  for (var y = 0; y < features.height; y++) {
    for (var x = 0; x < features.width; x++) {
      var sum = 0.0;
      for (var c = 0; c < features.channels; c++) {
        sum += alpha[c] * features.at(y, x, c);
      }
      raw[y][x] = max(0.0, sum); // ReLU
    }
  }

  var minVal = double.infinity;
  var maxVal = -double.infinity;
  for (final row in raw) {
    for (final v in row) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }
  final range = (maxVal - minVal).abs() < 1e-9 ? 1.0 : (maxVal - minVal);

  return raw
      .map((row) => row.map((v) => (v - minVal) / range).toList())
      .toList();
}
