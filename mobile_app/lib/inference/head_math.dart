import 'dart:math';
import 'dart:typed_data';
import 'feature_map.dart';
import 'head_weights.dart';

double hardswish(double x) {
  if (x <= -3) return 0.0;
  if (x >= 3) return x;
  return x * (x + 3) / 6.0;
}

double hardswishDerivative(double x) {
  if (x <= -3) return 0.0;
  if (x >= 3) return 1.0;
  return (2 * x + 3) / 6.0;
}

double _sigmoid(double x) => 1.0 / (1.0 + exp(-x));

class HeadForwardResult {
  final Float64List pooled;        // [576]
  final Float64List preActivation; // [1024] (before hardswish)
  final Float64List activated;     // [1024] (after hardswish)
  final Float64List logits;        // [5]
  final Map<String, double> probabilities;

  HeadForwardResult({
    required this.pooled,
    required this.preActivation,
    required this.activated,
    required this.logits,
    required this.probabilities,
  });
}

HeadForwardResult headForward(FeatureMap features, HeadWeights weights) {
  final pooled = features.globalAveragePool(); // [576]

  final preAct = Float64List(weights.convHeadBias.length); // [1024]
  for (var m = 0; m < preAct.length; m++) {
    var sum = weights.convHeadBias[m];
    final row = weights.convHeadWeight[m];
    for (var k = 0; k < pooled.length; k++) {
      sum += row[k] * pooled[k];
    }
    preAct[m] = sum;
  }

  final activated = Float64List(preAct.length);
  for (var m = 0; m < preAct.length; m++) {
    activated[m] = hardswish(preAct[m]);
  }

  final logits = Float64List(weights.classifierBias.length); // [5]
  for (var c = 0; c < logits.length; c++) {
    var sum = weights.classifierBias[c];
    final row = weights.classifierWeight[c];
    for (var m = 0; m < activated.length; m++) {
      sum += row[m] * activated[m];
    }
    logits[c] = sum;
  }

  final probabilities = <String, double>{};
  for (var c = 0; c < weights.labels.length; c++) {
    probabilities[weights.labels[c]] = _sigmoid(logits[c]);
  }

  return HeadForwardResult(
    pooled: pooled,
    preActivation: preAct,
    activated: activated,
    logits: logits,
    probabilities: probabilities,
  );
}
