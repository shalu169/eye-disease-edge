import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/feature_map.dart';
import 'package:fundus_screener/inference/head_weights.dart';
import 'package:fundus_screener/inference/head_math.dart';
import 'package:fundus_screener/models/labels.dart';
import 'package:fundus_screener/screens/result_screen.dart';

HeadWeights _fakeWeights() {
  return HeadWeights(
    // Each hidden unit reads a different channel (unit 3 blends all three)
    // so that selecting a different classifier row yields a genuinely
    // different alpha vector. A uniform 0.5-everywhere matrix (as in a
    // naive fixture) would make every disease's alpha identical, since all
    // hidden units would then compute the exact same value from the same
    // pooled input — no implementation could produce distinct heatmaps
    // from that degenerate case.
    convHeadWeight: [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
      [0.3, 0.3, 0.3],
    ],
    convHeadBias: List.filled(4, 0.0),
    classifierWeight: [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.5, 0.5, 0.5, 0.5],
    ],
    classifierBias: List.filled(5, 0.0),
    labels: kDiseaseLabels,
  );
}

void main() {
  testWidgets('shows one row per disease and a distinct heatmap per tapped row',
      (tester) async {
    final weights = _fakeWeights();
    // Genuine per-pixel (spatial) variation, not just per-channel variation:
    // channel 0 ramps up across the 9 positions, channel 1 ramps down,
    // channel 2 is flat. A feature map that only varies by channel (same
    // vector repeated at every spatial position) would make computeHeatmap's
    // min-max normalization collapse every class's heatmap to a uniform
    // (all-zero) map, regardless of alpha — so tapping different disease
    // rows could never plausibly produce different heatmaps.
    final features = FeatureMap(
        Float32List.fromList(List.generate(3 * 3 * 3, (i) {
          final spatialIdx = i ~/ 3;
          final channel = i % 3;
          if (channel == 0) return spatialIdx.toDouble();
          if (channel == 1) return (8 - spatialIdx).toDouble();
          return 0.0;
        })),
        3, 3, 3);
    final forward = headForward(features, weights);
    final original = img.Image(width: 10, height: 10);

    String? selected;
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(
        originalImage: original,
        features: features,
        weights: weights,
        forward: forward,
        onDiseaseSelected: (d) => selected = d,
        onSave: () {},
        onRetake: () {},
      ),
    ));

    for (final label in kDiseaseLabels) {
      expect(find.text(kDiseaseNames[label]!), findsOneWidget);
    }

    await tester.tap(find.text(kDiseaseNames['D']!));
    await tester.pump();
    expect(selected, 'D');
    expect(find.byKey(const Key('gradcam_overlay_image')), findsOneWidget);
    final bytesD = (tester
            .widget<Image>(find.byKey(const Key('gradcam_overlay_image')))
            .image as MemoryImage)
        .bytes;

    await tester.tap(find.text(kDiseaseNames['G']!));
    await tester.pump();
    expect(selected, 'G');
    // Still exactly one overlay shown (the new one replaced the old one,
    // not stacked), AND its actual pixel content changed — guards against
    // a stale/cached heatmap from the previous tap being redisplayed.
    expect(find.byKey(const Key('gradcam_overlay_image')), findsOneWidget);
    final bytesG = (tester
            .widget<Image>(find.byKey(const Key('gradcam_overlay_image')))
            .image as MemoryImage)
        .bytes;
    expect(bytesD, isNot(equals(bytesG)));
  });

  testWidgets('Save button calls onSave and Retake calls onRetake, never both',
      (tester) async {
    final weights = _fakeWeights();
    final features = FeatureMap(
        Float32List.fromList(List.generate(3 * 3 * 3, (i) => (i % 3).toDouble())),
        3, 3, 3);
    final forward = headForward(features, weights);

    var saved = false;
    var retaken = false;
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(
        originalImage: img.Image(width: 10, height: 10),
        features: features,
        weights: weights,
        forward: forward,
        onDiseaseSelected: (_) {},
        onSave: () => saved = true,
        onRetake: () => retaken = true,
      ),
    ));

    await tester.tap(find.byKey(const Key('retake_button')));
    await tester.pump();
    expect(retaken, isTrue);
    expect(saved, isFalse);
  });
}
