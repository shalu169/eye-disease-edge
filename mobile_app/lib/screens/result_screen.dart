import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import '../inference/feature_map.dart';
import '../inference/head_weights.dart';
import '../inference/head_math.dart';
import '../inference/gradcam.dart';
import '../inference/heatmap_overlay.dart';
import '../models/labels.dart';

class ResultScreen extends StatefulWidget {
  final img.Image originalImage;
  final FeatureMap features;
  final HeadWeights weights;
  final HeadForwardResult forward;
  final void Function(String disease) onDiseaseSelected;
  final void Function() onSave;
  final void Function() onRetake;

  const ResultScreen({
    super.key,
    required this.originalImage,
    required this.features,
    required this.weights,
    required this.forward,
    required this.onDiseaseSelected,
    required this.onSave,
    required this.onRetake,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  Uint8List? _overlayPngBytes;

  void _selectDisease(String label) {
    final classIndex = widget.weights.labels.indexOf(label);
    final alpha = gradCamAlpha(widget.forward, widget.weights, classIndex);
    final heatmap = computeHeatmap(widget.features, alpha);
    final overlaid = renderOverlay(widget.originalImage, heatmap);

    setState(() {
      _overlayPngBytes = Uint8List.fromList(img.encodePng(overlaid));
    });
    widget.onDiseaseSelected(label);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Screening result')),
      body: Column(
        children: [
          if (_overlayPngBytes != null)
            ConstrainedBox(
              constraints: BoxConstraints(
                maxHeight: MediaQuery.of(context).size.height * 0.4,
              ),
              child: Image.memory(
                _overlayPngBytes!,
                key: const Key('gradcam_overlay_image'),
                fit: BoxFit.contain,
              ),
            ),
          Expanded(
            child: ListView(
              children: kDiseaseLabels.map((label) {
                final prob = widget.forward.probabilities[label] ?? 0.0;
                return ListTile(
                  title: Text(kDiseaseNames[label]!),
                  trailing: Text('${(prob * 100).toStringAsFixed(1)}%'),
                  onTap: () => _selectDisease(label),
                );
              }).toList(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                OutlinedButton(
                  key: const Key('retake_button'),
                  onPressed: widget.onRetake,
                  child: const Text('Retake'),
                ),
                ElevatedButton(
                  key: const Key('save_button'),
                  onPressed: widget.onSave,
                  child: const Text('Save & continue'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
