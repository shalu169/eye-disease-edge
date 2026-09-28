import 'dart:typed_data';
import 'package:flutter/material.dart';

abstract class CameraSource {
  Widget buildPreview();
  Future<Uint8List> takePicture();
  Future<void> setTorch(bool enabled);
}

class CaptureScreen extends StatefulWidget {
  final CameraSource cameraSource;
  final void Function(Uint8List imageBytes) onCaptured;

  const CaptureScreen({
    super.key,
    required this.cameraSource,
    required this.onCaptured,
  });

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  bool _isCapturing = false;

  @override
  void initState() {
    super.initState();
    widget.cameraSource.setTorch(true);
  }

  Future<void> _onShutterPressed() async {
    // Guard against a double-tap firing two concurrent takePicture() calls
    // on the platform camera (which can throw or produce two captures for
    // one shutter press).
    if (_isCapturing) return;
    _isCapturing = true;
    try {
      final bytes = await widget.cameraSource.takePicture();
      widget.onCaptured(bytes);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not capture photo: $e')),
      );
    } finally {
      _isCapturing = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          widget.cameraSource.buildPreview(),
          Center(
            child: Container(
              key: const Key('framing_guide'),
              width: 240,
              height: 240,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.greenAccent, width: 3),
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 32.0),
              child: FloatingActionButton(
                key: const Key('shutter_button'),
                onPressed: _onShutterPressed,
                child: const Icon(Icons.camera),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
