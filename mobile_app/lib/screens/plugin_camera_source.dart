import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/widgets.dart';
import 'capture_screen.dart';

class PluginCameraSource implements CameraSource {
  final CameraController controller;

  PluginCameraSource(this.controller);

  static Future<PluginCameraSource> create() async {
    final cameras = await availableCameras();
    final backCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );
    final controller = CameraController(backCamera, ResolutionPreset.high, enableAudio: false);
    await controller.initialize();
    return PluginCameraSource(controller);
  }

  @override
  Widget buildPreview() => CameraPreview(controller);

  @override
  Future<void> setTorch(bool enabled) async {
    await controller.setFlashMode(enabled ? FlashMode.torch : FlashMode.off);
  }

  @override
  Future<Uint8List> takePicture() async {
    final file = await controller.takePicture();
    return file.readAsBytes();
  }

  Future<void> dispose() async {
    await controller.dispose();
  }
}
