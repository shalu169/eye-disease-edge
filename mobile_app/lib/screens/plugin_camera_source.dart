import 'dart:io';
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
    try {
      return await file.readAsBytes();
    } finally {
      // The camera plugin writes a real temp JPEG to the platform cache dir
      // (e.g. Android's cache dir) for every capture, including
      // retaken/discarded ones. Deleting it here (after the bytes are
      // safely read into memory) is what makes "image bytes only exist in
      // memory until an explicit Save" (see main.dart's _onCaptured) and
      // "retake leaves no trace" actually true on a real device -- without
      // this, every capture of a patient's retina would leave a stray JPEG
      // sitting in app cache indefinitely.
      await File(file.path).delete();
    }
  }

  Future<void> dispose() async {
    await controller.dispose();
  }
}
