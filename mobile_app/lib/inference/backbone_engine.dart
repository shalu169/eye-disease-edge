import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'feature_map.dart';

class BackboneEngine {
  Interpreter? _interpreter;

  Future<void> load(String assetPath) async {
    _interpreter = await Interpreter.fromAsset(assetPath);
  }

  FeatureMap run(Float32List preprocessedInput) {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('BackboneEngine.load() must complete before run()');
    }

    final input = preprocessedInput.reshape([1, 224, 224, 3]);
    final output = List.generate(
        1, (_) => List.generate(7, (_) => List.generate(7, (_) => List.filled(576, 0.0))));

    interpreter.run(input, output);

    final flat = Float32List(7 * 7 * 576);
    var idx = 0;
    for (var y = 0; y < 7; y++) {
      for (var x = 0; x < 7; x++) {
        for (var c = 0; c < 576; c++) {
          flat[idx++] = output[0][y][x][c];
        }
      }
    }
    return FeatureMap(flat, 7, 7, 576);
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}
