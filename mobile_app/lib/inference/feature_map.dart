import 'dart:typed_data';

class FeatureMap {
  final Float32List data; // flat, NHWC: index = (y*width + x)*channels + c
  final int height;
  final int width;
  final int channels;

  FeatureMap(this.data, this.height, this.width, this.channels);

  double at(int y, int x, int c) => data[(y * width + x) * channels + c];

  /// Global average pool over spatial dims -> one value per channel.
  Float64List globalAveragePool() {
    final pooled = Float64List(channels);
    final count = height * width;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        for (var c = 0; c < channels; c++) {
          pooled[c] += at(y, x, c);
        }
      }
    }
    for (var c = 0; c < channels; c++) {
      pooled[c] /= count;
    }
    return pooled;
  }
}
