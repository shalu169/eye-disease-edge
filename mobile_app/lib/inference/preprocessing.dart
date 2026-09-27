import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'preprocess_config.dart';

/// Precomputed per-output-index resampling weights for one axis (rows or
/// columns), matching Pillow's `precompute_coeffs` for a triangle
/// ("bilinear") filter. When downscaling (srcSize > dstSize), the filter
/// support is widened by the scale factor, which is what gives PIL/
/// torchvision's `Resize` its antialiasing: each output sample is a
/// weighted average of several source samples rather than a plain 2-tap
/// bilinear lookup.
class _AxisWeights {
  final List<int> start; // first source index contributing to each output index
  final List<int> length; // number of source samples contributing
  final List<List<double>> weights; // per-output-index tap weights

  _AxisWeights(this.start, this.length, this.weights);

  factory _AxisWeights.compute(int srcSize, int dstSize) {
    const filterSupport = 1.0; // triangle filter half-width
    final scale = srcSize / dstSize;
    final filterScale = scale > 1.0 ? scale : 1.0;
    final support = filterSupport * filterScale;
    final kernelSize = support.ceil() * 2 + 1;

    final start = List<int>.filled(dstSize, 0);
    final length = List<int>.filled(dstSize, 0);
    final weights = List<List<double>>.generate(
        dstSize, (_) => List<double>.filled(kernelSize, 0.0));

    final invFilterScale = 1.0 / filterScale;
    for (var o = 0; o < dstSize; o++) {
      // Pixel-center convention: the destination pixel's center maps to
      // (o + 0.5) * scale in source space (matches PIL, not "corner
      // aligned" scaling).
      final center = (o + 0.5) * scale;

      var srcMin = (center - support + 0.5).floor();
      if (srcMin < 0) srcMin = 0;
      var srcMax = (center + support + 0.5).floor();
      if (srcMax > srcSize) srcMax = srcSize;
      final len = srcMax - srcMin;

      final row = weights[o];
      var weightSum = 0.0;
      for (var i = 0; i < len; i++) {
        final x = (i + srcMin - center + 0.5) * invFilterScale;
        final absX = x.abs();
        final w = absX < 1.0 ? 1.0 - absX : 0.0;
        row[i] = w;
        weightSum += w;
      }
      if (weightSum != 0.0) {
        for (var i = 0; i < len; i++) {
          row[i] /= weightSum;
        }
      }
      start[o] = srcMin;
      length[o] = len;
    }
    return _AxisWeights(start, length, weights);
  }
}

double _clampByte(double v) => v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);

/// Resizes [image] to [dstWidth]x[dstHeight] reproducing Pillow's
/// antialiased bilinear resize (horizontal pass, then vertical pass, with
/// intermediate results rounded/clamped to 8-bit at each stage), which is
/// what `torchvision.transforms.Resize` uses under the hood for a PIL
/// image (as in `baseline.py`'s `val_tfm`). The `image` package's built-in
/// `copyResize(interpolation: linear)` does a plain corner-aligned 2-tap
/// bilinear lookup with no antialiasing, which diverges sharply from PIL
/// at high-contrast edges when downscaling by a non-integer factor —
/// hence this dedicated implementation instead.
List<List<Float64List>> _resizeLikePillow(
    img.Image image, int dstWidth, int dstHeight) {
  final srcWidth = image.width;
  final srcHeight = image.height;

  final xWeights = _AxisWeights.compute(srcWidth, dstWidth);
  final yWeights = _AxisWeights.compute(srcHeight, dstHeight);

  // Read source pixels once into row-major [row][col] arrays of [r,g,b].
  final src = List<List<Float64List>>.generate(
      srcHeight,
      (y) => List<Float64List>.generate(srcWidth, (x) {
            final p = image.getPixel(x, y);
            return Float64List.fromList(
                [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()]);
          }));

  // Horizontal pass: srcHeight x dstWidth x 3.
  final horizontal = List<List<Float64List>>.generate(
      srcHeight, (_) => List<Float64List>.generate(dstWidth, (_) => Float64List(3)));
  for (var y = 0; y < srcHeight; y++) {
    final srcRow = src[y];
    final dstRow = horizontal[y];
    for (var ox = 0; ox < dstWidth; ox++) {
      final startX = xWeights.start[ox];
      final len = xWeights.length[ox];
      final w = xWeights.weights[ox];
      var r = 0.0, g = 0.0, b = 0.0;
      for (var i = 0; i < len; i++) {
        final px = srcRow[startX + i];
        final wi = w[i];
        r += px[0] * wi;
        g += px[1] * wi;
        b += px[2] * wi;
      }
      final out = dstRow[ox];
      out[0] = _clampByte(r.roundToDouble());
      out[1] = _clampByte(g.roundToDouble());
      out[2] = _clampByte(b.roundToDouble());
    }
  }

  // Vertical pass: dstHeight x dstWidth x 3.
  final result = List<List<Float64List>>.generate(
      dstHeight, (_) => List<Float64List>.generate(dstWidth, (_) => Float64List(3)));
  for (var oy = 0; oy < dstHeight; oy++) {
    final startY = yWeights.start[oy];
    final len = yWeights.length[oy];
    final w = yWeights.weights[oy];
    final dstRow = result[oy];
    for (var x = 0; x < dstWidth; x++) {
      var r = 0.0, g = 0.0, b = 0.0;
      for (var i = 0; i < len; i++) {
        final px = horizontal[startY + i][x];
        final wi = w[i];
        r += px[0] * wi;
        g += px[1] * wi;
        b += px[2] * wi;
      }
      final out = dstRow[x];
      out[0] = _clampByte(r.roundToDouble());
      out[1] = _clampByte(g.roundToDouble());
      out[2] = _clampByte(b.roundToDouble());
    }
  }
  return result;
}

/// Resizes [image] to `config.imageSize` square and normalizes it,
/// matching `baseline.py`'s `val_tfm`:
/// `Resize((224,224))` + `ToTensor()` + `Normalize(mean, std)`.
/// Returns a flat NHWC `[1, imageSize, imageSize, 3]` float32 buffer.
Float32List preprocessImage(img.Image image, PreprocessConfig config) {
  final resized =
      _resizeLikePillow(image, config.imageSize, config.imageSize);

  final buffer = Float32List(config.imageSize * config.imageSize * 3);
  var idx = 0;
  for (var y = 0; y < config.imageSize; y++) {
    final row = resized[y];
    for (var x = 0; x < config.imageSize; x++) {
      final px = row[x];
      final r = px[0] / 255.0;
      final g = px[1] / 255.0;
      final b = px[2] / 255.0;
      buffer[idx++] = (r - config.mean[0]) / config.std[0];
      buffer[idx++] = (g - config.mean[1]) / config.std[1];
      buffer[idx++] = (b - config.mean[2]) / config.std[2];
    }
  }
  return buffer;
}
