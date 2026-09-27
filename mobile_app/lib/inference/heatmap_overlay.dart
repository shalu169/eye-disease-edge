import 'package:image/image.dart' as img;

double _bilinearSample(List<List<double>> map, double y, double x) {
  final h = map.length;
  final w = map[0].length;
  final y0 = y.floor().clamp(0, h - 1);
  final x0 = x.floor().clamp(0, w - 1);
  final y1 = (y0 + 1).clamp(0, h - 1);
  final x1 = (x0 + 1).clamp(0, w - 1);
  final fy = y - y0;
  final fx = x - x0;

  final top = map[y0][x0] * (1 - fx) + map[y0][x1] * fx;
  final bottom = map[y1][x0] * (1 - fx) + map[y1][x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

img.Image renderOverlay(img.Image original, List<List<double>> heatmap,
    {double opacity = 0.5}) {
  final result = img.Image.from(original);
  final srcH = heatmap.length;
  final srcW = heatmap[0].length;

  for (var y = 0; y < result.height; y++) {
    // Map output pixel to heatmap coordinate space.
    final srcY = (y / (result.height - 1)) * (srcH - 1);
    for (var x = 0; x < result.width; x++) {
      final srcX = (x / (result.width - 1)) * (srcW - 1);
      final value = _bilinearSample(heatmap, srcY, srcX).clamp(0.0, 1.0);

      final heatR = (255 * value).round();
      final heatG = (255 * (1 - value) * 0.3).round();
      const heatB = 0;

      final basePixel = original.getPixel(x, y);
      final blendedR = (basePixel.r * (1 - opacity) + heatR * opacity).round();
      final blendedG = (basePixel.g * (1 - opacity) + heatG * opacity).round();
      final blendedB = (basePixel.b * (1 - opacity) + heatB * opacity).round();

      result.setPixelRgb(x, y, blendedR, blendedG, blendedB);
    }
  }
  return result;
}
