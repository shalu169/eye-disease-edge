import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/heatmap_overlay.dart';

void main() {
  test('renderOverlay upsamples heatmap to image size and tints hot pixels', () {
    final original = img.Image(width: 28, height: 28);
    img.fill(original, color: img.ColorRgb8(50, 50, 50));

    // Heatmap hot only at bottom-right corner.
    final heatmap = List.generate(
        7, (y) => List<double>.generate(7, (x) => (y == 6 && x == 6) ? 1.0 : 0.0));

    final overlaid = renderOverlay(original, heatmap);

    expect(overlaid.width, 28);
    expect(overlaid.height, 28);

    final hotPixel = overlaid.getPixel(27, 27);
    final coldPixel = overlaid.getPixel(0, 0);

    // Hot corner should shift toward red relative to the untouched corner.
    expect(hotPixel.r, greaterThan(coldPixel.r));
    // Original image must not be mutated.
    expect(original.getPixel(27, 27).r, 50);
  });
}
