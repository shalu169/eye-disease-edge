import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/screens/capture_screen.dart';

class FakeCameraSource implements CameraSource {
  bool torchEnabled = false;
  bool pictureTaken = false;

  @override
  Widget buildPreview() => const ColoredBox(color: Colors.black, child: SizedBox.expand());

  @override
  Future<void> setTorch(bool enabled) async {
    torchEnabled = enabled;
  }

  @override
  Future<Uint8List> takePicture() async {
    pictureTaken = true;
    return Uint8List.fromList([42]);
  }
}

void main() {
  testWidgets('forces torch on at startup and shows a framing guide', (tester) async {
    final source = FakeCameraSource();
    await tester.pumpWidget(MaterialApp(
      home: CaptureScreen(cameraSource: source, onCaptured: (_) {}),
    ));
    await tester.pump();

    expect(source.torchEnabled, isTrue);
    expect(find.byKey(const Key('framing_guide')), findsOneWidget);
  });

  testWidgets('tapping the shutter takes a picture and calls onCaptured',
      (tester) async {
    final source = FakeCameraSource();
    Uint8List? captured;
    await tester.pumpWidget(MaterialApp(
      home: CaptureScreen(
        cameraSource: source,
        onCaptured: (bytes) => captured = bytes,
      ),
    ));
    await tester.pump();

    await tester.tap(find.byKey(const Key('shutter_button')));
    await tester.pumpAndSettle();

    expect(source.pictureTaken, isTrue);
    expect(captured, Uint8List.fromList([42]));
  });
}
