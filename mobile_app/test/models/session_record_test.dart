import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/models/session_record.dart';

void main() {
  test('SessionRecord round-trips through JSON', () {
    final record = SessionRecord(
      sessionId: 'patient-07',
      timestamp: DateTime.utc(2026, 9, 27, 10, 30),
      imageFileName: 'capture_001.jpg',
      predictions: {'D': 0.12, 'G': 0.03, 'C': 0.44, 'A': 0.02, 'H': 0.01},
      mode: 'local',
      deviceTag: 's25ultra_diyretcam',
    );

    final json = record.toJson();
    final restored = SessionRecord.fromJson(json);

    expect(restored.sessionId, 'patient-07');
    expect(restored.timestamp, DateTime.utc(2026, 9, 27, 10, 30));
    expect(restored.imageFileName, 'capture_001.jpg');
    expect(restored.predictions['C'], 0.44);
    expect(restored.mode, 'local');
    expect(restored.deviceTag, 's25ultra_diyretcam');
  });
}
