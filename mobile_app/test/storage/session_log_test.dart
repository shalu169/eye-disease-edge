import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:archive/archive_io.dart';
import 'package:fundus_screener/models/session_record.dart';
import 'package:fundus_screener/storage/session_log.dart';

void main() {
  late Directory tempDir;

  setUp(() {
    tempDir = Directory.systemTemp.createTempSync('session_log_test');
  });

  tearDown(() {
    tempDir.deleteSync(recursive: true);
  });

  test('commitRecord persists record and image; listRecords returns it', () async {
    final log = SessionLog(tempDir.path);
    final record = SessionRecord(
      sessionId: 'patient-01',
      timestamp: DateTime.utc(2026, 9, 27),
      imageFileName: 'capture_a.jpg',
      predictions: {'D': 0.1, 'G': 0.2, 'C': 0.3, 'A': 0.4, 'H': 0.5},
      mode: 'local',
      deviceTag: 's25ultra_diyretcam',
    );

    await log.commitRecord(record, Uint8List.fromList([1, 2, 3]));
    final records = await log.listRecords();

    expect(records.length, 1);
    expect(records.first.sessionId, 'patient-01');
    expect(File('${tempDir.path}/images/capture_a.jpg').existsSync(), isTrue);
  });

  test('a retake that is never committed leaves no record and no orphan file', () async {
    final log = SessionLog(tempDir.path);

    // First, commit one real record
    final record = SessionRecord(
      sessionId: 'patient-01',
      timestamp: DateTime.utc(2026, 9, 27),
      imageFileName: 'capture_a.jpg',
      predictions: {'D': 0.1, 'G': 0.2, 'C': 0.3, 'A': 0.4, 'H': 0.5},
      mode: 'local',
      deviceTag: 's25ultra_diyretcam',
    );
    await log.commitRecord(record, Uint8List.fromList([1, 2, 3]));

    // Simulate a retake: a preview image is written directly to images/,
    // but discarded without calling commitRecord (either by crash or future
    // caller writing preview files before user confirms capture).
    final imagesDir = Directory('${tempDir.path}/images');
    final orphanFile = File('${imagesDir.path}/orphan_preview.jpg');
    await orphanFile.writeAsBytes(Uint8List.fromList([9, 9, 9]));

    // Verify the orphan file exists before pruning
    expect(orphanFile.existsSync(), isTrue);

    // Prune orphan images
    await log.pruneOrphanImages();

    // Verify: orphan file is deleted, but committed record remains
    expect(orphanFile.existsSync(), isFalse);
    final records = await log.listRecords();
    expect(records.length, 1);
    expect(records.first.sessionId, 'patient-01');
    expect(File('${imagesDir.path}/capture_a.jpg').existsSync(), isTrue);
  });

  test('exportZip packages every committed image plus a manifest.csv row per record', () async {
    final log = SessionLog(tempDir.path);
    await log.commitRecord(
      SessionRecord(
        sessionId: 'patient-01',
        timestamp: DateTime.utc(2026, 9, 27),
        imageFileName: 'capture_a.jpg',
        predictions: {'D': 0.1, 'G': 0.2, 'C': 0.3, 'A': 0.4, 'H': 0.5},
        mode: 'local',
        deviceTag: 's25ultra_diyretcam',
      ),
      Uint8List.fromList([1, 2, 3]),
    );

    final zipPath = '${tempDir.path}/export.zip';
    await log.exportZip(zipPath);

    final bytes = File(zipPath).readAsBytesSync();
    final archive = ZipDecoder().decodeBytes(bytes);
    final names = archive.files.map((f) => f.name).toList();

    expect(names, contains('manifest.csv'));
    expect(names, contains('images/capture_a.jpg'));

    final manifestFile = archive.findFile('manifest.csv')!;
    final manifestText = String.fromCharCodes(manifestFile.content as List<int>);
    expect(manifestText, contains('patient-01'));
    expect(manifestText, contains('capture_a.jpg'));
  });
}
