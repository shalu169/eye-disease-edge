import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive_io.dart';
import 'package:csv/csv.dart';
import '../models/labels.dart';
import '../models/session_record.dart';

class SessionLog {
  final String rootDirectoryPath;

  SessionLog(this.rootDirectoryPath);

  File get _manifestFile => File('$rootDirectoryPath/manifest.jsonl');
  Directory get _imagesDir => Directory('$rootDirectoryPath/images');

  Future<void> commitRecord(SessionRecord record, Uint8List imageBytes) async {
    if (!_imagesDir.existsSync()) {
      _imagesDir.createSync(recursive: true);
    }
    final imageFile = File('${_imagesDir.path}/${record.imageFileName}');
    await imageFile.writeAsBytes(imageBytes);

    final line = jsonEncode(record.toJson());
    await _manifestFile.writeAsString('$line\n', mode: FileMode.append);
  }

  Future<List<SessionRecord>> listRecords() async {
    if (!_manifestFile.existsSync()) return [];
    final lines = await _manifestFile.readAsLines();
    return lines
        .where((line) => line.trim().isNotEmpty)
        .map((line) => SessionRecord.fromJson(jsonDecode(line)))
        .toList();
  }

  Future<void> pruneOrphanImages() async {
    if (!_imagesDir.existsSync()) return;

    final records = await listRecords();
    final referencedFileNames = records.map((r) => r.imageFileName).toSet();

    final files = _imagesDir.listSync();
    for (final entity in files) {
      if (entity is File) {
        final fileName = entity.path.split('/').last;
        if (!referencedFileNames.contains(fileName)) {
          await entity.delete();
        }
      }
    }
  }

  Future<File> exportZip(String outputPath) async {
    await pruneOrphanImages();

    final records = await listRecords();
    final encoder = ZipFileEncoder();
    encoder.create(outputPath);

    final rows = <List<dynamic>>[
      ['sessionId', 'timestamp', 'imageFileName', ...kDiseaseLabels, 'mode', 'deviceTag'],
    ];
    for (final record in records) {
      rows.add([
        record.sessionId,
        record.timestamp.toIso8601String(),
        record.imageFileName,
        ...kDiseaseLabels.map((label) => record.predictions[label]),
        record.mode,
        record.deviceTag,
      ]);
      final imagePath = '${_imagesDir.path}/${record.imageFileName}';
      if (File(imagePath).existsSync()) {
        encoder.addFile(File(imagePath), 'images/${record.imageFileName}');
      }
    }

    final manifestCsv = ListToCsvConverter().convert(rows);
    encoder.addArchiveFile(ArchiveFile(
        'manifest.csv', manifestCsv.length, utf8.encode(manifestCsv)));
    encoder.close();

    return File(outputPath);
  }
}
