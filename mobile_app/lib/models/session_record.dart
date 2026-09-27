class SessionRecord {
  final String sessionId;
  final DateTime timestamp;
  final String imageFileName;
  final Map<String, double> predictions;
  final String mode;
  final String deviceTag;

  SessionRecord({
    required this.sessionId,
    required this.timestamp,
    required this.imageFileName,
    required this.predictions,
    required this.mode,
    required this.deviceTag,
  });

  Map<String, dynamic> toJson() => {
        'sessionId': sessionId,
        'timestamp': timestamp.toIso8601String(),
        'imageFileName': imageFileName,
        'predictions': predictions,
        'mode': mode,
        'deviceTag': deviceTag,
      };

  factory SessionRecord.fromJson(Map<String, dynamic> json) => SessionRecord(
        sessionId: json['sessionId'] as String,
        timestamp: DateTime.parse(json['timestamp'] as String),
        imageFileName: json['imageFileName'] as String,
        predictions: Map<String, double>.from(json['predictions'] as Map),
        mode: json['mode'] as String,
        deviceTag: json['deviceTag'] as String,
      );
}
