class PreprocessConfig {
  final int imageSize;
  final List<double> mean;
  final List<double> std;

  const PreprocessConfig({
    required this.imageSize,
    required this.mean,
    required this.std,
  });

  factory PreprocessConfig.fromJson(Map<String, dynamic> json) {
    return PreprocessConfig(
      imageSize: json['image_size'] as int,
      mean: (json['mean'] as List).map((e) => (e as num).toDouble()).toList(),
      std: (json['std'] as List).map((e) => (e as num).toDouble()).toList(),
    );
  }
}
