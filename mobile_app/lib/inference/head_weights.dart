class HeadWeights {
  final List<List<double>> convHeadWeight; // [1024][576]
  final List<double> convHeadBias;         // [1024]
  final List<List<double>> classifierWeight; // [5][1024]
  final List<double> classifierBias;         // [5]
  final List<String> labels;

  HeadWeights({
    required this.convHeadWeight,
    required this.convHeadBias,
    required this.classifierWeight,
    required this.classifierBias,
    required this.labels,
  });

  factory HeadWeights.fromJson(Map<String, dynamic> json) {
    List<List<double>> to2D(dynamic raw) => (raw as List)
        .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
        .toList();
    List<double> to1D(dynamic raw) =>
        (raw as List).map((e) => (e as num).toDouble()).toList();

    return HeadWeights(
      convHeadWeight: to2D(json['conv_head_weight']),
      convHeadBias: to1D(json['conv_head_bias']),
      classifierWeight: to2D(json['classifier_weight']),
      classifierBias: to1D(json['classifier_bias']),
      labels: (json['labels'] as List).map((e) => e as String).toList(),
    );
  }
}
