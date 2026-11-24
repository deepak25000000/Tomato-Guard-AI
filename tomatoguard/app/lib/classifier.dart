import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelMeta {
  final List<String> labels;
  final int inputSize;
  final List<double> mean;
  final List<double> std;
  final double temperature;
  final double uncertainThreshold;
  final String? healthyLabel;
  final String? notTomatoLabel;
  final String modelVersion;

  ModelMeta.fromJson(Map<String, dynamic> j)
      : labels = List<String>.from(j['labels']),
        inputSize = j['input_size'] as int,
        mean = List<double>.from((j['mean'] as List).map((e) => (e as num).toDouble())),
        std = List<double>.from((j['std'] as List).map((e) => (e as num).toDouble())),
        temperature = (j['temperature'] as num).toDouble(),
        uncertainThreshold = (j['uncertain_threshold'] as num).toDouble(),
        healthyLabel = j['healthy_label'] as String?,
        notTomatoLabel = j['not_tomato_label'] as String?,
        modelVersion = j['model_version'] as String? ?? 'unknown';
}

class Prediction {
  final String label;
  final double confidence;
  final List<MapEntry<String, double>> top3;
  final List<MapEntry<String, double>> allScores;
  final int latencyMs;
  final bool uncertain;
  final bool isHealthy;
  final bool notTomato;

  Prediction(this.label, this.confidence, this.top3, this.allScores, this.latencyMs, this.uncertain, this.isHealthy, this.notTomato);
}

/// Runs the exported TFLite model fully on-device (no network).
/// Pre-processing mirrors research/models.py eval_tf(): resize to SxS, scale to [0,1], ImageNet normalise.
class Classifier {
  final Interpreter _interp;
  final ModelMeta meta;
  final bool _nchw;
  final List<int> _inShape;

  Classifier._(this._interp, this.meta, this._nchw, this._inShape);

  static Future<Classifier> load() async {
    final meta = ModelMeta.fromJson(
        jsonDecode(await rootBundle.loadString('assets/models/model_meta.json')) as Map<String, dynamic>);
    final interp = await Interpreter.fromAsset(
      'assets/models/tomatoguard.tflite',
      options: InterpreterOptions()..threads = 2,
    );
    final shape = interp.getInputTensor(0).shape; // [1,3,H,W] or [1,H,W,3]
    return Classifier._(interp, meta, shape[1] == 3, shape);
  }

  Float32List _preprocess(img.Image src) {
    final s = meta.inputSize;
    final oriented = img.bakeOrientation(src);
    final r = img.copyResize(oriented, width: s, height: s, interpolation: img.Interpolation.linear);
    final buf = Float32List(3 * s * s);
    for (var y = 0; y < s; y++) {
      for (var x = 0; x < s; x++) {
        final p = r.getPixel(x, y);
        final c = [p.r / 255.0, p.g / 255.0, p.b / 255.0];
        for (var k = 0; k < 3; k++) {
          final v = ((c[k] - meta.mean[k]) / meta.std[k]).toDouble();
          final idx = _nchw ? k * s * s + y * s + x : (y * s + x) * 3 + k;
          buf[idx] = v;
        }
      }
    }
    return buf;
  }

  List<double> _logits(Float32List input) {
    final n = meta.labels.length;
    final output = List.filled(n, 0.0).reshape([1, n]);
    _interp.run(input.reshape(_inShape), output);
    return List<double>.from((output[0] as List).map((e) => (e as num).toDouble()));
  }

  List<double> _softmax(List<double> z, double t) {
    final scaled = z.map((v) => v / t).toList();
    final m = scaled.reduce(math.max);
    final e = scaled.map((v) => math.exp(v - m)).toList();
    final sum = e.reduce((a, b) => a + b);
    return e.map((v) => v / sum).toList();
  }

  Prediction classify(img.Image image) {
    final sw = Stopwatch()..start();
    final probs = _softmax(_logits(_preprocess(image)), meta.temperature);
    sw.stop();
    final ranked = List.generate(probs.length, (i) => MapEntry(meta.labels[i], probs[i]))
      ..sort((a, b) => b.value.compareTo(a.value));
    final top = ranked.first;
    return Prediction(
      top.key,
      top.value,
      ranked.take(3).toList(),
      ranked.toList(),
      sw.elapsedMilliseconds,
      top.value < meta.uncertainThreshold,
      top.key == meta.healthyLabel,
      top.key == meta.notTomatoLabel,
    );
  }

  Map<String, double> benchmark({int warmup = 20, int runs = 100}) {
    final n = meta.inputSize;
    final input = Float32List(3 * n * n)..fillRange(0, 3 * n * n, 0.1);
    for (var i = 0; i < warmup; i++) {
      _logits(input);
    }
    final t = <double>[];
    for (var i = 0; i < runs; i++) {
      final sw = Stopwatch()..start();
      _logits(input);
      sw.stop();
      t.add(sw.elapsedMicroseconds / 1000.0);
    }
    t.sort();
    return {'median_ms': t[t.length ~/ 2], 'p95_ms': t[math.min(t.length - 1, (t.length * 0.95).floor())]};
  }

  void close() => _interp.close();
}
