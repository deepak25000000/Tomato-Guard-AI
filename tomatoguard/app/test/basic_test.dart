import 'package:flutter_test/flutter_test.dart';
import 'package:tomatoguard/classifier.dart';
import 'package:tomatoguard/risk.dart';

void main() {
  test('ModelMeta parses', () {
    final m = ModelMeta.fromJson({
      'labels':['A','Healthy','C'], 'input_size':224, 'mean':[0.485,0.456,0.406],
      'std':[0.229,0.224,0.225], 'temperature':1.0, 'uncertain_threshold':0.6,
      'healthy_label':'Healthy', 'not_tomato_label': null, 'model_version':'v1'
    });
    expect(m.labels.length, 3);
    expect(m.uncertainThreshold, 0.6);
    expect(m.modelVersion, 'v1');
  });

  test('riskFor uncertain', () {
    final p = Prediction('A', 0.5, const [], const [], 10, true, false, false);
    expect(riskFor(p), Risk.moderate);
  });

  test('riskFor healthy', () {
    final p = Prediction('Healthy', 0.9, const [], const [], 10, false, true, false);
    expect(riskFor(p), Risk.low);
  });

  test('riskFor disease high signal', () {
    final p = Prediction('Early Blight', 0.9, const [], const [], 10, false, false, false);
    expect(riskFor(p), Risk.high);
  });
}
