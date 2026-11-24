import 'package:flutter/material.dart';
import 'classifier.dart';
import 'strings.dart';

/// Simple UI categorization for the research prototype. NOT a clinical/field label.
enum Risk { low, moderate, high }

Risk riskFor(Prediction p) {
  if (p.uncertain) return Risk.moderate;
  return p.isHealthy ? Risk.low : Risk.high;
}

extension RiskInfo on Risk {
  String get title => switch (this) { Risk.low => 'Low signal', Risk.moderate => 'Uncertain', Risk.high => 'High signal' };
  Color get color => switch (this) { Risk.low => Colors.green, Risk.moderate => Colors.orange, Risk.high => Colors.red };
  String action(UiStrings s) => switch (this) {
        Risk.low => 'No strong disease signal in this photo. Continue routine monitoring.',
        Risk.moderate => s.uncertain,
        Risk.high => 'Model score is high for a condition class. Re-image and seek expert or laboratory confirmation.',
      };
}
