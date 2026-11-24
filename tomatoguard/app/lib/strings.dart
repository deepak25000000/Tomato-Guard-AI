import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

class UiStrings {
  final String disclaimer;
  final String scope;
  final String tobrfv;
  final String uncertain;
  final String researchBanner;
  final String modelScoreLabel;
  UiStrings({
    required this.disclaimer, required this.scope, required this.tobrfv,
    required this.uncertain, required this.researchBanner, required this.modelScoreLabel,
  });
  static UiStrings? _cached;
  static Future<UiStrings> load() async {
    if (_cached != null) return _cached!;
    final j = jsonDecode(await rootBundle.loadString('assets/ui_strings.json')) as Map<String, dynamic>;
    _cached = UiStrings(
      disclaimer: j['DISCLAIMER'] as String,
      scope: j['SCOPE'] as String,
      tobrfv: j['TOBRFV'] as String,
      uncertain: j['UNCERTAIN'] as String,
      researchBanner: j['RESEARCH_BANNER'] as String,
      modelScoreLabel: j['MODEL_SCORE_LABEL'] as String? ?? 'model score',
    );
    return _cached!;
  }
}
