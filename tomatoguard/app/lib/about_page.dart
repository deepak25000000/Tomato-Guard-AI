import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'classifier.dart';
import 'strings.dart';

class AboutPage extends StatelessWidget {
  final UiStrings ui; final Classifier classifier;
  const AboutPage({super.key, required this.ui, required this.classifier});
  @override
  Widget build(BuildContext context) => FutureBuilder<String>(
    future: rootBundle.loadString('assets/model_card.json').catchError((_)=>'{}'),
    builder: (c,s) {
      final card = s.data==null ? <String,dynamic>{} : (jsonDecode(s.data!) as Map<String,dynamic>);
      final lim = (card['limitations'] as List?)?.cast<String>() ?? const [];
      return ListView(padding: const EdgeInsets.all(16), children:[
        Container(padding: const EdgeInsets.all(10), decoration: BoxDecoration(color: Colors.amber.shade100, borderRadius: BorderRadius.circular(8)),
          child: Text(ui.researchBanner, style: const TextStyle(fontWeight: FontWeight.w700))),
        const SizedBox(height:12),
        Text(ui.disclaimer, style: const TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height:4),
        Text(ui.scope),
        const SizedBox(height:4),
        Text(ui.tobrfv),
        const SizedBox(height:8),
        Container(padding: const EdgeInsets.all(10), decoration: BoxDecoration(color: Colors.orange.shade50, borderRadius: BorderRadius.circular(8)),
          child: const Text('Lab and farm/outdoor tomato leaf photos are supported as input. Farm accuracy is unknown because training data is lab-style; for farm photos, use good lighting, clear single-leaf framing, and confirm important results with an agronomist or laboratory test.')),
        const Divider(height:24),
        Text('Model', style: Theme.of(context).textTheme.titleLarge),
        Text('Version: ${classifier.meta.modelVersion}'),
        Text('Input: ${classifier.meta.inputSize}x${classifier.meta.inputSize} RGB, direct resize (no crop), ImageNet norm'),
        Text('Classes: ${classifier.meta.labels.join(", ")}'),
        Text('Threshold (uncertain): ${classifier.meta.uncertainThreshold}'),
        const SizedBox(height:8),
        const Text('Model score = raw softmax (uncalibrated). Temperature scaling made calibration worse in validation and is not applied.'),
        const Divider(height:24),
        Text('Limitations', style: Theme.of(context).textTheme.titleLarge),
        for (final l in lim) Padding(padding: const EdgeInsets.only(top:4), child: Text('• $l')),
        const Divider(height:24),
        Text('Dataset licences / attribution', style: Theme.of(context).textTheme.titleLarge),
        const Text('• Track A leaf images: public tomato leaf dataset as used in results/ (see results/RESULTS.md provenance).'),
        const Text('• Track B (research viewer): TOBRFV-LMID, CC BY 4.0 — cite the dataset authors/paper as recorded in results/phase3.'),
        const SizedBox(height:16),
        FilledButton.icon(onPressed: ()=> _bench(context), icon: const Icon(Icons.speed), label: const Text('Run benchmark (20 warm-up + 100 timed)')),
        const SizedBox(height:24),
      ]);
    },
  );
  Future<void> _bench(BuildContext context) async {
    final r = classifier.benchmark(warmup:20, runs:100);
    if (!context.mounted) return;
    showDialog(context: context, builder:(_)=>AlertDialog(
      title: const Text('On-device benchmark'),
      content: Text('Model ${classifier.meta.modelVersion}, ${classifier.meta.inputSize}px\n20 warm-up + 100 timed\n\n'
        'Median: ${r['median_ms']!.toStringAsFixed(2)} ms\np95: ${r['p95_ms']!.toStringAsFixed(2)} ms\n\n'
        'Record this in benchmarks/device_results_template.csv with real device info. Debug builds are slower; use release build.'),
      actions:[TextButton(onPressed:()=>Navigator.pop(context), child: const Text('OK'))],
    ));
  }
}
