import 'package:flutter/material.dart';
import 'classifier.dart';
import 'strings.dart';

class SettingsPage extends StatelessWidget {
  final UiStrings ui;
  final Classifier classifier;
  final ThemeMode mode;
  final ValueChanged<ThemeMode> onModeChanged;
  const SettingsPage({super.key, required this.ui, required this.classifier, required this.mode, required this.onModeChanged});

  @override
  Widget build(BuildContext context) {
    return ListView(padding: const EdgeInsets.all(16), children: [
      Text('Settings', style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
      const SizedBox(height: 12),
      Card(child: Column(children: [
        const ListTile(leading: Icon(Icons.dark_mode_outlined), title: Text('Theme')),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: SegmentedButton<ThemeMode>(
            segments: const [
              ButtonSegment(value: ThemeMode.system, label: Text('System')),
              ButtonSegment(value: ThemeMode.light, label: Text('Light')),
              ButtonSegment(value: ThemeMode.dark, label: Text('Dark')),
            ],
            selected: {mode},
            onSelectionChanged: (s) => onModeChanged(s.first),
          ),
        ),
      ])),
      const SizedBox(height: 12),
      Card(child: ListTile(
        leading: const Icon(Icons.memory_outlined),
        title: const Text('Model info'),
        subtitle: Text('${classifier.meta.modelVersion} • ${classifier.meta.inputSize}px • threshold ${classifier.meta.uncertainThreshold.toStringAsFixed(3)}\nRaw softmax model score, uncalibrated. On-device only.'),
      )),
      const SizedBox(height: 12),
      Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children:[
        Text('About & disclaimer', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: 8),
        Text(ui.disclaimer, style: const TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 6),
        Text(ui.scope),
        const SizedBox(height: 6),
        Text(ui.tobrfv),
        const SizedBox(height: 8),
        const Text('This app runs fully offline. No internet permission is requested for the app model inference flow.'),
      ]))),
    ]);
  }
}
