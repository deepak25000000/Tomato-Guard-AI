import 'package:flutter/material.dart';
import 'db.dart';
import 'strings.dart';

class HomePage extends StatefulWidget {
  final UiStrings ui;
  final VoidCallback onScan;
  const HomePage({super.key, required this.ui, required this.onScan});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late Future<List<ScanRecord>> _rows = AppDb.instance.all();
  @override
  void didChangeDependencies() { super.didChangeDependencies(); _rows = AppDb.instance.all(); }
  void refresh() => setState(() => _rows = AppDb.instance.all());

  static const _tips = [
    'Photograph a single tomato leaf in good natural light for a clearer model signal.',
    'For farm photos, avoid heavy shadows and fill the frame with one leaf.',
    'If the result is Uncertain, re-image from a different angle or better light.',
    'Confirm important condition signals with an agronomist or laboratory test.',
  ];

  @override
  Widget build(BuildContext context) {
    final tip = _tips[DateTime.now().day % _tips.length];
    final scheme = Theme.of(context).colorScheme;
    return RefreshIndicator(
      onRefresh: () async { refresh(); await _rows; },
      child: ListView(padding: const EdgeInsets.all(16), children: [
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(colors: [scheme.primary, scheme.tertiary], begin: Alignment.topLeft, end: Alignment.bottomRight),
            borderRadius: BorderRadius.circular(24),
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('TomatoGuard AI', style: Theme.of(context).textTheme.headlineMedium?.copyWith(color: scheme.onPrimary, fontWeight: FontWeight.w800)),
            const SizedBox(height: 4),
            Text('Protect Your Harvest', style: TextStyle(color: scheme.onPrimary.withValues(alpha: 0.92), fontSize: 16)),
            const SizedBox(height: 16),
            Text('Scan a tomato leaf for a research screening signal. Works offline on-device.',
                style: TextStyle(color: scheme.onPrimary.withValues(alpha: 0.95))),
            const SizedBox(height: 16),
            FilledButton.tonalIcon(
              onPressed: widget.onScan,
              icon: const Icon(Icons.center_focus_strong),
              label: const Text('Scan Leaf'),
              style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14)),
            ),
          ]),
        ),
        const SizedBox(height: 14),
        Card(child: ListTile(leading: const Icon(Icons.lightbulb_outline), title: const Text('Farming tip'), subtitle: Text(tip))),
        const SizedBox(height: 14),
        FutureBuilder<List<ScanRecord>>(future: _rows, builder: (c, s) {
          final rows = s.data ?? const <ScanRecord>[];
          final total = rows.length;
          final healthy = rows.where((r) => r.label == 'Healthy' && !r.uncertain).length;
          final diseased = rows.where((r) => r.label != 'Healthy' && !r.uncertain).length;
          final uncertain = rows.where((r) => r.uncertain).length;
          return Column(children: [
            Row(children: [
              _stat(context, 'Scans', '$total', Icons.photo_camera_outlined),
              _stat(context, 'Healthy', '$healthy', Icons.eco_outlined),
              _stat(context, 'Signals', '$diseased', Icons.warning_amber_rounded),
              _stat(context, 'Uncertain', '$uncertain', Icons.help_outline),
            ]),
            const SizedBox(height: 14),
            Align(alignment: Alignment.centerLeft, child: Text('Recent scans', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold))),
            const SizedBox(height: 8),
            if (rows.isEmpty)
              Card(child: Padding(padding: const EdgeInsets.all(24), child: Column(children:[
                Icon(Icons.yard_outlined, size: 56, color: scheme.primary),
                const SizedBox(height: 8),
                const Text('No scans yet'),
                const SizedBox(height: 4),
                const Text('Tap Scan Leaf to capture or select a tomato leaf photo.', textAlign: TextAlign.center),
              ])))
            else
              SizedBox(height: 140, child: ListView.separated(
                scrollDirection: Axis.horizontal,
                itemCount: rows.take(10).length,
                separatorBuilder: (_, __) => const SizedBox(width: 10),
                itemBuilder: (c, i) {
                  final r = rows[i];
                  return SizedBox(width: 220, child: Card(child: Padding(padding: const EdgeInsets.all(12), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children:[
                    Text(r.uncertain ? 'Uncertain' : r.label, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 4),
                    Text('${(r.score*100).toStringAsFixed(1)}% model score', style: TextStyle(color: scheme.onSurfaceVariant)),
                    const Spacer(),
                    Text('${r.ts.substring(0,16)} • ${r.condition}', style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
                  ]))));
                },
              )),
          ]);
        }),
      ]),
    );
  }

  Widget _stat(BuildContext context, String label, String value, IconData icon) => Expanded(child: Card(child: Padding(padding: const EdgeInsets.symmetric(vertical: 14), child: Column(children:[
    Icon(icon, size: 20, color: Theme.of(context).colorScheme.primary),
    const SizedBox(height: 4),
    Text(value, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
    Text(label, style: const TextStyle(fontSize: 12)),
  ]))));
}
