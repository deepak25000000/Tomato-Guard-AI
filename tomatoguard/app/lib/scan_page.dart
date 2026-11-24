import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'classifier.dart';
import 'db.dart';
import 'risk.dart';
import 'strings.dart';

class ScanPage extends StatefulWidget {
  final Classifier classifier;
  final UiStrings ui;
  const ScanPage({super.key, required this.classifier, required this.ui});
  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  final _picker = ImagePicker();
  final _plantId = TextEditingController();
  String _angle = '0';
  String _condition = 'lab'; // lab | farm
  bool _isTomatoLeaf = false;
  File? _file;
  Prediction? _pred;
  String? _hint;
  bool _busy = false;

  @override
  void dispose() {
    _plantId.dispose();
    super.dispose();
  }

  String _qualityHint(img.Image im) {
    // Heuristic only - labelled "not validated" in UI.
    double sum = 0, sum2 = 0;
    int n = 0;
    for (int y = 0; y < im.height; y += 10) {
      for (int x = 0; x < im.width; x += 10) {
        final p = im.getPixel(x, y);
        final l = (0.299 * p.r + 0.587 * p.g + 0.114 * p.b) / 255.0;
        sum += l;
        sum2 += l * l;
        n++;
      }
    }
    final mean = sum / n;
    final variance = sum2 / n - mean * mean;
    if (mean < 0.15) return 'Heuristic (not validated): photo looks dark. Try better light.';
    if (mean > 0.92) return 'Heuristic (not validated): photo looks overexposed.';
    if (variance < 0.005) return 'Heuristic (not validated): photo may be blurry or low contrast. Re-photograph.';
    return 'Heuristic (not validated): brightness/blur check passed.';
  }

  Future<void> _pick(ImageSource src) async {
    if (_plantId.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Plant ID is required before scanning.')));
      return;
    }
    if (!_isTomatoLeaf) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Please confirm the photo is a tomato leaf before scanning.')));
      return;
    }
    final x = await _picker.pickImage(source: src, maxWidth: 2048);
    if (x == null) return;
    setState(() => _busy = true);
    try {
      final bytes = await File(x.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Could not decode image');
      final hint = _qualityHint(decoded);
      final pred = widget.classifier.classify(decoded);
      final rec = ScanRecord(
        ts: DateTime.now().toIso8601String(),
        plantId: _plantId.text.trim(),
        angle: _angle,
        condition: _condition,
        label: pred.label,
        score: pred.confidence,
        uncertain: pred.uncertain,
        modelVersion: widget.classifier.meta.modelVersion,
        imagePath: x.path,
      );
      await AppDb.instance.insert(rec);
      setState(() {
        _file = File(x.path);
        _pred = pred;
        _hint = hint;
      });
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Scan failed: $e')));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final ui = widget.ui;
    final pred = _pred;
    final scheme = Theme.of(context).colorScheme;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _banner(ui.researchBanner, Colors.amber.shade100, Colors.amber.shade900),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Photo details', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),
              TextField(
                controller: _plantId,
                decoration: const InputDecoration(labelText: 'Plant ID (required)', border: OutlineInputBorder(), prefixIcon: Icon(Icons.local_florist_outlined)),
              ),
              const SizedBox(height: 12),
              DropdownButtonFormField<String>(
                initialValue: _angle,
                decoration: const InputDecoration(labelText: 'Leaf angle', border: OutlineInputBorder(), prefixIcon: Icon(Icons.rotate_90_degrees_ccw)),
                items: const ['0', '5', '15', 'other'].map((a) => DropdownMenuItem(value: a, child: Text('$a degrees'))).toList(),
                onChanged: (v) => setState(() => _angle = v ?? '0'),
              ),
              const SizedBox(height: 12),
              DropdownButtonFormField<String>(
                initialValue: _condition,
                decoration: const InputDecoration(labelText: 'Photo condition', border: OutlineInputBorder(), prefixIcon: Icon(Icons.wb_sunny_outlined)),
                items: const [
                  DropdownMenuItem(value: 'lab', child: Text('Lab / controlled')),
                  DropdownMenuItem(value: 'farm', child: Text('Farm / outdoor')),
                ],
                onChanged: (v) => setState(() => _condition = v ?? 'lab'),
              ),
              if (_condition == 'farm') ...[
                const SizedBox(height: 10),
                _banner(
                  'Farm/outdoor photo: the model was trained on lab-style leaf photos. Accuracy on farm photos is unknown. Use good light, avoid heavy shadows/blur, and confirm important results with an agronomist or laboratory test.',
                  Colors.orange.shade50,
                  Colors.orange.shade900,
                ),
              ],
              const SizedBox(height: 12),
              _banner(
                'Framing guide: single tomato leaf, fill the frame, in focus, good light. Avoid hands, soil, multiple leaves, or heavy background clutter.',
                scheme.surfaceContainerHighest,
                scheme.onSurface,
              ),
              CheckboxListTile(
                contentPadding: EdgeInsets.zero,
                value: _isTomatoLeaf,
                onChanged: (v) => setState(() => _isTomatoLeaf = v ?? false),
                controlAffinity: ListTileControlAffinity.leading,
                title: const Text('I confirm this photo is a tomato leaf (not another plant/object).'),
                subtitle: const Text('Out-of-scope images can produce misleading model scores.'),
              ),
            ]),
          ),
        ),
        const SizedBox(height: 12),
        Row(children: [
          Expanded(child: FilledButton.icon(onPressed: _busy ? null : () => _pick(ImageSource.camera), icon: const Icon(Icons.camera_alt), label: const Text('Camera'))),
          const SizedBox(width: 10),
          Expanded(child: OutlinedButton.icon(onPressed: _busy ? null : () => _pick(ImageSource.gallery), icon: const Icon(Icons.photo_library_outlined), label: const Text('Gallery'))),
        ]),
        if (_busy) const Padding(padding: EdgeInsets.all(20), child: Center(child: CircularProgressIndicator())),
        if (_hint != null) Padding(padding: const EdgeInsets.only(top: 10), child: Text(_hint!, style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant))),
        if (_file != null)
          Padding(
            padding: const EdgeInsets.only(top: 12),
            child: ClipRRect(borderRadius: BorderRadius.circular(12), child: Image.file(_file!, height: 230, fit: BoxFit.cover)),
          ),
        if (pred != null) _resultCard(pred, ui),
      ],
    );
  }

  Widget _banner(String text, Color bg, Color fg) => Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(10)),
        child: Text(text, style: TextStyle(color: fg, fontWeight: FontWeight.w500)),
      );

  Widget _resultCard(Prediction pred, UiStrings ui) {
    final r = riskFor(pred);
    final scheme = Theme.of(context).colorScheme;
    return Card(
      margin: const EdgeInsets.only(top: 14),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(Icons.analytics_outlined, color: r.color),
            const SizedBox(width: 8),
            Text(r.title, style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: r.color)),
          ]),
          const SizedBox(height: 8),
          Text(pred.uncertain ? ui.uncertain : 'Top class: ${pred.label}', style: const TextStyle(fontSize: 15)),
          const SizedBox(height: 6),
          Text('Photo condition: ${_condition == 'farm' ? 'Farm / outdoor' : 'Lab / controlled'}',
              style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
          if (_condition == 'farm')
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                'Farm photo note: model training was lab-style; farm accuracy is unknown. Treat this as a screening signal only.',
                style: TextStyle(fontSize: 12, color: Colors.orange.shade900),
              ),
            ),
          const SizedBox(height: 12),
          Text('All class model scores (raw softmax, uncalibrated)', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          for (final e in pred.allScores) _scoreRow(e.key, e.value, e.key == pred.label),
          const SizedBox(height: 10),
          _banner(r.action(ui), r.color.withValues(alpha: 0.08), scheme.onSurface),
          const Divider(height: 24),
          Text('Model: ${widget.classifier.meta.modelVersion} • ${pred.latencyMs} ms • threshold ${widget.classifier.meta.uncertainThreshold.toStringAsFixed(3)}',
              style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant)),
          const SizedBox(height: 8),
          Text(ui.disclaimer, style: const TextStyle(fontSize: 12)),
          const SizedBox(height: 2),
          Text(ui.scope, style: const TextStyle(fontSize: 12)),
          const SizedBox(height: 2),
          Text(ui.tobrfv, style: const TextStyle(fontSize: 12)),
        ]),
      ),
    );
  }

  Widget _scoreRow(String label, double score, bool isTop) {
    final pct = (score * 100).toStringAsFixed(1);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Expanded(child: Text(label, style: TextStyle(fontWeight: isTop ? FontWeight.w700 : FontWeight.w500))),
          Text('$pct%', style: TextStyle(fontWeight: isTop ? FontWeight.w700 : FontWeight.w500)),
        ]),
        const SizedBox(height: 2),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(value: score.clamp(0.0, 1.0), minHeight: 7, backgroundColor: Colors.black12),
        ),
      ]),
    );
  }
}
