import 'package:flutter/material.dart';

class DiseaseInfo {
  final String name;
  final String severity; // Low/Moderate/High signal context
  final Color color;
  final String summary;
  final String symptoms;
  final String generalCare;
  const DiseaseInfo(this.name, this.severity, this.color, this.summary, this.symptoms, this.generalCare);
}

const diseaseLibrary = <DiseaseInfo>[
  DiseaseInfo('Bacterial Spot', 'Condition class', Color(0xFFD32F2F),
    'A bacterial leaf condition in the model’s tomato-leaf dataset.',
    'Small dark spots, possible yellowing; appearance can vary by stage and photo quality.',
    'General: isolate affected plants when possible, avoid overhead watering, use clean tools, and confirm with a local agronomist/laboratory before any treatment decision.'),
  DiseaseInfo('Early Blight', 'Condition class', Color(0xFFEF6C00),
    'A fungal leaf condition class. Can be confused with Leaf Miner Flies in this model.',
    'Brown lesions, concentric rings may appear; older leaves often affected.',
    'General: remove heavily affected leaves, improve airflow, avoid wet foliage, and seek local agronomic guidance for confirmed management.'),
  DiseaseInfo('Healthy', 'Low signal', Color(0xFF2E7D32),
    'No strong disease signal detected by the model in the photo.',
    'No visible disease class signal from this model output.',
    'General: continue routine monitoring, good watering practices, and periodic re-imaging under clear light.'),
  DiseaseInfo('Late Blight', 'Condition class', Color(0xFFC62828),
    'A serious leaf condition class in the dataset; visual confirmation is important.',
    'Dark/water-soaked lesions may spread quickly under favorable conditions.',
    'General: monitor urgently, separate suspicious plants, avoid spreading contaminated material, and confirm with an agronomist/laboratory promptly.'),
  DiseaseInfo('Leaf Miner Flies', 'Condition class', Color(0xFFF9A825),
    'Leaf miner damage pattern class. Can be confused with Early Blight in this model.',
    'Winding trails/mines on leaves; severity varies.',
    'General: monitor affected leaves, use integrated pest management practices, and consult local agronomic guidance for confirmed identification.'),
  DiseaseInfo('Magnesium Deficiency', 'Condition class', Color(0xFF7CB342),
    'A nutrient-related leaf condition class in the dataset.',
    'Interveinal yellowing patterns may appear; symptoms can overlap with other stresses.',
    'General: confirm with soil/plant nutrient assessment before applying supplements; consult local agronomic guidance.'),
  DiseaseInfo('Yellow Leaf Curl Virus', 'Condition class', Color(0xFFD84315),
    'A viral condition class in the dataset; laboratory/expert confirmation is important.',
    'Yellowing, curling, stunted growth patterns may appear; symptoms vary.',
    'General: manage vectors and infected plant material per local guidance; confirm with agronomist/laboratory before action.'),
];

class DiseaseLibraryPage extends StatefulWidget {
  const DiseaseLibraryPage({super.key});
  @override
  State<DiseaseLibraryPage> createState() => _DiseaseLibraryPageState();
}

class _DiseaseLibraryPageState extends State<DiseaseLibraryPage> {
  String q = '';
  @override
  Widget build(BuildContext context) {
    final items = diseaseLibrary.where((d) => d.name.toLowerCase().contains(q.toLowerCase())).toList();
    return Column(children: [
      Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
        child: TextField(
          decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search conditions', border: OutlineInputBorder()),
          onChanged: (v) => setState(() => q = v),
        ),
      ),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Text('Informational library for the 7 model classes. Not a diagnosis; confirm with agronomist/laboratory.',
            style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant)),
      ),
      const SizedBox(height: 8),
      Expanded(
        child: ListView.builder(
          padding: const EdgeInsets.all(16),
          itemCount: items.length,
          itemBuilder: (c, i) {
            final d = items[i];
            return Card(
              margin: const EdgeInsets.only(bottom: 12),
              child: ListTile(
                leading: CircleAvatar(backgroundColor: d.color.withValues(alpha: 0.15), child: Icon(Icons.eco, color: d.color)),
                title: Text(d.name, style: const TextStyle(fontWeight: FontWeight.w700)),
                subtitle: Text(d.summary),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => DiseaseDetailPage(info: d))),
              ),
            );
          },
        ),
      ),
    ]);
  }
}

class DiseaseDetailPage extends StatelessWidget {
  final DiseaseInfo info;
  const DiseaseDetailPage({super.key, required this.info});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(info.name)),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        Card(color: info.color.withValues(alpha: 0.08), child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children:[
          Text(info.name, style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          Chip(label: Text(info.severity), backgroundColor: info.color.withValues(alpha: 0.15)),
          const SizedBox(height: 8),
          Text(info.summary),
        ]))),
        const SizedBox(height: 12),
        _section(context, 'What to look for', info.symptoms),
        _section(context, 'General care guidance', info.generalCare),
        const SizedBox(height: 12),
        Text('Research screening aid only. Not a diagnosis. Confirm with an agronomist or laboratory test.',
            style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant)),
      ]),
    );
  }
  Widget _section(BuildContext context, String title, String body) => Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children:[
    Text(title, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
    const SizedBox(height: 8),
    Text(body),
  ])));
}
