import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'db.dart';
import 'strings.dart';

class HistoryPage extends StatefulWidget {
  final UiStrings ui;
  const HistoryPage({super.key, required this.ui});
  @override State<HistoryPage> createState() => _HistoryPageState();
}
class _HistoryPageState extends State<HistoryPage> {
  late Future<List<ScanRecord>> _rows = AppDb.instance.all();
  void _reload()=>setState(()=>_rows = AppDb.instance.all());

  Future<void> _export() async {
    final csv = await AppDb.instance.exportCsv();
    final dir = await getTemporaryDirectory();
    final f = File('${dir.path}/tomatoguard_history.csv');
    await f.writeAsString(csv);
    await Share.shareXFiles([XFile(f.path)], text: 'TomatoGuard history (local export)');
  }

  @override
  Widget build(BuildContext context) => Column(children:[
    Padding(padding: const EdgeInsets.all(12), child: Row(children:[
      Expanded(child: OutlinedButton.icon(onPressed:_export, icon: const Icon(Icons.download), label: const Text('Export CSV'))),
      const SizedBox(width:8),
      Expanded(child: OutlinedButton.icon(onPressed:() async { await AppDb.instance.deleteAll(); _reload(); },
        icon: const Icon(Icons.delete_forever), label: const Text('Delete all data'))),
    ])),
    Expanded(child: FutureBuilder<List<ScanRecord>>(future:_rows, builder:(c,s){
      if(!s.hasData) return const Center(child:CircularProgressIndicator());
      final rows=s.data!;
      if(rows.isEmpty) return const Center(child:Text('No scans yet.'));
      return ListView.builder(itemCount: rows.length, itemBuilder:(c,i){ final r=rows[i]; return ListTile(
        title: Text('${r.label}  (${(r.score*100).toStringAsFixed(1)}% ${widget.ui.modelScoreLabel}${r.uncertain?" • uncertain":""})'),
        subtitle: Text('${r.ts.substring(0,16)} • plant ${r.plantId} • ${r.angle}° • ${r.condition} • ${r.modelVersion}\nConfirmed: ${r.confirmedLabel ?? "(none)"}'),
        isThreeLine: true,
        trailing: PopupMenuButton<String>(onSelected:(v) async {
          if(v=='delete'){ await AppDb.instance.delete(r.id!); _reload(); }
          if(v=='confirm'){ final v = await _confirmDialog(r.confirmedLabel); await AppDb.instance.setConfirmed(r.id!, v); _reload(); }
        }, itemBuilder:(c)=>const [PopupMenuItem(value:'confirm', child:Text('Set confirmed label')), PopupMenuItem(value:'delete', child:Text('Delete'))]),
      );});
    })),
  ]);

  Future<String?> _confirmDialog(String? current) async {
    final ctrl = TextEditingController(text: current ?? '');
    return showDialog<String?>(context: context, builder:(c)=>AlertDialog(
      title: const Text('Confirmed label (optional)'),
      content: TextField(controller: ctrl, decoration: const InputDecoration(hintText:'e.g. Healthy / Early Blight / unknown')),
      actions: [
        TextButton(onPressed:()=>Navigator.pop(c,null), child: const Text('Cancel')),
        TextButton(onPressed:()=>Navigator.pop(c, ctrl.text.trim().isEmpty?null:ctrl.text.trim()), child: const Text('Save')),
      ],
    ));
  }
}
