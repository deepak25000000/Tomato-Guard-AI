import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

class ScanRecord {
  final int? id;
  final String ts;
  final String plantId;
  final String angle;
  final String condition; // lab / farm / unknown
  final String label;
  final double score;
  final bool uncertain;
  final String modelVersion;
  final String? confirmedLabel;
  final String? imagePath;
  ScanRecord({this.id, required this.ts, required this.plantId, required this.angle,
    this.condition = 'unknown',
    required this.label, required this.score, required this.uncertain,
    required this.modelVersion, this.confirmedLabel, this.imagePath});
  Map<String, Object?> toMap() => {
    'id': id, 'ts': ts, 'plant_id': plantId, 'angle': angle, 'condition': condition, 'label': label,
    'score': score, 'uncertain': uncertain ? 1 : 0, 'model_version': modelVersion,
    'confirmed_label': confirmedLabel, 'image_path': imagePath,
  };
  static ScanRecord fromMap(Map<String, Object?> m) => ScanRecord(
    id: m['id'] as int?, ts: m['ts'] as String, plantId: m['plant_id'] as String? ?? '',
    angle: m['angle'] as String? ?? '', condition: m['condition'] as String? ?? 'unknown',
    label: m['label'] as String,
    score: (m['score'] as num).toDouble(), uncertain: (m['uncertain'] as int) == 1,
    modelVersion: m['model_version'] as String? ?? 'unknown',
    confirmedLabel: m['confirmed_label'] as String?, imagePath: m['image_path'] as String?,
  );
}

class AppDb {
  AppDb._();
  static final AppDb instance = AppDb._();
  Database? _db;
  Future<Database> get db async {
    _db ??= await openDatabase(
      p.join(await getDatabasesPath(), 'tomatoguard.db'),
      version: 2,
      onCreate: (d, _) => d.execute('''CREATE TABLE scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, plant_id TEXT, angle TEXT, condition TEXT,
        label TEXT, score REAL, uncertain INTEGER, model_version TEXT,
        confirmed_label TEXT, image_path TEXT)'''),
      onUpgrade: (d, oldV, newV) async {
        if (oldV < 2) {
          await d.execute("ALTER TABLE scans ADD COLUMN condition TEXT DEFAULT 'unknown'");
        }
      },
    );
    return _db!;
  }
  Future<int> insert(ScanRecord r) async => (await db).insert('scans', r.toMap());
  Future<List<ScanRecord>> all() async {
    final rows = await (await db).query('scans', orderBy: 'id DESC');
    return rows.map(ScanRecord.fromMap).toList();
  }
  Future<void> setConfirmed(int id, String? label) async =>
    (await db).update('scans', {'confirmed_label': label}, where: 'id=?', whereArgs: [id]);
  Future<void> delete(int id) async =>
    (await db).delete('scans', where: 'id=?', whereArgs: [id]);
  Future<void> deleteAll() async => (await db).delete('scans');
  Future<String> exportCsv() async {
    final rows = await all();
    final b = StringBuffer('id,ts,plant_id,angle,condition,label,model_score,uncertain,model_version,confirmed_label\n');
    String esc(String? v) { final s = v ?? ''; return '"${s.replaceAll('"', '""')}"'; }
    for (final r in rows) {
      b.writeln('${r.id},${esc(r.ts)},${esc(r.plantId)},${esc(r.angle)},${esc(r.condition)},${esc(r.label)},${r.score},${r.uncertain},${esc(r.modelVersion)},${esc(r.confirmedLabel)}');
    }
    return b.toString();
  }
}
