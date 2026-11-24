import 'package:flutter/material.dart';
import 'classifier.dart';
import 'strings.dart';
import 'home_page.dart';
import 'scan_page.dart';
import 'history_page.dart';
import 'disease_library.dart';
import 'settings_page.dart';

void main() => runApp(const TomatoGuardApp());

class TomatoGuardApp extends StatefulWidget {
  const TomatoGuardApp({super.key});
  @override
  State<TomatoGuardApp> createState() => _TomatoGuardAppState();
}

class _TomatoGuardAppState extends State<TomatoGuardApp> {
  ThemeMode _mode = ThemeMode.system;
  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF2E7D32);
    return MaterialApp(
      title: 'TomatoGuard AI',
      debugShowCheckedModeBanner: false,
      themeMode: _mode,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.light),
        scaffoldBackgroundColor: const Color(0xFFFAFAF6),
        cardTheme: const CardThemeData(elevation: 0, margin: EdgeInsets.zero),
        fontFamily: 'Roboto',
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.dark),
        cardTheme: const CardThemeData(elevation: 0, margin: EdgeInsets.zero),
        fontFamily: 'Roboto',
      ),
      home: Splash(mode: _mode, onModeChanged: (m) => setState(() => _mode = m)),
    );
  }
}

class Splash extends StatefulWidget {
  final ThemeMode mode;
  final ValueChanged<ThemeMode> onModeChanged;
  const Splash({super.key, required this.mode, required this.onModeChanged});
  @override
  State<Splash> createState() => _SplashState();
}

class _SplashState extends State<Splash> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))..forward();
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 1700), () {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => Boot(mode: widget.mode, onModeChanged: widget.onModeChanged)));
    });
  }
  @override
  void dispose() { _c.dispose(); super.dispose(); }
  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(gradient: LinearGradient(colors: [scheme.primary, scheme.tertiaryContainer], begin: Alignment.topLeft, end: Alignment.bottomRight)),
        child: Center(child: FadeTransition(opacity: CurvedAnimation(parent: _c, curve: Curves.easeOut), child: Column(mainAxisSize: MainAxisSize.min, children: [
          Container(padding: const EdgeInsets.all(22), decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.15), shape: BoxShape.circle), child: const Icon(Icons.eco, size: 72, color: Colors.white)),
          const SizedBox(height: 20),
          const Text('TomatoGuard AI', style: TextStyle(color: Colors.white, fontSize: 34, fontWeight: FontWeight.w800)),
          const SizedBox(height: 8),
          const Text('Protect Your Harvest', style: TextStyle(color: Colors.white70, fontSize: 16)),
        ]))),
      ),
    );
  }
}

class Boot extends StatefulWidget {
  final ThemeMode mode;
  final ValueChanged<ThemeMode> onModeChanged;
  const Boot({super.key, required this.mode, required this.onModeChanged});
  @override
  State<Boot> createState() => _BootState();
}

class _BootState extends State<Boot> {
  bool _consent = false;
  bool _bannerDismissed = false;
  @override
  Widget build(BuildContext context) {
    if (!_consent) return ConsentScreen(onAccept: () => setState(() => _consent = true));
    return HomeShell(mode: widget.mode, onModeChanged: widget.onModeChanged, bannerDismissed: _bannerDismissed, onDismissBanner: () => setState(() => _bannerDismissed = true));
  }
}

class ConsentScreen extends StatelessWidget {
  final VoidCallback onAccept;
  const ConsentScreen({super.key, required this.onAccept});
  @override
  Widget build(BuildContext context) => FutureBuilder<UiStrings>(
    future: UiStrings.load(),
    builder: (c, s) {
      final ui = s.data;
      return Scaffold(
        appBar: AppBar(title: const Text('TomatoGuard AI')),
        body: Padding(padding: const EdgeInsets.all(20), child: ListView(children: [
          if (ui != null) ...[
            _banner(ui.researchBanner),
            const SizedBox(height: 12),
            Text(ui.disclaimer, style: const TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Text(ui.scope),
            const SizedBox(height: 8),
            Text(ui.tobrfv),
          ] else const Center(child: CircularProgressIndicator()),
          const SizedBox(height: 20),
          const Text('Consent', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text('I understand this is a research screening prototype, not a diagnosis. I will confirm important results with an agronomist or laboratory test. Photos and results stay on this device unless I export them.'),
          const SizedBox(height: 20),
          FilledButton(onPressed: ui == null ? null : onAccept, child: const Text('I understand — continue')),
        ])),
      );
    },
  );
}

Widget _banner(String t) => Container(
  padding: const EdgeInsets.all(12),
  decoration: BoxDecoration(color: Colors.amber.shade100, borderRadius: BorderRadius.circular(12)),
  child: Text(t, style: const TextStyle(fontWeight: FontWeight.w700)),
);

class HomeShell extends StatefulWidget {
  final ThemeMode mode;
  final ValueChanged<ThemeMode> onModeChanged;
  final bool bannerDismissed;
  final VoidCallback onDismissBanner;
  const HomeShell({super.key, required this.mode, required this.onModeChanged, required this.bannerDismissed, required this.onDismissBanner});
  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;
  late final Future<Classifier> _clf = Classifier.load();
  late final Future<UiStrings> _ui = UiStrings.load();

  @override
  Widget build(BuildContext context) => FutureBuilder<List<Object>>(
    future: Future.wait([_clf, _ui]),
    builder: (context, snap) {
      if (snap.hasError) return Scaffold(appBar: AppBar(title: const Text('TomatoGuard AI')), body: Padding(padding: const EdgeInsets.all(24), child: Text('Model failed to load.\n\n${snap.error}\n\nEnsure assets/models/tomatoguard.tflite and model_meta.json are present.')));
      if (!snap.hasData) return const Scaffold(body: Center(child: CircularProgressIndicator()));
      final clf = snap.data![0] as Classifier;
      final ui = snap.data![1] as UiStrings;
      final pages = [
        HomePage(ui: ui, onScan: () => setState(() => _tab = 1)),
        ScanPage(classifier: clf, ui: ui),
        HistoryPage(ui: ui),
        const DiseaseLibraryPage(),
      ];
      return Scaffold(
        appBar: AppBar(
          title: const Text('TomatoGuard AI'),
          actions: [IconButton(icon: const Icon(Icons.settings_outlined), onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => SettingsPage(ui: ui, classifier: clf, mode: widget.mode, onModeChanged: widget.onModeChanged))))],
        ),
        body: Column(children: [
          if (!widget.bannerDismissed)
            MaterialBanner(
              backgroundColor: Colors.amber.shade100,
              content: Text(ui.researchBanner),
              leading: const Icon(Icons.science_outlined),
              actions: [TextButton(onPressed: widget.onDismissBanner, child: const Text('Dismiss'))],
            ),
          Expanded(child: IndexedStack(index: _tab, children: pages)),
        ]),
        floatingActionButton: _tab == 1 ? null : FloatingActionButton.extended(onPressed: () => setState(() => _tab = 1), icon: const Icon(Icons.center_focus_strong), label: const Text('Scan')),
        bottomNavigationBar: NavigationBar(
          selectedIndex: _tab,
          onDestinationSelected: (i) => setState(() => _tab = i),
          destinations: const [
            NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Home'),
            NavigationDestination(icon: Icon(Icons.camera_alt_outlined), selectedIcon: Icon(Icons.camera_alt), label: 'Scan'),
            NavigationDestination(icon: Icon(Icons.history_outlined), selectedIcon: Icon(Icons.history), label: 'History'),
            NavigationDestination(icon: Icon(Icons.menu_book_outlined), selectedIcon: Icon(Icons.menu_book), label: 'Library'),
          ],
        ),
      );
    },
  );
}
