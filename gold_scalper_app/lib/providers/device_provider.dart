import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/security_service.dart';

class DeviceProvider extends ChangeNotifier {
  String? _deviceId;
  bool _loading = true;
  bool _firstLaunch = true;
  DateTime? _credentialsSavedAt;
  DateTime? _lastWelcomeAt;
  bool _botStartedOnce = false;
  bool _accountTied = false;
  bool _tutorialSeen = false;

  static const _launchCountKey = 'launch_count';
  static const _tutorialSeenKey = 'tutorial_seen';
  static const _lastWelcomeKey = 'last_welcome_at';

  String? get deviceId => _deviceId;
  bool get loading => _loading;
  bool get firstLaunch => _firstLaunch;

  bool get shouldShowWelcome {
    if (_firstLaunch) return true;
    if (_lastWelcomeAt == null) return true;
    return DateTime.now().difference(_lastWelcomeAt!) >= const Duration(hours: 24);
  }
  DateTime? get credentialsSavedAt => _credentialsSavedAt;
  bool get botStartedOnce => _botStartedOnce;
  bool get accountTied => _accountTied;
  bool get tutorialSeen => _tutorialSeen;

  Duration? get cooldownRemaining {
    if (_accountTied) return null;
    if (_credentialsSavedAt == null) return Duration.zero;
    final elapsed = DateTime.now().difference(_credentialsSavedAt!);
    const cooldown = Duration(hours: 24);
    if (elapsed >= cooldown) return Duration.zero;
    return cooldown - elapsed;
  }

  bool get canEditCredentials =>
      !_accountTied &&
      (_credentialsSavedAt == null ||
          DateTime.now().difference(_credentialsSavedAt!) >= const Duration(hours: 24));

  Future<void> init() async {
    final sec = SecurityService.instance;
    final prefs = await SharedPreferences.getInstance();

    _deviceId = await sec.getDeviceId();
    if (_deviceId == null) {
      _deviceId = _generateFingerprint();
      await sec.setDeviceId(_deviceId!);
    }

    _firstLaunch = (prefs.getInt(_launchCountKey) ?? 0) == 0;
    await prefs.setInt(_launchCountKey, (prefs.getInt(_launchCountKey) ?? 0) + 1);

    final savedTs = await sec.getCredsSavedAt();
    if (savedTs != null) {
      _credentialsSavedAt = DateTime.tryParse(savedTs);
    }

    final lastWelcome = prefs.getString(_lastWelcomeKey);
    if (lastWelcome != null) {
      _lastWelcomeAt = DateTime.tryParse(lastWelcome);
    }

    final botStarted = await sec.getBotStarted();
    _botStartedOnce = botStarted == 'true';

    final accountTied = await sec.getAccountTied();
    _accountTied = accountTied == 'true';

    _tutorialSeen = prefs.getBool(_tutorialSeenKey) ?? false;

    _loading = false;
    notifyListeners();
  }

  Future<void> saveCredentialsTimestamp() async {
    _credentialsSavedAt = DateTime.now();
    await SecurityService.instance.setCredsSavedAt(_credentialsSavedAt!.toIso8601String());
    notifyListeners();
  }

  Future<void> markBotStarted() async {
    _botStartedOnce = true;
    final sec = SecurityService.instance;
    await sec.setBotStarted('true');
    if (_credentialsSavedAt != null && !_accountTied) {
      _accountTied = true;
      await sec.setAccountTied('true');
    }
    notifyListeners();
  }

  Future<void> markTutorialSeen() async {
    _tutorialSeen = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_tutorialSeenKey, true);
    notifyListeners();
  }

  Future<void> markWelcomeShown() async {
    _lastWelcomeAt = DateTime.now();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_lastWelcomeKey, _lastWelcomeAt!.toIso8601String());
  }

  Map<String, String> get headers => {
        'X-Device-Id': _deviceId ?? '',
        'Content-Type': 'application/json',
      };

  String _generateFingerprint() {
    final buf = StringBuffer();
    try {
      buf.write(Platform.operatingSystem);
      buf.write(Platform.operatingSystemVersion);
      buf.write(Platform.localHostname);
      buf.write(Platform.numberOfProcessors);
      buf.write(DateTime.now().timeZoneOffset.inMinutes);
    } catch (_) {
      debugPrint('Fingerprint error: $_');
    }

    final raw = buf.toString();
    if (raw.isEmpty) return _fallbackUuid();
    return 'fp_${_fnv1a(raw)}';
  }

  String _fnv1a(String input) {
    final bytes = utf8.encode(input);
    int hash = 0x811C9DC5;
    const prime = 0x01000193;
    for (final byte in bytes) {
      hash ^= byte;
      hash = (hash * prime) & 0xFFFFFFFF;
    }
    return hash.toRadixString(16).padLeft(8, '0');
  }

  String _fallbackUuid() {
    final now = DateTime.now().millisecondsSinceEpoch;
    final r = (now * 123456 + now % 98765) % 0xFFFFFFFF;
    return '${now.toString()}-${r.toString()}-${(now % 65536).toRadixString(16)}';
  }
}
