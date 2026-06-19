import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class DeviceProvider extends ChangeNotifier {
  String? _deviceId;
  bool _loading = true;
  bool _firstLaunch = true;
  DateTime? _credentialsSavedAt;
  bool _botStartedOnce = false;
  bool _accountTied = false;
  bool _tutorialSeen = false;

  static const _deviceIdKey = 'device_id';
  static const _launchCountKey = 'launch_count';
  static const _credsSavedAtKey = 'credentials_saved_at';
  static const _botStartedKey = 'bot_started_once';
  static const _accountTiedKey = 'account_tied';
  static const _tutorialSeenKey = 'tutorial_seen';

  String? get deviceId => _deviceId;
  bool get loading => _loading;
  bool get firstLaunch => _firstLaunch;
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
    final prefs = await SharedPreferences.getInstance();
    _deviceId = prefs.getString(_deviceIdKey);
    if (_deviceId == null) {
      _deviceId = _generateUuid();
      await prefs.setString(_deviceIdKey, _deviceId!);
    }
    _firstLaunch = (prefs.getInt(_launchCountKey) ?? 0) == 0;
    await prefs.setInt(_launchCountKey, (prefs.getInt(_launchCountKey) ?? 0) + 1);

    final savedTs = prefs.getString(_credsSavedAtKey);
    if (savedTs != null) {
      _credentialsSavedAt = DateTime.tryParse(savedTs);
    }
    _botStartedOnce = prefs.getBool(_botStartedKey) ?? false;
    _accountTied = prefs.getBool(_accountTiedKey) ?? false;
    _tutorialSeen = prefs.getBool(_tutorialSeenKey) ?? false;

    _loading = false;
    notifyListeners();
  }

  Future<void> saveCredentialsTimestamp() async {
    _credentialsSavedAt = DateTime.now();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_credsSavedAtKey, _credentialsSavedAt!.toIso8601String());
    notifyListeners();
  }

  Future<void> markBotStarted() async {
    _botStartedOnce = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_botStartedKey, true);
    if (_credentialsSavedAt != null && !_accountTied) {
      _accountTied = true;
      await prefs.setBool(_accountTiedKey, true);
    }
    notifyListeners();
  }

  Future<void> markTutorialSeen() async {
    _tutorialSeen = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_tutorialSeenKey, true);
    notifyListeners();
  }

  Map<String, String> get headers => {
        'X-Device-Id': _deviceId ?? '',
        'Content-Type': 'application/json',
      };

  String _generateUuid() {
    final now = DateTime.now().millisecondsSinceEpoch;
    final r = (now * 123456 + now % 98765) % 0xFFFFFFFF;
    return '${now.toString()}-${r.toString()}-${_randomHex(12)}';
  }

  String _randomHex(int len) {
    final buf = <int>[];
    for (int i = 0; i < len; i++) {
      buf.add((DateTime.now().microsecondsSinceEpoch % 16).toInt());
    }
    return buf.map((n) => n.toRadixString(16)).join();
  }
}
