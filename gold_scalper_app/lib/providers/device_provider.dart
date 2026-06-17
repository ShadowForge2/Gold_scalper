import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class DeviceProvider extends ChangeNotifier {
  String? _deviceId;
  bool _loading = true;
  bool _firstLaunch = true;

  static const _deviceIdKey = 'device_id';
  static const _launchCountKey = 'launch_count';

  String? get deviceId => _deviceId;
  bool get loading => _loading;
  bool get firstLaunch => _firstLaunch;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _deviceId = prefs.getString(_deviceIdKey);
    if (_deviceId == null) {
      _deviceId = _generateUuid();
      await prefs.setString(_deviceIdKey, _deviceId!);
    }
    _firstLaunch = (prefs.getInt(_launchCountKey) ?? 0) == 0;
    await prefs.setInt(_launchCountKey, (prefs.getInt(_launchCountKey) ?? 0) + 1);
    _loading = false;
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
