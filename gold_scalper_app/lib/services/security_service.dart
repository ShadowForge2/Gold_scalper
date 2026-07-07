import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class SecurityService {
  SecurityService._();
  static final instance = SecurityService._();
  final _secure = const FlutterSecureStorage();

  static const _deviceIdKey = 'device_id';
  static const _credsTsKey = 'credentials_saved_at';
  static const _botStartedKey = 'bot_started_once';
  static const _accountTiedKey = 'account_tied';

  Future<String?> getDeviceId() => _secure.read(key: _deviceIdKey);
  Future<void> setDeviceId(String v) => _secure.write(key: _deviceIdKey, value: v);

  Future<String?> getCredsSavedAt() => _secure.read(key: _credsTsKey);
  Future<void> setCredsSavedAt(String v) => _secure.write(key: _credsTsKey, value: v);

  Future<String?> getBotStarted() => _secure.read(key: _botStartedKey);
  Future<void> setBotStarted(String v) => _secure.write(key: _botStartedKey, value: v);

  Future<String?> getAccountTied() => _secure.read(key: _accountTiedKey);
  Future<void> setAccountTied(String v) => _secure.write(key: _accountTiedKey, value: v);

  Future<void> clearAll() => _secure.deleteAll();

  static bool isJailbroken() {
    if (kIsWeb) return false;
    if (Platform.isAndroid) {
      return _checkAndroidRoot();
    } else if (Platform.isIOS) {
      return _checkIOSJailbreak();
    }
    return false;
  }

  static bool _checkAndroidRoot() {
    const paths = [
      '/system/app/Superuser.apk',
      '/sbin/su',
      '/system/bin/su',
      '/system/xbin/su',
      '/data/local/xbin/su',
      '/data/local/bin/su',
      '/system/sd/xbin/su',
      '/system/bin/failsafe/su',
      '/data/local/su',
      '/su/bin/su',
    ];
    try {
      for (final path in paths) {
        if (File(path).existsSync()) return true;
      }
      final result = Process.runSync('which', ['su']);
      if (result.exitCode == 0 && result.stdout.toString().trim().isNotEmpty) return true;
    } catch (_) {}
    return false;
  }

  static bool _checkIOSJailbreak() {
    const paths = [
      '/Applications/Cydia.app',
      '/Applications/blackra1n.app',
      '/Applications/FakeCarrier.app',
      '/Applications/Icy.app',
      '/Applications/IntelliScreen.app',
      '/Applications/MxTube.app',
      '/Applications/RockApp.app',
      '/Applications/SBSettings.app',
      '/Applications/WinterBoard.app',
      '/Library/MobileSubstrate/MobileSubstrate.dylib',
      '/bin/bash',
      '/usr/sbin/sshd',
      '/etc/apt',
    ];
    try {
      for (final path in paths) {
        if (File(path).existsSync()) return true;
      }
    } catch (_) {}
    return false;
  }
}
