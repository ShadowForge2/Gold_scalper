import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class AuthProvider extends ChangeNotifier {
  String? _token;
  String? _email;
  bool _loading = false;
  String? _error;
  String? _activeUrl;

  static const _baseUrls = [
    'https://gold-scalper.onrender.com',
    'https://gold-scalper-qyhg.onrender.com',
  ];
  static const _tokenKey = 'auth_token';
  static const _emailKey = 'auth_email';

  String? get token => _token;
  String? get email => _email;
  bool get loading => _loading;
  bool get isLoggedIn => _token != null;
  String? get error => _error;
  String get baseUrl => _activeUrl ?? _baseUrls.first;

  Future<String> _resolveUrl() async {
    if (_activeUrl != null) return _activeUrl!;
    for (final url in _baseUrls) {
      try {
        await http.get(Uri.parse('$url/health')).timeout(const Duration(seconds: 3));
        _activeUrl = url;
        return url;
      } catch (e) {
        debugPrint('auth resolveUrl failed: $url -> $e');
      }
    }
    _activeUrl = _baseUrls.first;
    return _activeUrl!;
  }

  Future<void> init() async {
    await _resolveUrl();
    final prefs = await SharedPreferences.getInstance();
    _token = prefs.getString(_tokenKey);
    _email = prefs.getString(_emailKey);
    if (_token != null) {
      final ok = await _validateToken();
      if (!ok) {
        _token = null;
        _email = null;
        await prefs.remove(_tokenKey);
        await prefs.remove(_emailKey);
      }
    }
    notifyListeners();
  }

  Future<bool> _validateToken() async {
    final url = baseUrl;
    try {
      final r = await http.get(
        Uri.parse('$url/auth/me'),
        headers: {'Authorization': 'Bearer $_token'},
      );
      if (r.statusCode == 200) {
        final data = jsonDecode(r.body);
        _email = data['email'];
        return true;
      }
    } catch (e) {
      debugPrint('_validateToken failed: $e');
      _activeUrl = null;
      await _resolveUrl();
    }
    return false;
  }

  Future<bool> register(String email, String password) async {
    _loading = true;
    _error = null;
    notifyListeners();
    final url = baseUrl;
    try {
      final r = await http.post(
        Uri.parse('$url/auth/register'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'email': email, 'password': password}),
      );
      final data = jsonDecode(r.body);
      if (r.statusCode == 200 && data['success'] == true) {
        await _saveToken(data['token'], data['email']);
        _loading = false;
        notifyListeners();
        return true;
      }
      _error = data['error'] ?? data['detail'] ?? 'Registration failed';
      _loading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _activeUrl = null;
      await _resolveUrl();
      _error = 'Network error: ${e.toString()}';
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<bool> login(String email, String password) async {
    _loading = true;
    _error = null;
    notifyListeners();
    final url = baseUrl;
    try {
      final r = await http.post(
        Uri.parse('$url/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'email': email, 'password': password}),
      );
      final data = jsonDecode(r.body);
      if (r.statusCode == 200 && data['success'] == true) {
        await _saveToken(data['token'], data['email']);
        _loading = false;
        notifyListeners();
        return true;
      }
      _error = data['error'] ?? data['detail'] ?? 'Login failed';
      _loading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _activeUrl = null;
      await _resolveUrl();
      _error = 'Network error: ${e.toString()}';
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<void> _saveToken(String token, String email) async {
    _token = token;
    _email = email;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, token);
    await prefs.setString(_emailKey, email);
  }

  Future<void> logout() async {
    _token = null;
    _email = null;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_emailKey);
    notifyListeners();
  }

  Map<String, String> get authHeaders => {
        'Authorization': 'Bearer $_token',
        'Content-Type': 'application/json',
      };
}
