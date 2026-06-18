import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class AuthProvider extends ChangeNotifier {
  String? _token;
  String? _email;
  bool _loading = false;
  String? _error;

  static const _baseUrl = 'https://gold-scalper.onrender.com';
  static const _tokenKey = 'auth_token';
  static const _emailKey = 'auth_email';

  String? get token => _token;
  String? get email => _email;
  bool get loading => _loading;
  bool get isLoggedIn => _token != null;
  String? get error => _error;
  String get baseUrl => _baseUrl;

  Future<void> init() async {
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
    try {
      final r = await http.get(
        Uri.parse('$_baseUrl/auth/me'),
        headers: {'Authorization': 'Bearer $_token'},
      );
      if (r.statusCode == 200) {
        final data = jsonDecode(r.body);
        _email = data['email'];
        return true;
      }
    } catch (_) {}
    return false;
  }

  Future<bool> register(String email, String password) async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      final r = await http.post(
        Uri.parse('$_baseUrl/auth/register'),
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
    try {
      final r = await http.post(
        Uri.parse('$_baseUrl/auth/login'),
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
