import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/bot_state.dart';
import '../models/trade.dart';
import '../models/performance.dart';
import '../models/config.dart';
import '../widgets/terminal_log.dart';

class ApiService {
  final String baseUrl;
  final Map<String, String> Function() authHeaders;
  final http.Client _client = http.Client();

  ApiService({
    required this.baseUrl,
    required this.authHeaders,
  });

  static const _timeout = Duration(seconds: 10);

  Future<Map<String, dynamic>> _get(String path) async {
    final r = await _client.get(
      Uri.parse('$baseUrl$path'),
      headers: authHeaders(),
    ).timeout(_timeout);
    if (r.statusCode == 200) return jsonDecode(r.body);
    throw Exception('GET $path: ${r.statusCode} ${r.body}');
  }

  Future<Map<String, dynamic>> _post(
      String path, Map<String, dynamic> body) async {
    final headers = authHeaders();
    headers['Content-Type'] = 'application/json';
    final r = await _client.post(
      Uri.parse('$baseUrl$path'),
      headers: headers,
      body: jsonEncode(body),
    ).timeout(_timeout);
    if (r.statusCode != 200) {
      try {
        final data = jsonDecode(r.body);
        throw Exception('POST $path: ${r.statusCode} ${data['error'] ?? r.body}');
      } catch (_) {
        throw Exception('POST $path: ${r.statusCode} ${r.body}');
      }
    }
    return jsonDecode(r.body);
  }

  Future<BotState> getState() async {
    final data = await _get('/api/user/bot/state');
    final bot = data['bot'] ?? {};
    final account = data['account'] ?? {};
    return BotState(
      status: data['running'] == true ? 'running' : 'stopped',
      state: bot['state'] ?? 'IDLE',
      connected: account['error'] == null,
      broker: 'Capital.com',
      symbol: bot['symbol'] ?? 'XAUUSD',
      balance: (account['balance'] ?? 0).toDouble(),
      dailyPnl: (bot['positions']?['daily_pnl'] ?? 0).toDouble(),
      bid: (account['bid'] ?? 0).toDouble(),
      ask: (account['ask'] ?? 0).toDouble(),
      bias: bot['bias']?['bias'] ?? 'NEUTRAL',
      biasStrength: ((bot['bias']?['strength'] ?? 0) * 100).toDouble(),
      openPositions: bot['positions']?['open_count'] ?? 0,
      timestamp: DateTime.now(),
    );
  }

  Future<List<Trade>> getRecentTrades({int limit = 50}) async {
    return [];
  }

  Future<List<Trade>> getAllTrades() async {
    return [];
  }

  Future<Performance> getPerformance() async {
    return Performance(
      totalTrades: 0,
      wins: 0,
      losses: 0,
      winRate: 0,
      grossProfit: 0,
      grossLoss: 0,
      netPnl: 0,
      profitFactor: 0,
      avgWin: 0,
      avgLoss: 0,
      maxDrawdown: 0,
      startingBalance: 0,
      endingBalance: 0,
      returnPct: 0,
      monthly: [],
      daily: [],
    );
  }

  Future<List<EquityPoint>> getEquityCurve() async {
    return [];
  }

  Future<BotConfig> getConfig() async {
    return BotConfig();
  }

  Future<bool> updateConfig(BotConfig config) async {
    return true;
  }

  Future<List<LogEntry>> getLogs() async {
    try {
      final data = await _get('/api/user/bot/logs');
      final list = data['logs'] as List? ?? [];
      return list.map((l) => LogEntry.fromJson(l)).toList();
    } catch (_) {
      return [];
    }
  }

  Future<bool> startBot() async {
    try {
      await _post('/api/user/bot/start', {});
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> stopBot() async {
    try {
      await _post('/api/user/bot/stop', {});
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<Map<String, dynamic>> closeAllPositions() async {
    try {
      final data = await _post('/api/trades/close_all', {});
      return data;
    } catch (e) {
      return {'message': 'Failed to close positions', 'closed_count': 0};
    }
  }

  Future<List<Map<String, dynamic>>> getAccounts() async {
    try {
      final data = await _get('/api/user/accounts');
      return List<Map<String, dynamic>>.from(data['accounts'] ?? []);
    } catch (_) {
      return [];
    }
  }

  Future<bool> addAccount(
      String apiKey, String identifier, String password, bool demo) async {
    try {
      await _post('/api/user/accounts', {
        'api_key': apiKey,
        'identifier': identifier,
        'password': password,
        'demo': demo,
      });
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> removeAccount(String identifier) async {
    try {
      final headers = authHeaders();
      final r = await _client.delete(
        Uri.parse('$baseUrl/api/user/accounts/$identifier'),
        headers: headers,
      );
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  void dispose() {
    _client.close();
  }
}
