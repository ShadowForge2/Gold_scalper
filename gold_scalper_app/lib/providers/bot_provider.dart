import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/bot_state.dart';
import '../models/trade.dart';
import '../models/performance.dart';
import '../models/config.dart';
import '../widgets/terminal_log.dart';
import 'device_provider.dart';

class BotProvider extends ChangeNotifier {
  final DeviceProvider _device;
  final http.Client _client = http.Client();
  Timer? _timer;
  final _rand = Random(42);

  bool _useMockData = false;

  BotState? _state;
  List<Trade> _recentTrades = [];
  Performance? _performance;
  BotConfig _config = BotConfig();
  List<EquityPoint> _equityCurve = [];
  List<LogEntry> _logs = [];
  bool _loading = true;
  bool _botRunning = false;
  Map<String, dynamic> _subscription = {};

  String? _activeUrl;
  int? _navigateToTab;
  bool _highlightCredentials = false;
  bool _subscriptionBlocked = false;
  bool _navigateToSubscription = false;

  static const _configKey = 'saved_bot_config';

  static const _baseUrls = [
    'https://gold-scalper-qyhg.onrender.com',
    'https://gold-scalper.onrender.com',
  ];

  BotProvider(this._device) {
    _resolveUrl();
  }

  String get baseUrl => _activeUrl ?? _baseUrls.first;
  int? get navigateToTab => _navigateToTab;
  bool get highlightCredentials => _highlightCredentials;

  BotState? get state => _state;
  List<Trade> get recentTrades => _recentTrades;
  Performance? get performance => _performance;
  BotConfig get config => _config;
  List<EquityPoint> get equityCurve => _equityCurve;
  List<LogEntry> get logs => _logs;
  bool get loading => _loading;
  bool get botRunning => _botRunning;
  Map<String, dynamic> get subscription => _subscription;
  bool get canTrade => _subscription['can_trade'] == true;
  bool get isDemo => _subscription['demo'] == true;
  bool get trialActive => _subscription['trial_active'] == true;
  int get daysRemaining => _subscription['days_remaining'] ?? 0;
  double get dueAmount => (_subscription['due_amount'] ?? 0).toDouble();
  double get unpaidFees => (_subscription['unpaid_fees'] ?? 0).toDouble();
  double get currentMonthProfit =>
      (_subscription['current_month_profit'] ?? 0).toDouble();
  double get currentMonthFee =>
      (_subscription['current_month_fee'] ?? 0).toDouble();
  List<Map<String, dynamic>> get monthlyPeriods =>
      List<Map<String, dynamic>>.from(_subscription['monthly_periods'] ?? []);
  bool get hasNoAccounts => _subscription['error'] != null || _subscription['is_new'] == true;
  bool get subscriptionBlocked => _subscriptionBlocked;
  bool get navigateToSubscription => _navigateToSubscription;

  Future<void> _resolveUrl() async {
    for (final url in _baseUrls) {
      try {
        await _client
            .get(Uri.parse('$url/health'))
            .timeout(const Duration(seconds: 3));
        _activeUrl = url;
        return;
      } catch (_) {}
    }
    _activeUrl = _baseUrls.first;
  }

  void toggleMockData() {
    _useMockData = !_useMockData;
    notifyListeners();
  }

  Map<String, String> get _getHeaders {
    final h = <String, String>{'X-Device-Id': _device.deviceId ?? ''};
    return h;
  }

  Future<Map<String, dynamic>> _get(String path, {bool retried = false}) async {
    if (_useMockData) return _mockResponse(path);
    final url = baseUrl;
    try {
      final r = await _client.get(
        Uri.parse('$url$path'),
        headers: _getHeaders,
      );
      if (r.statusCode == 200) return jsonDecode(r.body);
      throw Exception('GET $path: ${r.statusCode}');
    } catch (_) {
      if (!retried) {
        _activeUrl = null;
        await _resolveUrl();
        return _get(path, retried: true);
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>> _post(
      String path, Map<String, dynamic> body, {bool retried = false}) async {
    if (_useMockData) return _mockPostResponse(path, body);
    final url = baseUrl;
    try {
      final r = await _client.post(
        Uri.parse('$url$path'),
        headers: _device.headers,
        body: jsonEncode(body),
      );
      final data = jsonDecode(r.body);
      if (r.statusCode == 200) return data;
      throw Exception('POST $path: ${data['error'] ?? r.body}');
    } catch (_) {
      if (!retried) {
        _activeUrl = null;
        await _resolveUrl();
        return _post(path, body, retried: true);
      }
      rethrow;
    }
  }

  Map<String, dynamic> _mockResponse(String path) {
    if (path.contains('/bot/state')) {
      return {
        'running': _botRunning,
        'bot': {
          'state': _botRunning ? 'AWAITING_SIGNAL' : 'IDLE',
          'symbol': 'XAUUSD',
          'bias': {'bias': 'BULLISH', 'strength': 0.78},
          'signal': {'momentum': 0.62, 'candle_strength': 0.45},
          'positions': {
            'daily_pnl': 12.35,
            'event_pnl': 4.10,
            'open_count': 0,
          },
          'risk': {
            'consecutive_losses': 0,
            'session_trades': 3,
            'cooldown_active': false,
            'max_daily_loss': 10.0,
            'max_event_loss': 5.0,
            'max_trades_per_event': 5,
            'cooldown_seconds': 60,
          },
        },
        'account': {
          'balance': 1250.42,
          'equity': 1262.77,
          'profit': 12.35,
          'bid': 2345.67,
          'ask': 2345.89,
          'free_margin': 1120.33,
          'name': 'Demo Account',
          'leverage': 30,
        },
      };
    }
    if (path.contains('/bot/logs')) {
      return {
        'logs': _generateMockLogs(),
      };
    }
    if (path.contains('/bot/performance')) {
      return {
        'trades': 385,
        'wins': 266,
        'losses': 119,
        'win_rate': 69.1,
        'gross_profit': 5842.0,
        'gross_loss': 2190.0,
        'net_pnl': 3652.0,
        'profit_factor': 4.25,
        'avg_win': 21.96,
        'avg_loss': 18.40,
        'max_dd': 2186.0,
        'starting_balance': 500.0,
        'ending_balance': 4152.0,
        'return_pct': 730.4,
        'monthly': [],
        'daily': [],
      };
    }
    if (path.contains('/equity_curve')) {
      return {
        'points': [
          {'time': DateTime.now().subtract(const Duration(days: 60)).toIso8601String(), 'balance': 500.0},
          {'time': DateTime.now().subtract(const Duration(days: 45)).toIso8601String(), 'balance': 680.0},
          {'time': DateTime.now().subtract(const Duration(days: 30)).toIso8601String(), 'balance': 920.0},
          {'time': DateTime.now().subtract(const Duration(days: 15)).toIso8601String(), 'balance': 1350.0},
          {'time': DateTime.now().subtract(const Duration(days: 5)).toIso8601String(), 'balance': 2100.0},
          {'time': DateTime.now().toIso8601String(), 'balance': 4152.0},
        ],
      };
    }
    if (path.contains('/subscription')) {
      return {
        'trial_active': true,
        'trial_end': '2026-07-17T00:00:00',
        'days_remaining': 24,
        'subscribed': false,
        'subscription_end': null,
        'can_trade': true,
        'due_amount': 0.0,
        'unpaid_fees': 0.0,
        'demo': true,
        'current_month_profit': 250.42,
        'current_month_fee': 37.56,
        'monthly_periods': [
          {
            'period_start': '2026-06-01T00:00:00',
            'period_end': null,
            'starting_balance': 1000.0,
            'cumulative_profit': 250.42,
            'fee_15pct': 37.56,
            'fee_paid': false,
            'paid_at': null,
          }
        ],
      };
    }
    if (path.contains('/accounts')) {
      return {
        'accounts': [
          {
            'api_key': 'demo****key',
            'identifier': 'trader@email.com',
            'password': '****',
            'demo': true,
          }
        ],
      };
    }
    return {'success': true};
  }

  Map<String, dynamic> _mockPostResponse(
      String path, Map<String, dynamic> body) {
    if (path.contains('/bot/start')) {
      _botRunning = true;
      _state = _state?.copyWith(state: 'AWAITING_SIGNAL') ?? _mockState();
      return {'message': 'Bot started'};
    }
    if (path.contains('/bot/stop')) {
      _botRunning = false;
      _state = _state?.copyWith(state: 'IDLE') ?? _mockState();
      return {'message': 'Bot stopped'};
    }
    if (path.contains('/payment/initialize')) {
      return {
        'authorization_url': 'https://paystack.com/pay/mock-ref-12345',
        'reference': 'mock-ref-12345',
        'access_code': 'mock-access-code-12345',
      };
    }
    if (path.contains('/payment/verify')) {
      return {'message': 'Payment verified', 'status': 'success'};
    }
    return {'success': true};
  }

  List<dynamic> _generateMockLogs() {
    final levels = ['INFO', 'SIGNAL', 'TRADE', 'BIAS', 'WARNING'];
    final msgs = [
      'Bias updated: BULLISH (strength=0.78, H1=UPTREND) TRADEABLE',
      'Signal check: momentum=0.62 threshold=0.50',
      'No entry: score_below_entry_threshold score=0.48 threshold=0.50',
      r'Daily P&L: +$12.35 | Event P&L: +$4.10',
      'Trial active: 24 day(s) remaining.',
      'Connection healthy. Latency: 45ms',
      'Analyzing H1 structure...',
      'Candle pattern: ENGULFING_BULLISH',
      'Session: LONDON active (07:00-16:00 UTC)',
      'Risk check: daily_loss_ok=true event_loss_ok=true',
    ];
    final now = DateTime.now();
    return List.generate(30, (i) {
      final t = now.subtract(Duration(seconds: (30 - i) * 12));
      final lvl = levels[i % levels.length];
      return {
        'time': '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}:${t.second.toString().padLeft(2, '0')}',
        'message': msgs[i % msgs.length],
        'level': lvl,
      };
    });
  }

  BotState _mockState() {
    return BotState(
      status: _botRunning ? 'running' : 'stopped',
      state: _botRunning ? 'AWAITING_SIGNAL' : 'IDLE',
      connected: true,
      broker: 'Capital.com',
      symbol: 'XAUUSD',
      balance: 1250.42,
      dailyPnl: 12.35,
      bid: 2345.67,
      ask: 2345.89,
      bias: 'BULLISH',
      biasStrength: 78.0,
      openPositions: 0,
      timestamp: DateTime.now(),
    );
  }

  MockPerformance _mockPerformance() {
    return MockPerformance(
      totalTrades: 385,
      wins: 266,
      losses: 119,
      winRate: 69.1,
      grossProfit: 5842.0,
      grossLoss: 2190.0,
      netPnl: 3652.0,
      profitFactor: 4.25,
      avgWin: 21.96,
      avgLoss: 18.40,
      maxDrawdown: 2186.0,
      startingBalance: 500.0,
      endingBalance: 4152.0,
      returnPct: 730.4,
    );
  }

  List<EquityPoint> _mockEquityCurve() {
    final now = DateTime.now();
    var bal = 500.0;
    return List.generate(60, (i) {
      bal += _rand.nextDouble() * 30 - 8;
      if (bal < 400) bal = 400;
      return EquityPoint(
        time: now.subtract(Duration(days: 60 - i)),
        balance: bal,
      );
    });
  }

  Future<void> init() async {
    _loading = true;
    await _loadConfig();
    notifyListeners();

    if (_useMockData) {
      _state = _mockState();
      _logs = _generateMockLogs().map((l) => LogEntry.fromJson(l)).toList();
      _subscription = _mockResponse('/api/device/subscription');
      _performance = _mockPerformance();
      _equityCurve = _mockEquityCurve();
      _botRunning = true;
    } else {
      await _fetchAll();
    }

    _loading = false;
    notifyListeners();

    _timer?.cancel();
    final interval = _useMockData
        ? const Duration(seconds: 4)
        : const Duration(seconds: 5);
    _timer = Timer.periodic(interval, (_) {
      _useMockData ? _tickMock() : _tickLive();
    });
  }

  Future<void> _fetchAll() async {
    try {
      final stateData = await _get('/api/device/bot/state');
      _state = BotState.fromApiResponse(stateData);
      _botRunning = stateData['running'] == true;
    } catch (_) {}

    try {
      final logData = await _get('/api/device/bot/logs');
      final backendLogs = (logData['logs'] as List).map((l) => LogEntry.fromJson(l)).toList();
      if (backendLogs.isNotEmpty) {
        final existingMsgs = _logs.map((e) => e.message).toSet();
        for (final entry in backendLogs) {
          if (!existingMsgs.contains(entry.message)) {
            _logs.add(entry);
          }
        }
        if (_logs.length > 200) _logs.removeRange(0, _logs.length - 200);
      }
    } catch (_) {}

    try {
      _subscription = await _get('/api/device/subscription');
    } catch (_) {}

    try {
      final cfgData = await _get('/api/device/bot/config');
      _config = BotConfig.fromJson(cfgData);
    } catch (_) {}

    try {
      final perfData = await _get('/api/device/bot/performance');
      _performance = Performance.fromJson(perfData);
    } catch (_) {}

    try {
      final tradeData = await _get('/api/device/bot/trades');
      final tradesList = tradeData['trades'] as List? ?? [];
      _recentTrades = tradesList.map((t) => Trade.fromJson(t)).toList();
    } catch (_) {}

    await _fetchEquityCurve();
  }

  Future<void> _fetchEquityCurve() async {
    try {
      final data = await _get('/api/device/bot/equity_curve');
      final points = data['points'] as List? ?? [];
      _equityCurve = points.map((p) => EquityPoint(
        time: DateTime.tryParse(p['time'] ?? '') ?? DateTime.now(),
        balance: (p['balance'] ?? 0).toDouble(),
      )).toList();
    } catch (_) {}
  }

  Future<void> _tickLive() async {
    try {
      final stateData = await _get('/api/device/bot/state');
      _state = BotState.fromApiResponse(stateData);
      _botRunning = stateData['running'] == true;

      final logData = await _get('/api/device/bot/logs');
      final backendLogs = (logData['logs'] as List).map((l) => LogEntry.fromJson(l)).toList();
      if (backendLogs.isNotEmpty) {
        final existingMsgs = _logs.map((e) => e.message).toSet();
        for (final entry in backendLogs) {
          if (!existingMsgs.contains(entry.message)) {
            _logs.add(entry);
          }
        }
        if (_logs.length > 200) _logs.removeRange(0, _logs.length - 200);
      }
      _subscription = await _get('/api/device/subscription');
    } catch (_) {}

    if (_botRunning && !isDemo && !hasNoAccounts && !canTrade && !_subscriptionBlocked) {
      _subscriptionBlocked = true;
      addLog('Your free trial has ended. Please subscribe to continue trading.', level: 'WARNING');
      requestSubscription();
      try {
        await _post('/api/device/bot/stop', {});
        _botRunning = false;
        addLog('Bot stopped automatically due to expired trial/subscription.', level: 'WARNING');
      } catch (_) {}
    }
    if (_subscriptionBlocked && (isDemo || canTrade)) {
      _subscriptionBlocked = false;
    }

    try {
      final perfData = await _get('/api/device/bot/performance');
      _performance = Performance.fromJson(perfData);
    } catch (_) {}

    await _fetchEquityCurve();

    notifyListeners();
  }

  void _tickMock() {
    final states = ['AWAITING_SIGNAL', 'BIAS_ANALYSIS', 'AWAITING_SIGNAL', 'IN_TRADE', 'AWAITING_SIGNAL'];
    final biases = ['BULLISH', 'BULLISH', 'NEUTRAL', 'BULLISH', 'CONFLICT'];
    final strengths = [0.78, 0.82, 0.15, 0.71, 0.45];
    final idx = DateTime.now().second % states.length;

    final pnlDelta = (_rand.nextDouble() - 0.45) * 0.8;
    final newDailyPnl = (_state?.dailyPnl ?? 0) + pnlDelta;
    final newBalance = (_state?.balance ?? 1250) + pnlDelta;

    _state = _state?.copyWith(
      status: idx == 4 ? 'stopped' : 'running',
      state: states[idx],
      bias: biases[idx],
      biasStrength: strengths[idx] * 100,
      bid: 2345.67 + _rand.nextDouble() * 2 - 1,
      ask: 2345.89 + _rand.nextDouble() * 2 - 1,
      balance: newBalance,
      dailyPnl: newDailyPnl,
    );

    _botRunning = idx != 4;
    _subscription['current_month_profit'] = newBalance - 1000;
    _subscription['current_month_fee'] = (newBalance - 1000) * 0.15;
    if ((newBalance - 1000) * 0.15 > 0) {
      _subscription['current_month_fee'] = double.parse(
          ((newBalance - 1000) * 0.15).toStringAsFixed(2));
    }

    final logMsgs = [
      'Signal check: momentum=${(0.5 + _rand.nextDouble() * 0.3).toStringAsFixed(2)}',
      'Bias updated: ${biases[idx]} (strength=${strengths[idx]})',
      'Price: XAUUSD ${(2345 + _rand.nextDouble() * 3).toStringAsFixed(2)}',
      'P&L update: \$${newDailyPnl.toStringAsFixed(2)} today',
    ];
    final t = DateTime.now();
    _logs.add(LogEntry(
      time: '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}:${t.second.toString().padLeft(2, '0')}',
      message: logMsgs[idx % logMsgs.length],
      level: ['INFO', 'BIAS', 'INFO', 'TRADE'][idx % 4],
    ));
    if (_logs.length > 200) _logs.removeRange(0, _logs.length - 200);

    _equityCurve.add(EquityPoint(time: DateTime.now(), balance: newBalance));
    if (_equityCurve.length > 200) _equityCurve.removeAt(0);

    notifyListeners();
  }

  void addLog(String message, {String level = 'INFO'}) {
    _logs.add(LogEntry(
      time: DateTime.now().toIso8601String().substring(11, 19),
      message: message,
      level: level,
    ));
    if (_logs.length > 200) _logs.removeRange(0, _logs.length - 200);
    notifyListeners();
  }

  void requestCredentialsSetup() {
    _highlightCredentials = true;
    _navigateToTab = 4;
    notifyListeners();
  }

  void clearNavigation() {
    _navigateToTab = null;
    notifyListeners();
  }

  void clearHighlight() {
    _highlightCredentials = false;
    notifyListeners();
  }

  void requestSubscription() {
    _navigateToSubscription = true;
    notifyListeners();
  }

  void clearSubscriptionNavigation() {
    _navigateToSubscription = false;
    notifyListeners();
  }

  Future<bool> startBot() async {
    addLog('Starting bot...');
    if (_useMockData) {
      _botRunning = true;
      _state = _state?.copyWith(status: 'running', state: 'AWAITING_SIGNAL');
      addLog('Bot started successfully', level: 'TRADE');
      await _device.markBotStarted();
      notifyListeners();
      return true;
    }
    try {
      final accts = await getAccounts();
      if (accts.isEmpty) {
        addLog('No account configured. Please add credentials first.', level: 'WARNING');
        notifyListeners();
        return false;
      }
      final demo = accts.isNotEmpty && accts.first['demo'] == true;
      if (!demo && !canTrade) {
        _subscriptionBlocked = true;
        addLog('Please subscribe to continue using the service.', level: 'WARNING');
        requestSubscription();
        notifyListeners();
        return false;
      }
      final url = baseUrl;
      final r = await _client.post(
        Uri.parse('$url/api/device/bot/start'),
        headers: _device.headers,
        body: jsonEncode({}),
      );
      final data = jsonDecode(r.body);
      if (r.statusCode == 200) {
        _botRunning = true;
        _subscriptionBlocked = false;
        addLog('Bot started successfully', level: 'TRADE');
        await _device.markBotStarted();
        notifyListeners();
        return true;
      }
      if (r.statusCode == 402) {
        _subscriptionBlocked = true;
        addLog('Trial expired. Please subscribe to continue.', level: 'WARNING');
        requestSubscription();
        notifyListeners();
        return false;
      }
      addLog('Failed to start: ${data['error'] ?? r.body}', level: 'ERROR');
      notifyListeners();
      return false;
    } catch (e) {
      addLog('Failed to start bot: $e', level: 'ERROR');
      notifyListeners();
      return false;
    }
  }

  Future<bool> stopBot() async {
    addLog('Stopping bot...', level: 'WARNING');
    if (_useMockData) {
      _botRunning = false;
      _state = _state?.copyWith(status: 'stopped', state: 'IDLE');
      addLog('Bot stopped successfully', level: 'WARNING');
    } else {
      try {
        await _post('/api/device/bot/stop', {});
        _botRunning = false;
        addLog('Bot stopped successfully', level: 'WARNING');
      } catch (e) {
        addLog('Failed to stop bot: $e', level: 'ERROR');
        notifyListeners();
        return false;
      }
    }
    notifyListeners();
    return true;
  }

  Future<Map<String, dynamic>> closeAllPositions() async {
    addLog('Closing all positions...', level: 'WARNING');
    if (_useMockData) {
      await Future.delayed(const Duration(milliseconds: 500));
      _state = _state?.copyWith(state: 'IDLE', openPositions: 0);
      addLog('All positions closed successfully (0 remaining)', level: 'TRADE');
      notifyListeners();
      return {'message': 'All positions closed', 'closed_count': 0};
    }
    try {
      final result = await _post('/api/device/trades/close_all', {});
      final count = result['closed_count'] ?? 0;
      addLog('All positions closed: $count position(s)', level: 'TRADE');
      _state = _state?.copyWith(state: 'IDLE', openPositions: 0);
      notifyListeners();
      return result;
    } catch (e) {
      addLog('Failed to close positions: $e', level: 'ERROR');
      return {'message': 'Failed: $e', 'closed_count': 0};
    }
  }

  Future<List<Map<String, dynamic>>> getAccounts() async {
    final data = _useMockData
        ? _mockResponse('/api/device/accounts')
        : await _get('/api/device/accounts');
    return List<Map<String, dynamic>>.from(data['accounts'] ?? []);
  }

  Future<bool> addAccount(
      String apiKey, String identifier, String password, bool demo) async {
    addLog('Saving account: $identifier');
    if (!_useMockData) {
      try {
        await _post('/api/device/accounts', {
          'api_key': apiKey,
          'identifier': identifier,
          'password': password,
          'demo': demo,
        });
      } catch (e) {
        addLog('Failed to save account: $e', level: 'ERROR');
        return false;
      }
    }
    await _device.saveCredentialsTimestamp();
    addLog('Account saved: $identifier', level: 'TRADE');
    return true;
  }

  Future<bool> removeAccount(String identifier) async {
    addLog('Removed: $identifier', level: 'WARNING');
    return true;
  }

  Future<Map<String, dynamic>?> initializePayment(String email, {List<String>? channels}) async {
    final data = _useMockData
        ? _mockPostResponse('/api/payment/initialize', {})
        : await _post('/api/payment/initialize', {
            'email': email,
            if (channels != null) 'channels': channels,
          });
    if (data['access_code'] != null) {
      addLog('Payment link generated');
    }
    return data;
  }

  Future<Map<String, dynamic>?> initCryptomusPayment(double amount, String email) async {
    if (_useMockData) return {'payment_url': 'https://pay.cryptomus.com/mock'};
    try {
      return await _post('/api/payment/cryptomus/init', {
        'amount': amount,
        'email': email,
        'url_return': 'https://gold-scalper-qyhg.onrender.com',
      });
    } catch (e) {
      addLog('Failed to create Cryptomus payment: $e', level: 'ERROR');
      return null;
    }
  }

  Future<bool> verifyPayment(String reference) async {
    if (!_useMockData) {
      await _post('/api/payment/verify', {'reference': reference});
    }
    addLog('Payment verified', level: 'INFO');
    _subscription['unpaid_fees'] = 0.0;
    _subscription['subscribed'] = true;
    notifyListeners();
    return true;
  }

  Future<void> _loadConfig() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_configKey);
      if (raw != null) {
        final json = Map<String, dynamic>.from(jsonDecode(raw));
        _config = BotConfig.fromJson(json);
      }
    } catch (_) {}
  }

  void updateConfig(BotConfig config) {
    _config = config;
    notifyListeners();
  }

  Future<bool> saveConfig() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_configKey, jsonEncode(_config.toJson()));
      addLog('Settings saved locally', level: 'INFO');
      return true;
    } catch (e) {
      addLog('Failed to save settings: $e', level: 'ERROR');
      return false;
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _client.close();
    super.dispose();
  }
}

class MockPerformance extends Performance {
  MockPerformance({
    required super.totalTrades,
    required super.wins,
    required super.losses,
    required super.winRate,
    required super.grossProfit,
    required super.grossLoss,
    required super.netPnl,
    required super.profitFactor,
    required super.avgWin,
    required super.avgLoss,
    required super.maxDrawdown,
    required super.startingBalance,
    required super.endingBalance,
    required super.returnPct,
  }) : super(
          monthly: [],
          daily: [],
        );
}
