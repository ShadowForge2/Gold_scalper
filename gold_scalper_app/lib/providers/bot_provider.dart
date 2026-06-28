import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/bot_state.dart';
import '../models/trade.dart';
import '../models/performance.dart';
import '../models/config.dart';
import '../models/notification_item.dart';
import '../widgets/terminal_log.dart';
import '../services/notification_service.dart';
import 'device_provider.dart';

class BotProvider extends ChangeNotifier {
  final DeviceProvider _device;
  final http.Client _client = http.Client();
  Timer? _timer;

  bool _initialized = false;
  bool _backendReady = false;

  BotState? _state;
  List<Trade> _recentTrades = [];
  Performance? _performance;
  BotConfig _config = BotConfig();
  List<EquityPoint> _equityCurve = [];
  List<EquityPoint> _yearlyCurve = [];
  List<EquityPoint> _monthlyCurve = [];
  List<LogEntry> _logs = [];
  bool _loading = true;
  bool _botRunning = false;
  Map<String, dynamic> _subscription = {};

  String? _activeUrl;
  int? _navigateToTab;
  bool _highlightCredentials = false;
  bool _subscriptionBlocked = false;
  bool _navigateToSubscription = false;

  List<NotificationItem> _notifications = [];
  int _unreadCount = 0;
  final Set<String> _seenNotificationIds = {};

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
  List<EquityPoint> get yearlyCurve => _yearlyCurve;
  List<EquityPoint> get monthlyCurve => _monthlyCurve;
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
  List<NotificationItem> get notifications => _notifications;
  int get unreadCount => _unreadCount;

  Future<void> _resolveUrl() async {
    for (final url in _baseUrls) {
      try {
        final r = await _client
            .get(Uri.parse('$url/health'))
            .timeout(const Duration(seconds: 5));
        final body = jsonDecode(r.body);
        if (r.statusCode == 200 && body['status'] == 'healthy') {
          _activeUrl = url;
          _backendReady = true;
          debugPrint('_resolveUrl: using $url');
          return;
        }
      } catch (e) {
        debugPrint('_resolveUrl failed: $url -> $e');
      }
    }
    _activeUrl = _baseUrls.first;
    _backendReady = true;
    debugPrint('_resolveUrl: fallback to ${_baseUrls.first}');
  }

  Map<String, String> get _getHeaders {
    final h = <String, String>{'X-Device-Id': _device.deviceId ?? ''};
    return h;
  }

  static const _timeout = Duration(seconds: 10);

  Map<String, dynamic> _decodeMap(String body) {
    try {
      return Map<String, dynamic>.from(jsonDecode(body));
    } catch (_) {
      return {'error': body};
    }
  }

  String _networkError(Object error) {
    final msg = error.toString();
    if (msg.contains('Failed host lookup') ||
        msg.contains('No address associated with hostname') ||
        msg.contains('nodename nor servname') ||
        msg.contains('SocketException')) {
      return 'Network error: could not reach the server. Check internet connection or DNS.';
    }
    if (error is TimeoutException || msg.toLowerCase().contains('timed out')) {
      return 'Network error: server took too long to respond.';
    }
    if (msg.contains('ClientException')) {
      return 'Network error: connection to server failed.';
    }
    if (msg.contains('FormatException')) {
      return 'Server returned an invalid response.';
    }
    return msg
        .replaceFirst(RegExp(r'^Exception:\s*'), '')
        .replaceFirst(RegExp(r'^(GET|POST|DELETE|PUT)\s+[^:]+:\s*'), '');
  }

  Future<Map<String, dynamic>> _get(String path, {bool retried = false}) async {
    final url = baseUrl;
    try {
      final r = await _client.get(
        Uri.parse('$url$path'),
        headers: _getHeaders,
      ).timeout(_timeout);
      if (r.statusCode == 200) return jsonDecode(r.body);
      throw Exception('GET $path: ${r.statusCode}');
    } catch (e) {
      debugPrint('_get $path failed: $e');
      if (!retried) {
        await _resolveUrl();
        return _get(path, retried: true);
      }
      throw Exception(_networkError(e));
    }
  }

  Future<Map<String, dynamic>> _post(
      String path, Map<String, dynamic> body, {bool retried = false}) async {
    final url = baseUrl;
    try {
      final r = await _client.post(
        Uri.parse('$url$path'),
        headers: _device.headers,
        body: jsonEncode(body),
      ).timeout(_timeout);
      final data = _decodeMap(r.body);
      if (r.statusCode == 200) return data;
      throw Exception('POST $path: ${data['error'] ?? r.body}');
    } catch (e) {
      debugPrint('_post $path failed: $e');
      if (!retried) {
        await _resolveUrl();
        return _post(path, body, retried: true);
      }
      throw Exception(_networkError(e));
    }
  }

  Future<Map<String, dynamic>> _delete(String path, {bool retried = false}) async {
    final url = baseUrl;
    try {
      final r = await _client.delete(
        Uri.parse('$url$path'),
        headers: _device.headers,
      ).timeout(_timeout);
      final data = _decodeMap(r.body);
      if (r.statusCode == 200) return data;
      throw Exception('DELETE $path: ${data['error'] ?? r.body}');
    } catch (e) {
      debugPrint('_delete $path failed: $e');
      if (!retried) {
        await _resolveUrl();
        return _delete(path, retried: true);
      }
      throw Exception(_networkError(e));
    }
  }

  Future<void> init() async {
    if (_initialized) return;
    _initialized = true;
    _loading = true;
    await _loadConfig();
    notifyListeners();

    await _fetchAll();

    _loading = false;
    notifyListeners();

    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 5), (_) {
      _tickLive().catchError((e) {
        debugPrint('_tickLive unhandled: $e');
      });
    });
  }

  Future<void> refresh() async {
    await _fetchAll();
  }

  Future<void> _fetchAll() async {
    await Future.wait([
      _fetchState(),
      _fetchLogs(),
      _fetchSubscription(),
      _fetchConfig(),
      _fetchPerformance(),
      _fetchTrades(),
      _fetchEquityCurve(),
      _fetchNotifications(),
    ]);
  }

  Future<void> _fetchState() async {
    try {
      final stateData = await _get('/api/device/bot/state');
      _state = BotState.fromApiResponse(stateData);
      _botRunning = stateData['running'] == true;
    } catch (e) {
      debugPrint('_fetchState failed: $e');
      _state ??= _defaultState();
    }
  }

  BotState _defaultState() {
    return BotState(
      status: 'stopped',
      state: 'IDLE',
      connected: false,
      broker: 'Capital.com',
      symbol: 'XAUUSD',
      balance: 0,
      dailyPnl: 0,
      bid: 0,
      ask: 0,
      bias: 'NEUTRAL',
      biasStrength: 0,
      openPositions: 0,
      timestamp: DateTime.now(),
    );
  }

  Future<void> _fetchLogs() async {
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
    } catch (e) {
      debugPrint('_fetchLogs failed: $e');
    }
  }

  Future<void> _fetchSubscription() async {
    try {
      _subscription = await _get('/api/device/subscription');
    } catch (e) {
      debugPrint('_fetchSubscription failed: $e');
    }
  }

  Future<void> _fetchConfig() async {
    try {
      final cfgData = await _get('/api/device/bot/config');
      _config = BotConfig.fromJson(cfgData);
    } catch (e) {
      debugPrint('_fetchConfig failed: $e');
    }
  }

  Future<void> _fetchPerformance() async {
    try {
      final perfData = await _get('/api/device/bot/performance');
      _performance = Performance.fromJson(perfData);
    } catch (e) {
      debugPrint('_fetchPerformance failed: $e');
    }
  }

  Future<void> _fetchTrades() async {
    try {
      final tradeData = await _get('/api/device/bot/trades');
      final tradesList = tradeData['trades'] as List? ?? [];
      _recentTrades = tradesList.map((t) => Trade.fromJson(t)).toList();
    } catch (e) {
      debugPrint('_fetchTrades failed: $e');
    }
  }

  Future<void> _fetchEquityCurve() async {
    await Future.wait([
      _fetchEquityCurvePeriod('all', (points) => _equityCurve = points),
      _fetchEquityCurvePeriod('yearly', (points) => _yearlyCurve = points),
      _fetchEquityCurvePeriod('monthly', (points) => _monthlyCurve = points),
    ]);
  }

  Future<void> _fetchEquityCurvePeriod(String period, void Function(List<EquityPoint>) setter) async {
    try {
      final data = await _get('/api/device/bot/equity_curve?period=$period');
      final points = (data['points'] as List? ?? []).map((p) => EquityPoint(
        time: DateTime.tryParse(p['time'] ?? '') ?? DateTime.now(),
        balance: (p['balance'] ?? 0).toDouble(),
      )).toList();
      setter(points);
    } catch (e) {
      debugPrint('_fetchEquityCurve $period failed: $e');
    }
  }

  Future<void> _fetchNotifications() async {
    try {
      final data = await _get('/api/device/notifications');
      final list = (data['notifications'] as List? ?? []);
      final parsed = list.map((n) => NotificationItem.fromJson(n)).toList();
      final newIds = parsed.map((n) => n.id).toSet();
      for (final n in parsed) {
        if (!_seenNotificationIds.contains(n.id)) {
          _seenNotificationIds.add(n.id);
          if (n.type == 'trade_open' || n.type == 'trade_close') {
            NotificationService.instance.showNotification(
              id: NotificationService.nextId,
              title: n.title,
              body: n.message,
              payload: n.id,
            );
          }
        }
      }
      _seenNotificationIds.retainAll(newIds);
      _notifications = parsed;
      _unreadCount = (data['unread_count'] ?? 0) as int;
    } catch (e) {
      debugPrint('_fetchNotifications failed: $e');
    }
  }

  Future<void> markNotificationsRead({String? id}) async {
    try {
      await _post('/api/device/notifications/mark-read', {
        if (id != null) 'id': id,
      });
      if (id != null) {
        final idx = _notifications.indexWhere((n) => n.id == id);
        if (idx >= 0) {
          final old = _notifications[idx];
          _notifications[idx] = NotificationItem(
            id: old.id,
            type: old.type,
            title: old.title,
            message: old.message,
            data: old.data,
            isRead: true,
            createdAt: old.createdAt,
          );
          if (_unreadCount > 0) _unreadCount--;
        }
      } else {
        _notifications = _notifications.map((n) => NotificationItem(
          id: n.id,
          type: n.type,
          title: n.title,
          message: n.message,
          data: n.data,
          isRead: true,
          createdAt: n.createdAt,
        )).toList();
        _unreadCount = 0;
      }
      notifyListeners();
    } catch (e) {
      debugPrint('markNotificationsRead failed: $e');
    }
  }

  Future<void> _tickLive() async {
    await Future.wait([
      _fetchStateAndLogs(),
      _fetchTrades(),
      _get('/api/device/subscription').then((d) => _subscription = d).catchError((e) { debugPrint('_tickLive subscription: $e'); return <String, dynamic>{}; }),
      _get('/api/device/bot/performance').then((d) { _performance = Performance.fromJson(d); }).catchError((e) { debugPrint('_tickLive perf: $e'); return; }),
      _fetchEquityCurve().catchError((e) { debugPrint('_tickLive equity: $e'); }),
      _fetchNotifications().catchError((e) { debugPrint('_tickLive notifications: $e'); }),
    ]);

    if (_botRunning && !isDemo && !hasNoAccounts && !canTrade && !_subscriptionBlocked) {
      _subscriptionBlocked = true;
      addLog('Your free trial has ended. Please subscribe to continue trading.', level: 'WARNING');
      requestSubscription();
      _post('/api/device/bot/stop', {}).then((_) {
        _botRunning = false;
        addLog('Bot stopped automatically due to expired trial/subscription.', level: 'WARNING');
      }).catchError((e) { debugPrint('_tickLive stop: $e'); });
    }
    if (_subscriptionBlocked && (isDemo || canTrade)) {
      _subscriptionBlocked = false;
    }

    notifyListeners();
  }

  Future<void> _fetchStateAndLogs() async {
    try {
      final results = await Future.wait([
        _get('/api/device/bot/state').catchError((e) {
          debugPrint('_fetchStateAndLogs state: $e');
          return <String, dynamic>{};
        }),
        _get('/api/device/bot/logs').catchError((e) {
          debugPrint('_fetchStateAndLogs logs: $e');
          return <String, dynamic>{};
        }),
      ]);
      final stateData = results[0];
      if (stateData.isNotEmpty) {
        _state = BotState.fromApiResponse(stateData);
        _botRunning = stateData['running'] == true;
      } else {
        _state ??= _defaultState();
      }

      final logData = results[1];
      final backendLogs = (logData['logs'] as List?)?.map((l) => LogEntry.fromJson(l)).toList() ?? [];
      if (backendLogs.isNotEmpty) {
        final existingMsgs = _logs.map((e) => e.message).toSet();
        for (final entry in backendLogs) {
          if (!existingMsgs.contains(entry.message)) {
            _logs.add(entry);
          }
        }
        if (_logs.length > 200) _logs.removeRange(0, _logs.length - 200);
      }
    } catch (e) {
      debugPrint('_fetchStateAndLogs failed: $e');
      _state ??= _defaultState();
    }
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
      final result = await _startBotRequest();
      if (result['status'] == 'started') {
        _botRunning = true;
        _subscriptionBlocked = false;
        addLog('Bot started successfully', level: 'TRADE');
        await _device.markBotStarted();
        notifyListeners();
        return true;
      }
      if (result['status'] == 'subscription_needed') {
        _subscriptionBlocked = true;
        addLog('Trial expired. Please subscribe to continue.', level: 'WARNING');
        requestSubscription();
        notifyListeners();
        return false;
      }
      addLog('Failed to start: ${result['error']}', level: 'ERROR');
      notifyListeners();
      return false;
    } catch (e) {
      addLog('Failed to start bot: ${_networkError(e)}', level: 'ERROR');
      notifyListeners();
      return false;
    }
  }

  Future<Map<String, dynamic>> _startBotRequest({bool retried = false}) async {
    final url = baseUrl;
    try {
      final r = await _client.post(
        Uri.parse('$url/api/device/bot/start'),
        headers: _device.headers,
        body: jsonEncode({}),
      ).timeout(_timeout);
      final data = jsonDecode(r.body);
      if (r.statusCode == 200) return {'status': 'started'};
      if (r.statusCode == 402) return {'status': 'subscription_needed'};
      return {'status': 'error', 'error': data['error'] ?? r.body};
    } catch (e) {
      debugPrint('_startBotRequest failed: $e');
      if (!retried) {
        await _resolveUrl();
        return _startBotRequest(retried: true);
      }
      rethrow;
    }
  }

  Future<String?> stopBot() async {
    addLog('Stopping bot...', level: 'WARNING');
    try {
      await _post('/api/device/bot/stop', {});
      _botRunning = false;
      addLog('Bot stopped successfully', level: 'WARNING');
      notifyListeners();
      return null;
    } catch (e) {
      final msg = e.toString();
      addLog('Failed to stop bot: $msg', level: 'ERROR');
      notifyListeners();
      return msg;
    }
  }

  Future<Map<String, dynamic>> closeAllPositions() async {
    addLog('Closing all positions...', level: 'WARNING');
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
    final data = await _get('/api/device/accounts');
    return List<Map<String, dynamic>>.from(data['accounts'] ?? []);
  }

  Future<String?> addAccount(
      String apiKey, String identifier, String password, bool demo) async {
    addLog('Saving account: $identifier');
    if (!_backendReady) {
      await _resolveUrl();
    }
    try {
      addLog('Connecting to $baseUrl...');
      await _post('/api/device/accounts', {
        'api_key': apiKey,
        'identifier': identifier,
        'password': password,
        'demo': demo,
      });
    } catch (e) {
      final clean = _networkError(e);
      addLog('Failed to save account: $clean', level: 'ERROR');
      return clean;
    }
    await _device.saveCredentialsTimestamp();
    addLog('Account saved: $identifier', level: 'TRADE');
    await _fetchState(); // immediately pick up auto-started state
    return null;
  }

  Future<bool> removeAccount(String identifier) async {
    try {
      await _delete('/api/device/accounts/$identifier');
      addLog('Removed: $identifier', level: 'WARNING');
      return true;
    } catch (e) {
      addLog('Failed to remove account: ${_networkError(e)}', level: 'ERROR');
      return false;
    }
  }

  Future<Map<String, dynamic>?> initializePayment(String email, {List<String>? channels}) async {
    final data = await _post('/api/payment/initialize', {
      'email': email,
      if (channels != null) 'channels': channels,
    });
    if (data['access_code'] != null) {
      addLog('Payment link generated');
    }
    return data;
  }

  Future<Map<String, dynamic>?> initMaxelpayPayment(double amount) async {
    try {
      return await _post('/api/payment/maxelpay/init', {
        'amount': amount,
      });
    } catch (e) {
      addLog('Failed to create MaxelPay payment: $e', level: 'ERROR');
      return null;
    }
  }

  Future<bool> verifyPayment(String reference) async {
    await _post('/api/payment/verify', {'reference': reference});
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
    } catch (e) {
      debugPrint('_loadConfig failed: $e');
    }
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
