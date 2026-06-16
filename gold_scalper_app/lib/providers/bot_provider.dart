import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/bot_state.dart';
import '../models/trade.dart';
import '../models/performance.dart';
import '../models/config.dart';
import '../services/api_service.dart';

class BotProvider extends ChangeNotifier {
  final ApiService _api = ApiService();
  Timer? _timer;

  BotState? _state;
  List<Trade> _recentTrades = [];
  Performance? _performance;
  BotConfig _config = BotConfig();
  List<EquityPoint> _equityCurve = [];
  bool _loading = true;

  BotState? get state => _state;
  List<Trade> get recentTrades => _recentTrades;
  Performance? get performance => _performance;
  BotConfig get config => _config;
  List<EquityPoint> get equityCurve => _equityCurve;
  bool get loading => _loading;

  Future<void> init() async {
    _loading = true;
    notifyListeners();
    await Future.wait([
      _loadState(),
      _loadTrades(),
      _loadPerformance(),
      _loadConfig(),
      _loadEquity(),
    ]);
    _loading = false;
    notifyListeners();

    _timer = Timer.periodic(const Duration(seconds: 5), (_) => _loadState());
  }

  Future<void> _loadState() async {
    _state = await _api.getState();
    notifyListeners();
  }

  Future<void> _loadTrades() async {
    _recentTrades = await _api.getRecentTrades(limit: 50);
    notifyListeners();
  }

  Future<void> _loadPerformance() async {
    _performance = await _api.getPerformance();
    notifyListeners();
  }

  Future<void> _loadConfig() async {
    _config = await _api.getConfig();
    notifyListeners();
  }

  Future<void> _loadEquity() async {
    _equityCurve = await _api.getEquityCurve();
    notifyListeners();
  }

  Future<bool> updateConfig(BotConfig cfg) async {
    final ok = await _api.updateConfig(cfg);
    if (ok) {
      _config = cfg;
      notifyListeners();
    }
    return ok;
  }

  Future<bool> startBot() async {
    final ok = await _api.startBot();
    if (ok) await _loadState();
    return ok;
  }

  Future<bool> stopBot() async {
    final ok = await _api.stopBot();
    if (ok) await _loadState();
    return ok;
  }

  Future<bool> closeAllPositions() async {
    final ok = await _api.closeAllPositions();
    if (ok) await _loadState();
    return ok;
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}
