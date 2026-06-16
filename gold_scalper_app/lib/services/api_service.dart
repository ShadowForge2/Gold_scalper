import '../models/bot_state.dart';
import '../models/trade.dart';
import '../models/performance.dart';
import '../models/config.dart';
import 'mock_data.dart';

class ApiService {
  final MockData _mock = MockData();

  Future<BotState> getState() async {
    await _delay();
    return _mock.currentState();
  }

  Future<List<Trade>> getRecentTrades({int limit = 50}) async {
    await _delay();
    return _mock.recentTrades(limit);
  }

  Future<List<Trade>> getAllTrades() async {
    await _delay();
    return _mock.allTrades();
  }

  Future<Performance> getPerformance() async {
    await _delay();
    return _mock.performance();
  }

  Future<List<EquityPoint>> getEquityCurve() async {
    await _delay();
    return _mock.equityCurve();
  }

  Future<BotConfig> getConfig() async {
    await _delay();
    return _mock.config();
  }

  Future<bool> updateConfig(BotConfig config) async {
    await _delay();
    return true;
  }

  Future<bool> startBot() async {
    await _delay();
    return true;
  }

  Future<bool> stopBot() async {
    await _delay();
    return true;
  }

  Future<bool> closeAllPositions() async {
    await _delay();
    return true;
  }

  Future<void> _delay() async {
    await Future.delayed(const Duration(milliseconds: 300));
  }
}
