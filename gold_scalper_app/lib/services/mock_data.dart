import 'dart:math';
import '../models/bot_state.dart';
import '../models/trade.dart';
import '../models/performance.dart';
import '../models/config.dart';

class MockData {
  final _random = Random(42);
  int _tradeCounter = 0;

  BotState currentState() {
    final states = ['AWAITING_SIGNAL', 'AWAITING_SIGNAL', 'AWAITING_SIGNAL', 'IDLE', 'IN_TRADE'];
    final biases = ['BULLISH', 'BEARISH', 'NEUTRAL', 'BULLISH', 'BULLISH'];
    final bIdx = DateTime.now().hour % biases.length;

    return BotState(
      status: 'healthy',
      state: states[DateTime.now().minute % states.length],
      connected: true,
      broker: 'CAPITAL',
      symbol: 'XAUUSD',
      balance: 20.0 + _random.nextDouble() * 5,
      dailyPnl: _random.nextDouble() * 2 - 0.5,
      bid: 4180 + _random.nextDouble() * 20,
      ask: 4180.5 + _random.nextDouble() * 20,
      bias: biases[bIdx],
      biasStrength: biases[bIdx] == 'NEUTRAL' ? 0.0 : 0.3,
      openPositions: bIdx == 4 ? 1 : 0,
      timestamp: DateTime.now(),
    );
  }

  List<Trade> recentTrades(int limit) {
    return allTrades().reversed.take(limit).toList();
  }

  List<Trade> allTrades() {
    return List.generate(50, (i) => _generateTrade(i));
  }

  Trade _generateTrade(int i) {
    _tradeCounter++;
    final isWin = _random.nextDouble() < 0.69;
    final dir = _random.nextBool() ? 'BUY' : 'SELL';
    final entry = 2700 + _random.nextDouble() * 200;
    final pnl = isWin ? 10 + _random.nextDouble() * 50 : -(_random.nextDouble() * 20 + 5);
    final exitPx = dir == 'BUY' ? entry + pnl / 100 : entry - pnl / 100;
    final daysAgo = i * 2;
    final day = DateTime.now().subtract(Duration(days: daysAgo ~/ 3));
    final entryTime = DateTime(day.year, day.month, day.day, _random.nextInt(8) + 8, _random.nextInt(60));
    final exitTime = entryTime.add(Duration(minutes: _random.nextInt(120) + 15));

    return Trade(
      id: 'TRADE_${_tradeCounter.toString().padLeft(4, '0')}',
      entryTime: entryTime,
      exitTime: isWin ? exitTime : entryTime.add(Duration(minutes: _random.nextInt(30) + 5)),
      direction: dir,
      entryPrice: double.parse(entry.toStringAsFixed(2)),
      exitPrice: double.parse(exitPx.toStringAsFixed(2)),
      pnl: double.parse(pnl.toStringAsFixed(2)),
      lot: 0.02 + _random.nextDouble() * 0.08,
      numTrades: 1,
      score: 0.5 + _random.nextDouble() * 0.5,
      exitReason: isWin ? 'take_profit' : 'stop_loss',
      balance: 20.0 + i * 3 + pnl,
    );
  }

  Performance performance() {
    return Performance(
      totalTrades: 385,
      wins: 266,
      losses: 119,
      winRate: 69.1,
      grossProfit: 71800,
      grossLoss: 16900,
      netPnl: 55194.81,
      profitFactor: 4.25,
      avgWin: 270.0,
      avgLoss: 142.0,
      maxDrawdown: 2186.46,
      startingBalance: 20.0,
      endingBalance: 55214.81,
      returnPct: 275974.05,
      monthly: [
        MonthlyBreakdown(month: '2025-01', trades: 22, pnl: 50.20, winRate: 77.3),
        MonthlyBreakdown(month: '2025-02', trades: 58, pnl: 817.07, winRate: 63.8),
        MonthlyBreakdown(month: '2025-03', trades: 18, pnl: 1194.28, winRate: 61.1),
        MonthlyBreakdown(month: '2025-04', trades: 23, pnl: -126.32, winRate: 52.2),
        MonthlyBreakdown(month: '2025-05', trades: 21, pnl: 2680.58, winRate: 47.6),
        MonthlyBreakdown(month: '2025-06', trades: 28, pnl: 5047.00, winRate: 75.0),
        MonthlyBreakdown(month: '2025-07', trades: 31, pnl: 2043.00, winRate: 71.0),
        MonthlyBreakdown(month: '2025-08', trades: 19, pnl: 4773.00, winRate: 89.5),
        MonthlyBreakdown(month: '2025-09', trades: 41, pnl: 10998.00, winRate: 80.5),
        MonthlyBreakdown(month: '2025-10', trades: 50, pnl: 16612.00, winRate: 74.0),
        MonthlyBreakdown(month: '2025-11', trades: 35, pnl: 5978.00, winRate: 71.4),
        MonthlyBreakdown(month: '2025-12', trades: 39, pnl: 5128.00, winRate: 61.5),
      ],
      daily: [
        DailyBreakdown(date: '2026-06-15', trades: 5, pnl: 124.50, winRate: 80.0),
        DailyBreakdown(date: '2026-06-14', trades: 8, pnl: -42.20, winRate: 50.0),
        DailyBreakdown(date: '2026-06-13', trades: 12, pnl: 256.80, winRate: 75.0),
        DailyBreakdown(date: '2026-06-12', trades: 4, pnl: 89.10, winRate: 100.0),
        DailyBreakdown(date: '2026-06-11', trades: 10, pnl: -15.40, winRate: 40.0),
        DailyBreakdown(date: '2026-06-10', trades: 7, pnl: 112.30, winRate: 71.4),
        DailyBreakdown(date: '2026-06-09', trades: 15, pnl: 420.50, winRate: 86.6),
      ],
    );
  }

  List<EquityPoint> equityCurve() {
    final points = <EquityPoint>[];
    double bal = 20;
    for (int d = 0; d < 365; d++) {
      final change = (bal * (_random.nextDouble() - 0.45) * 0.02).clamp(-bal * 0.5, bal * 2);
      bal = (bal + change).clamp(0, double.infinity);
      if (bal > 55000) break;
      points.add(EquityPoint(
        time: DateTime(2025, 1, 1).add(Duration(days: d)),
        balance: bal,
      ));
    }
    return points;
  }

  BotConfig config() {
    return BotConfig();
  }
}
