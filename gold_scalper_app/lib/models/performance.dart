class Performance {
  final int totalTrades;
  final int wins;
  final int losses;
  final double winRate;
  final double grossProfit;
  final double grossLoss;
  final double netPnl;
  final double profitFactor;
  final double avgWin;
  final double avgLoss;
  final double maxDrawdown;
  final double startingBalance;
  final double endingBalance;
  final double returnPct;
  final List<MonthlyBreakdown> monthly;
  final List<DailyBreakdown> daily;

  Performance({
    required this.totalTrades,
    required this.wins,
    required this.losses,
    required this.winRate,
    required this.grossProfit,
    required this.grossLoss,
    required this.netPnl,
    required this.profitFactor,
    required this.avgWin,
    required this.avgLoss,
    required this.maxDrawdown,
    required this.startingBalance,
    required this.endingBalance,
    required this.returnPct,
    required this.monthly,
    required this.daily,
  });

  factory Performance.fromJson(Map<String, dynamic> json) {
    return Performance(
      totalTrades: json['trades'] ?? 0,
      wins: json['wins'] ?? 0,
      losses: json['losses'] ?? 0,
      winRate: (json['win_rate'] ?? 0).toDouble(),
      grossProfit: (json['gross_profit'] ?? 0).toDouble(),
      grossLoss: (json['gross_loss'] ?? 0).toDouble(),
      netPnl: (json['net_pnl'] ?? 0).toDouble(),
      profitFactor: (json['profit_factor'] ?? 0).toDouble(),
      avgWin: (json['avg_win'] ?? 0).toDouble(),
      avgLoss: (json['avg_loss'] ?? 0).toDouble(),
      maxDrawdown: (json['max_dd'] ?? 0).toDouble(),
      startingBalance: (json['starting_balance'] ?? 0).toDouble(),
      endingBalance: (json['ending_balance'] ?? 0).toDouble(),
      returnPct: (json['return_pct'] ?? 0).toDouble(),
      monthly: (json['monthly'] as List?)
              ?.map((m) => MonthlyBreakdown.fromJson(m))
              .toList() ??
          [],
      daily: (json['daily'] as List?)
              ?.map((d) => DailyBreakdown.fromJson(d))
              .toList() ??
          [],
    );
  }
}

class DailyBreakdown {
  final String date;
  final int trades;
  final double pnl;
  final double winRate;

  DailyBreakdown({
    required this.date,
    required this.trades,
    required this.pnl,
    required this.winRate,
  });

  factory DailyBreakdown.fromJson(Map<String, dynamic> json) {
    return DailyBreakdown(
      date: json['date'] ?? '',
      trades: json['trades'] ?? 0,
      pnl: (json['pnl'] ?? 0).toDouble(),
      winRate: (json['wr'] ?? 0).toDouble(),
    );
  }
}

class MonthlyBreakdown {
  final String month;
  final int trades;
  final double pnl;
  final double winRate;

  MonthlyBreakdown({
    required this.month,
    required this.trades,
    required this.pnl,
    required this.winRate,
  });

  factory MonthlyBreakdown.fromJson(Map<String, dynamic> json) {
    return MonthlyBreakdown(
      month: json['month'] ?? '',
      trades: json['trades'] ?? 0,
      pnl: (json['pnl'] ?? 0).toDouble(),
      winRate: (json['wr'] ?? 0).toDouble(),
    );
  }
}

class EquityPoint {
  final DateTime time;
  final double balance;

  EquityPoint({required this.time, required this.balance});
}
