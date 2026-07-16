class Trade {
  final String id;
  final DateTime entryTime;
  final DateTime? exitTime;
  final String direction;
  final double entryPrice;
  final double? currentPrice;
  final double? exitPrice;
  final double pnl;
  final double lot;
  final int numTrades;
  final double score;
  final String exitReason;
  final double balance;
  final String symbol;

  Trade({
    required this.id,
    required this.entryTime,
    this.exitTime,
    required this.direction,
    required this.entryPrice,
    this.currentPrice,
    this.exitPrice,
    required this.pnl,
    required this.lot,
    required this.numTrades,
    required this.score,
    this.exitReason = '',
    required this.balance,
    this.symbol = '',
  });

  bool get isWin => pnl > 0;
  bool get isOpen => exitTime == null;
  int get priceDecimals {
    final s = symbol.toUpperCase();
    if (s.contains('US100') || s.contains('NASD') || s.contains('NAS')) return 1;
    if (s.contains('XAU') || s.contains('GOLD')) return 2;
    return entryPrice > 10000 ? 1 : 2;
  }
  int get barsHeld => exitTime != null
      ? exitTime!.difference(entryTime).inMinutes ~/ 5
      : 0;

  factory Trade.fromJson(Map<String, dynamic> json) {
    DateTime? parseTime(String? val) {
      if (val == null || val.isEmpty) return null;
      return DateTime.tryParse(val);
    }

    return Trade(
      id: json['id']?.toString() ?? json['ticket']?.toString() ?? '',
      entryTime: parseTime(json['entry_time']) ?? DateTime.utc(1970),
      exitTime: parseTime(json['exit_time']),
      direction: json['direction'] ?? 'BUY',
      entryPrice: (json['entry_price'] ?? 0).toDouble(),
      currentPrice: json['current_price']?.toDouble(),
      exitPrice: json['exit_price']?.toDouble(),
      pnl: (json['pnl'] ?? 0).toDouble(),
      lot: (json['lot'] ?? 0).toDouble(),
      numTrades: json['num_trades'] ?? 1,
      score: (json['score'] ?? json['entry_score'] ?? 0).toDouble(),
      exitReason: json['exit_reason'] ?? '',
      balance: (json['balance'] ?? 0).toDouble(),
      symbol: json['symbol'] ?? '',
    );
  }
}
