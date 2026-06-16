class Trade {
  final String id;
  final DateTime entryTime;
  final DateTime? exitTime;
  final String direction;
  final double entryPrice;
  final double? exitPrice;
  final double pnl;
  final double lot;
  final int numTrades;
  final double score;
  final String exitReason;
  final double balance;

  Trade({
    required this.id,
    required this.entryTime,
    this.exitTime,
    required this.direction,
    required this.entryPrice,
    this.exitPrice,
    required this.pnl,
    required this.lot,
    required this.numTrades,
    required this.score,
    this.exitReason = '',
    required this.balance,
  });

  bool get isWin => pnl > 0;
  bool get isOpen => exitTime == null;
  int get barsHeld => exitTime != null
      ? exitTime!.difference(entryTime).inMinutes ~/ 5
      : 0;

  factory Trade.fromJson(Map<String, dynamic> json) {
    return Trade(
      id: json['id'] ?? '',
      entryTime: DateTime.parse(json['entry_time']),
      exitTime: json['exit_time'] != null ? DateTime.parse(json['exit_time']) : null,
      direction: json['direction'] ?? 'BUY',
      entryPrice: (json['entry_price'] ?? 0).toDouble(),
      exitPrice: json['exit_price']?.toDouble(),
      pnl: (json['pnl'] ?? 0).toDouble(),
      lot: (json['lot'] ?? 0).toDouble(),
      numTrades: json['num_trades'] ?? 1,
      score: (json['score'] ?? 0).toDouble(),
      exitReason: json['exit_reason'] ?? '',
      balance: (json['balance'] ?? 0).toDouble(),
    );
  }
}
