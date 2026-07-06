class BotState {
  final String status;
  final String state;
  final bool connected;
  final String broker;
  final String symbol;
  final double balance;
  final double dailyPnl;
  final double bid;
  final double ask;
  final String bias;
  final double biasStrength;
  final int openPositions;
  final DateTime timestamp;

  BotState({
    required this.status,
    required this.state,
    required this.connected,
    required this.broker,
    required this.symbol,
    required this.balance,
    required this.dailyPnl,
    required this.bid,
    required this.ask,
    required this.bias,
    required this.biasStrength,
    required this.openPositions,
    required this.timestamp,
  });

  double get spread => ask - bid;
  bool get isTradeable => bias == "BULLISH" || bias == "BEARISH";

  factory BotState.fromJson(Map<String, dynamic> json) {
    return BotState(
      status: json['status'] ?? 'unknown',
      state: json['state'] ?? 'IDLE',
      connected: json['connected'] ?? false,
      broker: json['broker'] ?? '',
      symbol: json['symbol'] ?? 'XAUUSD',
      balance: (json['balance'] ?? 0).toDouble(),
      dailyPnl: (json['daily_pnl'] ?? 0).toDouble(),
      bid: (json['bid'] ?? 0).toDouble(),
      ask: (json['ask'] ?? 0).toDouble(),
      bias: json['bias'] ?? 'NEUTRAL',
      biasStrength: (json['bias_strength'] ?? 0).toDouble(),
      openPositions: json['open_positions'] ?? 0,
      timestamp: DateTime.now(),
    );
  }

  factory BotState.fromApiResponse(Map<String, dynamic> json) {
    final bot = json['bot'] as Map<String, dynamic>? ?? {};
    final account = json['account'] as Map<String, dynamic>? ?? {};
    final bias = bot['bias'] as Map<String, dynamic>? ?? {};
    final positions = bot['positions'] as Map<String, dynamic>? ?? {};

    final isRunning = json['running'] == true;
    final hasError = account['error'] != null;

    return BotState(
      status: isRunning ? 'running' : 'stopped',
      state: bot['state'] ?? (isRunning ? 'AWAITING_SIGNAL' : 'IDLE'),
      connected: !hasError,
      broker: 'Capital.com',
      symbol: bot['symbol'] ?? 'XAUUSD',
      balance: (account['balance'] ?? 0).toDouble(),
      dailyPnl: (positions['daily_pnl'] ?? 0).toDouble(),
      bid: (account['bid'] ?? 0).toDouble(),
      ask: (account['ask'] ?? 0).toDouble(),
      bias: bias['bias'] ?? 'NEUTRAL',
      biasStrength: ((bias['strength'] ?? 0) * 100).toDouble(),
      openPositions: (positions['open_count'] ?? 0).toInt(),
      timestamp: DateTime.now(),
    );
  }

  BotState copyWith({
    String? status, String? state, bool? connected, String? broker,
    String? symbol, double? balance, double? dailyPnl, double? bid,
    double? ask, String? bias, double? biasStrength, int? openPositions,
  }) {
    return BotState(
      status: status ?? this.status,
      state: state ?? this.state,
      connected: connected ?? this.connected,
      broker: broker ?? this.broker,
      symbol: symbol ?? this.symbol,
      balance: balance ?? this.balance,
      dailyPnl: dailyPnl ?? this.dailyPnl,
      bid: bid ?? this.bid,
      ask: ask ?? this.ask,
      bias: bias ?? this.bias,
      biasStrength: biasStrength ?? this.biasStrength,
      openPositions: openPositions ?? this.openPositions,
      timestamp: DateTime.now(),
    );
  }
}
