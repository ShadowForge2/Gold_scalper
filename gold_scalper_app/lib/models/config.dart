class BotConfig {
  double maxDailyLoss;
  double maxEventLoss;
  double exitThreshold;
  double maxSpreadPips;
  int maxTradesPerSession;
  int maxConsecutiveLosses;
  int biasUpdateIntervalSec;
  List<String> allowedSessions;
  bool asiaSession;
  bool londonSession;
  bool newYorkSession;

  BotConfig({
    this.maxDailyLoss = 10,
    this.maxEventLoss = 5,
    this.exitThreshold = 0.5,
    this.maxSpreadPips = 50,
    this.maxTradesPerSession = 10,
    this.maxConsecutiveLosses = 3,
    this.biasUpdateIntervalSec = 60,
    this.allowedSessions = const ['ASIA', 'LONDON', 'NEW_YORK'],
    this.asiaSession = false,
    this.londonSession = false,
    this.newYorkSession = false,
  });

  List<String> get activeSessions {
    final s = <String>[];
    if (asiaSession) s.add('ASIA');
    if (londonSession) s.add('LONDON');
    if (newYorkSession) s.add('NEW_YORK');
    return s;
  }

  BotConfig copyWith({
    double? maxDailyLoss,
    double? maxEventLoss,
    double? exitThreshold,
    double? maxSpreadPips,
    int? maxTradesPerSession,
    int? maxConsecutiveLosses,
    int? biasUpdateIntervalSec,
    bool? asiaSession,
    bool? londonSession,
    bool? newYorkSession,
  }) {
    return BotConfig(
      maxDailyLoss: maxDailyLoss ?? this.maxDailyLoss,
      maxEventLoss: maxEventLoss ?? this.maxEventLoss,
      exitThreshold: exitThreshold ?? this.exitThreshold,
      maxSpreadPips: maxSpreadPips ?? this.maxSpreadPips,
      maxTradesPerSession: maxTradesPerSession ?? this.maxTradesPerSession,
      maxConsecutiveLosses: maxConsecutiveLosses ?? this.maxConsecutiveLosses,
      biasUpdateIntervalSec: biasUpdateIntervalSec ?? this.biasUpdateIntervalSec,
      asiaSession: asiaSession ?? this.asiaSession,
      londonSession: londonSession ?? this.londonSession,
      newYorkSession: newYorkSession ?? this.newYorkSession,
    );
  }

  factory BotConfig.fromJson(Map<String, dynamic> json) {
    double _d(String key, double fallback) =>
        double.tryParse(json[key]?.toString() ?? '') ?? fallback;
    int _i(String key, int fallback) =>
        int.tryParse(json[key]?.toString() ?? '') ?? fallback;
    final sessions = (json['ALLOWED_SESSIONS'] as String? ?? 'ASIA,LONDON,NEW_YORK')
        .toUpperCase()
        .split(',')
        .map((s) => s.trim())
        .toList();

    return BotConfig(
      maxDailyLoss: _d('MAX_DAILY_LOSS_USD', 10),
      maxEventLoss: _d('MAX_EVENT_LOSS_USD', 5),
      exitThreshold: _d('EXIT_THRESHOLD_TIGHT', 0.5),
      maxSpreadPips: _d('MAX_SPREAD_PIPS', 50),
      maxTradesPerSession: _i('MAX_TRADES_PER_SESSION', 10),
      maxConsecutiveLosses: _i('MAX_CONSECUTIVE_LOSSES', 3),
      biasUpdateIntervalSec: _i('BIAS_UPDATE_INTERVAL_SEC', 60),
      asiaSession: sessions.contains('ASIA'),
      londonSession: sessions.contains('LONDON'),
      newYorkSession: sessions.contains('NEW_YORK'),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'MAX_DAILY_LOSS_USD': maxDailyLoss.toString(),
      'MAX_EVENT_LOSS_USD': maxEventLoss.toString(),
      'EXIT_THRESHOLD_TIGHT': exitThreshold.toString(),
      'MAX_SPREAD_PIPS': maxSpreadPips.toString(),
      'MAX_TRADES_PER_SESSION': maxTradesPerSession.toString(),
      'MAX_CONSECUTIVE_LOSSES': maxConsecutiveLosses.toString(),
      'BIAS_UPDATE_INTERVAL_SEC': biasUpdateIntervalSec.toString(),
      'ALLOWED_SESSIONS': activeSessions.join(','),
    };
  }
}
