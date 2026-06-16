class BotConfig {
  double lotMultiplier;
  double maxDailyLoss;
  double maxEventLoss;
  double signalEntryThreshold;
  double exitThreshold;
  double maxSpreadPips;
  int maxTradesPerEvent;
  int maxTradesPerSession;
  int maxConsecutiveLosses;
  int reEntryCooldownSec;
  int biasUpdateIntervalSec;
  List<String> allowedSessions;
  bool asiaSession;
  bool londonSession;
  bool newYorkSession;

  BotConfig({
    this.lotMultiplier = 2,
    this.maxDailyLoss = 10,
    this.maxEventLoss = 5,
    this.signalEntryThreshold = 0.5,
    this.exitThreshold = 0.5,
    this.maxSpreadPips = 50,
    this.maxTradesPerEvent = 10,
    this.maxTradesPerSession = 10,
    this.maxConsecutiveLosses = 3,
    this.reEntryCooldownSec = 60,
    this.biasUpdateIntervalSec = 60,
    this.allowedSessions = const ['ASIA', 'LONDON', 'NEW_YORK'],
    this.asiaSession = true,
    this.londonSession = true,
    this.newYorkSession = true,
  });

  List<String> get activeSessions {
    final s = <String>[];
    if (asiaSession) s.add('ASIA');
    if (londonSession) s.add('LONDON');
    if (newYorkSession) s.add('NEW_YORK');
    return s;
  }

  BotConfig copyWith({
    double? lotMultiplier,
    double? maxDailyLoss,
    double? maxEventLoss,
    double? signalEntryThreshold,
    double? exitThreshold,
    double? maxSpreadPips,
    int? maxTradesPerEvent,
    int? maxTradesPerSession,
    int? maxConsecutiveLosses,
    int? reEntryCooldownSec,
    int? biasUpdateIntervalSec,
    bool? asiaSession,
    bool? londonSession,
    bool? newYorkSession,
  }) {
    return BotConfig(
      lotMultiplier: lotMultiplier ?? this.lotMultiplier,
      maxDailyLoss: maxDailyLoss ?? this.maxDailyLoss,
      maxEventLoss: maxEventLoss ?? this.maxEventLoss,
      signalEntryThreshold: signalEntryThreshold ?? this.signalEntryThreshold,
      exitThreshold: exitThreshold ?? this.exitThreshold,
      maxSpreadPips: maxSpreadPips ?? this.maxSpreadPips,
      maxTradesPerEvent: maxTradesPerEvent ?? this.maxTradesPerEvent,
      maxTradesPerSession: maxTradesPerSession ?? this.maxTradesPerSession,
      maxConsecutiveLosses: maxConsecutiveLosses ?? this.maxConsecutiveLosses,
      reEntryCooldownSec: reEntryCooldownSec ?? this.reEntryCooldownSec,
      biasUpdateIntervalSec: biasUpdateIntervalSec ?? this.biasUpdateIntervalSec,
      asiaSession: asiaSession ?? this.asiaSession,
      londonSession: londonSession ?? this.londonSession,
      newYorkSession: newYorkSession ?? this.newYorkSession,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'LOT_MULTIPLIER': lotMultiplier.toString(),
      'MAX_DAILY_LOSS_USD': maxDailyLoss.toString(),
      'MAX_EVENT_LOSS_USD': maxEventLoss.toString(),
      'SIGNAL_ENTRY_THRESHOLD': signalEntryThreshold.toString(),
      'EXIT_THRESHOLD_TIGHT': exitThreshold.toString(),
      'MAX_SPREAD_PIPS': maxSpreadPips.toString(),
      'MAX_TRADES_PER_EVENT': maxTradesPerEvent.toString(),
      'MAX_TRADES_PER_SESSION': maxTradesPerSession.toString(),
      'MAX_CONSECUTIVE_LOSSES': maxConsecutiveLosses.toString(),
      'RE_ENTRY_COOLDOWN_SEC': reEntryCooldownSec.toString(),
      'BIAS_UPDATE_INTERVAL_SEC': biasUpdateIntervalSec.toString(),
      'ALLOWED_SESSIONS': activeSessions.join(','),
    };
  }
}
