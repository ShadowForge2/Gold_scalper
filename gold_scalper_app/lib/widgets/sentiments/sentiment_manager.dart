import 'dart:math';
import '../../models/ai_message.dart';
import 'sentiment_data.dart';

enum _NarrativePhase { boot, market, trading, protection, aggressive }

class SentimentManager {
  final Random _rng = Random();
  final _recentMessages = <String>[];
  _NarrativePhase _phase = _NarrativePhase.boot;
  int _bootIndex = 0;
  int _cycleIndex = 0;

  AiMessage pickNext() {
    final message = _selectMessage();
    _recentMessages.add(message.text);
    if (_recentMessages.length > 10) {
      _recentMessages.removeAt(0);
    }
    _cycleIndex++;
    return message;
  }

  AiMessage _selectMessage() {
    List<AiMessage> pool;

    switch (_phase) {
      case _NarrativePhase.boot:
        if (_bootIndex < bootMessages.length) {
          return bootMessages[_bootIndex++];
        }
        _phase = _NarrativePhase.market;
        _cycleIndex = 0;
        pool = marketMessages;
        break;
      case _NarrativePhase.market:
        pool = marketMessages;
        if (_cycleIndex > 2 && _rng.nextDouble() < 0.15) {
          _phase = _NarrativePhase.trading;
          _cycleIndex = 0;
          pool = tradingMessages;
        }
        break;
      case _NarrativePhase.trading:
        pool = tradingMessages;
        if (_cycleIndex > 1 && _rng.nextDouble() < 0.2) {
          _phase = _rng.nextDouble() < 0.3
              ? _NarrativePhase.protection
              : _NarrativePhase.aggressive;
          _cycleIndex = 0;
          pool = _phase == _NarrativePhase.protection
              ? protectionMessages
              : aggressiveMessages;
        }
        break;
      case _NarrativePhase.protection:
        pool = protectionMessages;
        if (_cycleIndex > 0) {
          _phase = _NarrativePhase.market;
          _cycleIndex = 0;
          pool = marketMessages;
        }
        break;
      case _NarrativePhase.aggressive:
        pool = aggressiveMessages;
        if (_cycleIndex > 1) {
          _phase = _NarrativePhase.market;
          _cycleIndex = 0;
          pool = marketMessages;
        }
        break;
    }

    final available = pool.where((m) => !_recentMessages.contains(m.text)).toList();
    if (available.isEmpty) return pool[_rng.nextInt(pool.length)];
    return available[_rng.nextInt(available.length)];
  }
}
