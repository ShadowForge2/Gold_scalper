import 'dart:math';
import 'package:flutter/material.dart';
import '../../models/ai_message.dart';
import 'sentiment_manager.dart';
import 'sentiment_widget.dart';

class AiSentimentEngine extends StatefulWidget {
  final void Function()? onCriticalMessage;

  const AiSentimentEngine({super.key, this.onCriticalMessage});

  @override
  State<AiSentimentEngine> createState() => _AiSentimentEngineState();
}

class _ActiveSentiment {
  final AiMessage message;
  final double seed;
  double age = 0;

  _ActiveSentiment({required this.message, required Random rng})
      : seed = rng.nextDouble() * 100;

  bool get isDead => age > 3.7;
}

class _AiSentimentEngineState extends State<AiSentimentEngine>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  final _active = <_ActiveSentiment>[];
  final _manager = SentimentManager();
  final _rng = Random();
  double _nextSpawn = 1.5;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 60),
    )..repeat();
    _nextSpawn = 1.5 + _rng.nextDouble() * 0.5;
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _update(double dt) {
    for (var i = _active.length - 1; i >= 0; i--) {
      _active[i].age += dt * 60;
      if (_active[i].isDead) _active.removeAt(i);
    }

    if (_elapsed >= _nextSpawn) {
      _spawnNext();
      _nextSpawn = _elapsed + 2.0 + _rng.nextDouble() * 0.8;
    }
  }

  double _elapsed = 0;

  void _spawnNext() {
    final message = _manager.pickNext();
    if (message.isCritical) widget.onCriticalMessage?.call();
    _active.add(_ActiveSentiment(message: message, rng: _rng));
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        _elapsed = _controller.value * 60;
        _update(_controller.value);
        return Stack(
          children: _active.map((a) {
            final pos = _zonePosition(a.message.zone, context);
            return Positioned(
              left: pos.dx,
              top: pos.dy,
              child: SentimentWidget(
                message: a.message,
                age: a.age,
                seed: a.seed,
              ),
            );
          }).toList(),
        );
      },
    );
  }

  Offset _zonePosition(SentimentZone zone, BuildContext context) {
    final size = MediaQuery.of(context).size;
    switch (zone) {
      case SentimentZone.leftHud:  return Offset(size.width * 0.05,  size.height * 0.15);
      case SentimentZone.rightHud: return Offset(size.width * 0.65,  size.height * 0.15);
      case SentimentZone.chest:    return Offset(size.width * 0.3,   size.height * 0.58);
      case SentimentZone.leftArm:  return Offset(size.width * 0.03,  size.height * 0.45);
      case SentimentZone.rightArm: return Offset(size.width * 0.68,  size.height * 0.45);
      case SentimentZone.lowerHud: return Offset(size.width * 0.15,  size.height * 0.78);
    }
  }
}
