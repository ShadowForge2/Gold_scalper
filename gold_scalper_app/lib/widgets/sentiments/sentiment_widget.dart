import 'dart:math';
import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../models/ai_message.dart';

class SentimentWidget extends StatelessWidget {
  final AiMessage message;
  final double age;
  final double seed;

  const SentimentWidget({
    super.key,
    required this.message,
    required this.age,
    this.seed = 0,
  });

  @override
  Widget build(BuildContext context) {
    final opacity = _currentOpacity;
    if (opacity <= 0) return const SizedBox.shrink();

    final driftY = sin(age * 0.03 + seed) * 4;
    final driftX = cos(age * 0.02 + seed) * 3;

    return Opacity(
      opacity: opacity,
      child: Transform.translate(
        offset: Offset(driftX, driftY - age * 0.3),
        child: Transform.rotate(
          angle: sin(age * 0.02 + seed) * 0.02,
          child: Text(
            message.text,
            style: TextStyle(
              color: AIColors.red.withValues(alpha: 0.9 * _brightness),
              fontSize: 11 + message.priority * 1.5,
              fontFamily: 'monospace',
              fontWeight: FontWeight.w600,
              shadows: [
                Shadow(
                  color: AIColors.red.withValues(alpha: 0.5 * _brightness),
                  blurRadius: 8,
                ),
                Shadow(
                  color: AIColors.glow.withValues(alpha: 0.3 * _brightness),
                  blurRadius: 16,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  double get _currentOpacity {
    if (age < 0.7) return age / 0.7;
    if (age < 3.0) return 1.0;
    if (age < 3.7) return 1.0 - (age - 3.0) / 0.7;
    return 0.0;
  }

  double get _brightness => 0.85 + 0.15 * sin(age * 2.0 + seed);
}
