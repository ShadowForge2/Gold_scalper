import 'dart:math';
import 'package:flutter/material.dart';
import '../utils/app_colors.dart';

class NeuralPainter extends CustomPainter {
  final double time;

  NeuralPainter(this.time);

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(42);
    final paint = Paint()
      ..color = AIColors.glow.withValues(alpha: 0.03)
      ..strokeWidth = 0.5;

    for (var i = 0; i < 8; i++) {
      final x1 = rng.nextDouble() * size.width;
      final y1 = rng.nextDouble() * size.height;
      final x2 = rng.nextDouble() * size.width;
      final y2 = rng.nextDouble() * size.height;
      final pulse = 0.5 + 0.5 * sin(time * 2 + i.toDouble());

      paint.color = AIColors.glow.withValues(alpha: 0.02 * pulse);
      canvas.drawLine(Offset(x1, y1), Offset(x2, y2), paint);

      paint.color = AIColors.glow.withValues(alpha: 0.04 * pulse);
      canvas.drawCircle(Offset(x1, y1), 1.5, paint);
      canvas.drawCircle(Offset(x2, y2), 1.5, paint);
    }
  }

  @override
  bool shouldRepaint(NeuralPainter oldDelegate) => oldDelegate.time != time;
}
