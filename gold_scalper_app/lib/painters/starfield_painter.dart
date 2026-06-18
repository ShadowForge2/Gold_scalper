import 'dart:math';
import 'package:flutter/material.dart';

class StarfieldPainter extends CustomPainter {
  final int starCount;
  final double scroll;

  StarfieldPainter({this.starCount = 100, this.scroll = 0.0});

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(42);
    final paint = Paint()..color = Colors.white.withValues(alpha: .6);

    for (var i = 0; i < starCount; i++) {
      final x = rng.nextDouble() * size.width;
      final rawY = rng.nextDouble() * size.height * 1.5;
      final y = (rawY + scroll * 50) % (size.height * 1.5) - size.height * .25;
      final radius = rng.nextDouble() * 1.5 + 0.5;

      paint.color = Colors.white.withValues(
        alpha: .3 + rng.nextDouble() * .4,
      );
      canvas.drawCircle(Offset(x, y), radius, paint);
    }
  }

  @override
  bool shouldRepaint(StarfieldPainter oldDelegate) =>
      oldDelegate.scroll != scroll;
}
