import 'dart:math';
import 'package:flutter/material.dart';

class ChartPainter extends CustomPainter {
  final double progress;
  final Color color;

  ChartPainter({this.progress = 0.0, this.color = const Color(0xff00F5FF)});

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(99);
    final paint = Paint()
      ..color = color.withValues(alpha: .6)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final path = Path();
    final points = 30;
    final visible = (points * progress).ceil();

    for (var i = 0; i < visible && i < points; i++) {
      final x = (i / points) * size.width;
      final y = size.height * .5 +
          (rng.nextDouble() - 0.5) * size.height * .6;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    canvas.drawPath(path, paint);

    final glowPaint = Paint()
      ..color = color.withValues(alpha: .15)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
    canvas.drawPath(path, glowPaint);
  }

  @override
  bool shouldRepaint(ChartPainter oldDelegate) =>
      oldDelegate.progress != progress;
}
