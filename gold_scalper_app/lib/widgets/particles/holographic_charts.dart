import 'dart:math';
import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class HolographicCharts extends StatelessWidget {
  final double intensity;
  const HolographicCharts({super.key, this.intensity = 1.0});

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        size: Size.infinite,
        painter: _ChartPainter(intensity),
        isComplex: true,
        willChange: true,
      ),
    );
  }
}

class _ChartPainter extends CustomPainter {
  final double intensity;
  _ChartPainter(this.intensity);

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(33);
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;
    final fill = Paint();

    for (var c = 0; c < 5; c++) {
      final cx = rng.nextDouble() * size.width;
      final cy = rng.nextDouble() * size.height;
      final cw = 20 + rng.nextDouble() * 20;
      final ch = 12 + rng.nextDouble() * 12;
      final rot = rng.nextDouble() * 0.4 - 0.2;
      final points = List.generate(8, (_) => rng.nextDouble());

      paint.color = AIColors.hologram.withValues(alpha: 0.04 * intensity);
      canvas.save();
      canvas.translate(cx, cy);
      canvas.rotate(rot);

      final path = Path();
      for (var i = 0; i < points.length; i++) {
        final px = (i / (points.length - 1)) * cw;
        final py = (1 - points[i]) * ch;
        if (i == 0) {
          path.moveTo(px, py);
        } else {
          path.lineTo(px, py);
        }
      }
      canvas.drawPath(path, paint);

      fill.shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          AIColors.hologram.withValues(alpha: 0.03 * intensity),
          AIColors.hologram.withValues(alpha: 0),
        ],
      ).createShader(Rect.fromLTWH(0, 0, cw, ch));
      final fillPath = Path.from(path)
        ..lineTo(cw, ch)
        ..lineTo(0, ch)
        ..close();
      canvas.drawPath(fillPath, fill);
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(_ChartPainter old) => old.intensity != intensity;
}
