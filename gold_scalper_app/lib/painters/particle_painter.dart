import 'dart:math';
import 'package:flutter/material.dart';
import '../models/particle.dart';
import '../utils/app_colors.dart';

class ParticlePainter extends CustomPainter {
  final List<Particle> particles;

  ParticlePainter(this.particles);

  @override
  void paint(Canvas canvas, Size size) {
    final sorted = particles.toList()..sort((a, b) => a.z.compareTo(b.z));
    for (final p in sorted) {
      _drawParticle(canvas, p);
    }
  }

  void _drawParticle(Canvas canvas, Particle p) {
    final paint = Paint();
    switch (p.type) {
      case 'star':
        final sz = p.scale * 1.5;
        final blur = (1 - p.z) * 2;
        paint
          ..color = Colors.white.withValues(alpha: p.opacity)
          ..maskFilter = blur > 0.5
              ? MaskFilter.blur(BlurStyle.normal, blur)
              : null;
        if (p.z > 0.7) {
          canvas.drawCircle(Offset(p.x, p.y), sz, paint);
          paint
            ..color = const Color(0xff4488ff).withValues(alpha: p.opacity * 0.3)
            ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3);
          canvas.drawCircle(Offset(p.x, p.y), sz * 2, paint);
        } else {
          canvas.drawCircle(Offset(p.x, p.y), sz * 0.6, paint);
        }
        if (p.z > 0.85) {
          paint
            ..color = Colors.white.withValues(alpha: p.opacity * 0.15)
            ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
          canvas.drawCircle(Offset(p.x, p.y), sz * 4, paint);
        }
        break;

      case 'binary':
        if (p.label == null) return;
        _drawText(canvas, p.label!, p.x, p.y, 7 + p.scale * 2,
            AIColors.hologram.withValues(alpha: p.opacity), p.rotation);
        break;

      case 'symbol':
        if (p.label == null) return;
        _drawText(canvas, p.label!, p.x, p.y, 8 + p.scale * 3,
            Colors.white.withValues(alpha: p.opacity), p.rotation);
        break;

      case 'chart':
        if (p.chartData == null || p.chartData!.length < 4) return;
        _drawChart(canvas, p);
        break;

      case 'neural':
        final nPaint = Paint()
          ..color = AIColors.red.withValues(alpha: p.opacity)
          ..strokeWidth = 0.5 + p.z
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 1);
        final ex = p.x + cos(p.neuralAngle) * p.neuralLength * p.scale;
        final ey = p.y + sin(p.neuralAngle) * p.neuralLength * p.scale;
        canvas.drawLine(Offset(p.x, p.y), Offset(ex, ey), nPaint);
        paint
          ..color = AIColors.red.withValues(alpha: p.opacity * 1.5)
          ..maskFilter = null;
        canvas.drawCircle(Offset(p.x, p.y), 1.0, paint);
        canvas.drawCircle(Offset(ex, ey), 0.6, paint);
        break;

      case 'dust':
        paint
          ..color = Colors.white.withValues(alpha: p.opacity)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2);
        canvas.drawCircle(Offset(p.x, p.y), 0.8 + p.z * 0.5, paint);
        break;
    }
  }

  void _drawText(Canvas canvas, String text, double x, double y,
      double size, Color color, double rotation) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(color: color, fontSize: size, fontFamily: 'monospace'),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    canvas.save();
    canvas.translate(x, y);
    canvas.rotate(rotation);
    tp.paint(canvas, Offset.zero);
    canvas.restore();
  }

  void _drawChart(Canvas canvas, Particle p) {
    final cw = 28 * p.scale;
    final ch = 16 * p.scale;
    final chartPaint = Paint()
      ..color = AIColors.hologram.withValues(alpha: p.opacity)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    canvas.save();
    canvas.translate(p.x, p.y);
    canvas.rotate(p.rotation * 0.5);

    final path = Path();
    for (var i = 0; i < p.chartData!.length; i++) {
      final px = (i / (p.chartData!.length - 1)) * cw;
      final py = (1 - p.chartData![i]) * ch;
      if (i == 0) {
        path.moveTo(px, py);
      } else {
        path.lineTo(px, py);
      }
    }
    canvas.drawPath(path, chartPaint);

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          AIColors.hologram.withValues(alpha: p.opacity * 0.3),
          AIColors.hologram.withValues(alpha: 0),
        ],
      ).createShader(Rect.fromLTWH(0, 0, cw, ch));
    final fillPath = Path.from(path)
      ..lineTo(cw, ch)
      ..lineTo(0, ch)
      ..close();
    canvas.drawPath(fillPath, fillPaint);
    canvas.restore();
  }

  @override
  bool shouldRepaint(ParticlePainter oldDelegate) => true;
}
