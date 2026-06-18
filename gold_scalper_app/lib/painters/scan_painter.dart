import 'package:flutter/material.dart';

class ScanWavePainter extends CustomPainter {
  final double progress;

  ScanWavePainter(this.progress);

  @override
  void paint(Canvas canvas, Size size) {
    final waveX = progress * size.width;

    final paint = Paint()
      ..color = const Color(0xffff1f1f).withValues(alpha: 0.12 * (1 - progress))
      ..strokeWidth = 2.0
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 15);

    canvas.drawLine(Offset(waveX, 0), Offset(waveX, size.height), paint);

    final glowPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.centerLeft,
        end: Alignment.centerRight,
        colors: [
          const Color(0xffff1f1f).withValues(alpha: 0),
          const Color(0xffff1f1f).withValues(alpha: 0.06 * (1 - progress)),
          const Color(0xffff1f1f).withValues(alpha: 0),
        ],
        stops: const [0.0, 0.5, 1.0],
      ).createShader(Rect.fromLTWH(waveX - 40, 0, 80, size.height));
    canvas.drawRect(Rect.fromLTWH(waveX - 40, 0, 80, size.height), glowPaint);
  }

  @override
  bool shouldRepaint(ScanWavePainter oldDelegate) =>
      oldDelegate.progress != progress;
}
