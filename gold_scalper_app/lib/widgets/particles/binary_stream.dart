import 'dart:math';
import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class BinaryStream extends StatelessWidget {
  final double intensity;
  const BinaryStream({super.key, this.intensity = 1.0});

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        size: Size.infinite,
        painter: _BinaryPainter(intensity),
        isComplex: true,
        willChange: true,
      ),
    );
  }
}

class _BinaryPainter extends CustomPainter {
  final double intensity;
  _BinaryPainter(this.intensity);

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(24);
    final tp = TextPainter(textDirection: TextDirection.ltr);
    for (var i = 0; i < 12; i++) {
      final x = rng.nextDouble() * size.width;
      final y = rng.nextDouble() * size.height;
      final str =
          '${rng.nextInt(2)}${rng.nextInt(2)}${rng.nextInt(2)}${rng.nextInt(2)}${rng.nextInt(2)}';
      tp.text = TextSpan(
        text: str,
        style: TextStyle(
          color: AIColors.hologram.withValues(alpha: 0.04 * intensity),
          fontSize: 6 + rng.nextDouble() * 4,
          fontFamily: 'monospace',
        ),
      );
      tp.layout();
      canvas.save();
      canvas.translate(x, y);
      canvas.rotate(rng.nextDouble() * 0.2 - 0.1);
      tp.paint(canvas, Offset.zero);
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(_BinaryPainter old) => old.intensity != intensity;
}
