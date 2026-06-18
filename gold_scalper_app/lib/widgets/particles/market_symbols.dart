import 'dart:math';
import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class MarketSymbols extends StatelessWidget {
  final double intensity;
  const MarketSymbols({super.key, this.intensity = 1.0});

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        size: Size.infinite,
        painter: _SymbolPainter(intensity),
        isComplex: true,
        willChange: true,
      ),
    );
  }
}

class _SymbolPainter extends CustomPainter {
  final double intensity;
  _SymbolPainter(this.intensity);

  static const _pairs = [
    'EURUSD', 'BTCUSD', 'XAUUSD', 'NASDAQ',
    'SP500', 'GBPJPY', 'ETHUSD',
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(77);
    final tp = TextPainter(textDirection: TextDirection.ltr);
    for (var i = 0; i < 8; i++) {
      final x = rng.nextDouble() * size.width;
      final y = rng.nextDouble() * size.height;
      tp.text = TextSpan(
        text: _pairs[i % _pairs.length],
        style: TextStyle(
          color: AIColors.white.withValues(alpha: 0.06 * intensity),
          fontSize: 8 + rng.nextDouble() * 6,
          fontFamily: 'monospace',
          fontWeight: FontWeight.w300,
        ),
      );
      tp.layout();
      canvas.save();
      canvas.translate(x, y);
      canvas.rotate(rng.nextDouble() * 0.3 - 0.15);
      tp.paint(canvas, Offset.zero);
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(_SymbolPainter old) => old.intensity != intensity;
}
