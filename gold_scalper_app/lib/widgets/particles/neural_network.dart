import 'dart:math';
import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class NeuralNetwork extends StatefulWidget {
  final double intensity;
  const NeuralNetwork({super.key, this.intensity = 1.0});

  @override
  State<NeuralNetwork> createState() => _NeuralNetworkState();
}

class _NeuralNetworkState extends State<NeuralNetwork>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this, duration: const Duration(seconds: 20),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return CustomPaint(
            size: Size.infinite,
            painter: _NeuralPainter(_controller.value, widget.intensity),
          );
        },
      ),
    );
  }
}

class _NeuralPainter extends CustomPainter {
  final double time;
  final double intensity;
  _NeuralPainter(this.time, this.intensity);

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(42);
    final paint = Paint()..strokeWidth = 0.5;

    for (var i = 0; i < 6; i++) {
      final x1 = rng.nextDouble() * size.width;
      final y1 = rng.nextDouble() * size.height;
      final x2 = rng.nextDouble() * size.width;
      final y2 = rng.nextDouble() * size.height;
      final pulse = 0.5 + 0.5 * sin(time * 2 + i.toDouble());

      paint.color = AIColors.red.withValues(alpha: 0.02 * pulse * intensity);
      canvas.drawLine(Offset(x1, y1), Offset(x2, y2), paint);

      paint.color = AIColors.glow.withValues(alpha: 0.03 * pulse * intensity);
      canvas.drawCircle(Offset(x1, y1), 1.5, paint);
      canvas.drawCircle(Offset(x2, y2), 1.5, paint);
    }
  }

  @override
  bool shouldRepaint(_NeuralPainter old) =>
      old.time != time || old.intensity != intensity;
}
