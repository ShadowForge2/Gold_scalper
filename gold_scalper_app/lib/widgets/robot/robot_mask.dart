import 'package:flutter/material.dart';

class RobotMask extends StatelessWidget {
  final Widget child;
  final double reveal;

  const RobotMask({super.key, required this.child, this.reveal = 1.0});

  @override
  Widget build(BuildContext context) {
    final fadeIn = _easeInOut((reveal * 1.3).clamp(0.0, 1.0));
    return ShaderMask(
      shaderCallback: (bounds) {
        return RadialGradient(
          center: const Alignment(0.0, -0.3),
          radius: 0.65,
          colors: [
            Colors.white.withValues(alpha: fadeIn),
            Colors.white.withValues(alpha: fadeIn * 0.55),
            Colors.black.withValues(alpha: 1.0 - fadeIn * 0.4),
            Colors.black,
          ],
          stops: const [0.0, 0.3, 0.65, 1.0],
        ).createShader(bounds);
      },
      blendMode: BlendMode.dstIn,
      child: ShaderMask(
        shaderCallback: (bounds) {
          return LinearGradient(
            begin: Alignment.centerLeft,
            end: Alignment.centerRight,
            colors: [
              Colors.black,
              Colors.white.withValues(alpha: fadeIn * 0.8),
              Colors.white.withValues(alpha: fadeIn * 0.95),
              Colors.white.withValues(alpha: fadeIn * 0.8),
              Colors.black,
            ],
            stops: const [0.0, 0.12, 0.5, 0.88, 1.0],
          ).createShader(bounds);
        },
        blendMode: BlendMode.dstIn,
        child: child,
      ),
    );
  }

  static double _easeInOut(double t) {
    return t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) * (-2 * t + 2) / 2;
  }
}
