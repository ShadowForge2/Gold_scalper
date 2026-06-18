import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class GlowingEyes extends StatelessWidget {
  final Animation<double> eyeAnimation;
  final double intensityBoost;

  const GlowingEyes({
    super.key,
    required this.eyeAnimation,
    this.intensityBoost = 0.0,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: eyeAnimation,
      builder: (context, child) {
        final breath = CurvedAnimation(
          parent: eyeAnimation,
          curve: const Cubic(0.4, 0.0, 0.6, 1.0),
        ).value;
        final boosted = (breath + intensityBoost).clamp(0.0, 1.0);
        return Center(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.only(right: 55),
                child: _Eye(breathValue: boosted),
              ),
              Padding(
                padding: const EdgeInsets.only(left: 55),
                child: _Eye(breathValue: boosted),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _Eye extends StatelessWidget {
  final double breathValue;
  const _Eye({required this.breathValue});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 50,
      height: 50,
      child: CustomPaint(
        painter: _EyePainter(breathValue),
      ),
    );
  }
}

class _EyePainter extends CustomPainter {
  final double intensity;
  _EyePainter(this.intensity);

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final baseRadius = size.width * 0.25;

    final darkRed = Color.lerp(
      const Color(0xff4a0000), AIColors.darkRed, intensity)!;
    final brightRed = Color.lerp(
      AIColors.darkRed, AIColors.glow, intensity)!;

    final spill = Paint()
      ..color = AIColors.glow.withValues(alpha: 0.05 * intensity)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 45);
    canvas.drawCircle(center, size.width * 0.8, spill);

    final halo = Paint()
      ..color = brightRed.withValues(alpha: 0.15 * intensity)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 24);
    canvas.drawCircle(center, baseRadius * 3, halo);

    final blur = Paint()
      ..color = brightRed.withValues(alpha: 0.25 * intensity)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);
    canvas.drawCircle(center, baseRadius * 2, blur);

    final glow = Paint()
      ..color = brightRed.withValues(alpha: 0.6 * intensity)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);
    canvas.drawCircle(center, baseRadius * 1.4, glow);

    final core = Paint()
      ..color = Color.lerp(darkRed, const Color(0xffff4444), intensity)!
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 1);
    canvas.drawCircle(center, baseRadius, core);

    final highlight = Paint()
      ..color = Colors.white.withValues(alpha: 0.3 * intensity);
    canvas.drawCircle(
      center - Offset(baseRadius * 0.3, baseRadius * 0.3),
      baseRadius * 0.25,
      highlight,
    );
  }

  @override
  bool shouldRepaint(_EyePainter old) => old.intensity != intensity;
}
