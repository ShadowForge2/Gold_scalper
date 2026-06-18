import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class AmbientCoreGlow extends StatelessWidget {
  final Animation<double> revealAnimation;
  final Animation<double> floatAnimation;

  const AmbientCoreGlow({
    super.key,
    required this.revealAnimation,
    required this.floatAnimation,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([revealAnimation, floatAnimation]),
      builder: (context, child) {
        final reveal = Curves.easeOut.transform(revealAnimation.value);
        final floatVal = Curves.easeInOut.transform(floatAnimation.value);
        final floatOffset = (floatVal - 0.5) * 8;

        return Positioned(
          left: 0,
          right: 0,
          top: MediaQuery.of(context).size.height * 0.25 + floatOffset,
          child: Center(
            child: Container(
              width: 400,
              height: 500,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  center: const Alignment(0.0, -0.2),
                  radius: 0.6,
                  colors: [
                    AIColors.glow.withValues(alpha: 0.08 * reveal),
                    AIColors.darkRed.withValues(alpha: 0.04 * reveal),
                    AIColors.background.withValues(alpha: 0),
                  ],
                  stops: const [0.0, 0.5, 1.0],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
