import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class WelcomeText extends StatelessWidget {
  final Animation<double> revealAnimation;

  const WelcomeText({super.key, required this.revealAnimation});

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: revealAnimation,
      builder: (context, child) {
        final value = CurvedAnimation(
          parent: revealAnimation,
          curve: Curves.easeOut,
        ).value;
        return Opacity(
          opacity: value,
          child: Text(
            'AI POWERED TRADING INTELLIGENCE',
            style: TextStyle(
              color: AIColors.white.withValues(alpha: value * 0.5),
              fontSize: 11,
              letterSpacing: 6,
              fontWeight: FontWeight.w100,
              shadows: [
                Shadow(
                  color: AIColors.hologram.withValues(alpha: 0.2 * value),
                  blurRadius: 8,
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
