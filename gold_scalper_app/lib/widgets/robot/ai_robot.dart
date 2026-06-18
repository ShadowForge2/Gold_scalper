import 'package:flutter/material.dart';
import 'robot_mask.dart';
import 'robot_float.dart';
import 'ambient_core_glow.dart';
import 'glowing_eyes.dart';

class AiRobot extends StatelessWidget {
  final Animation<double> revealAnimation;
  final Animation<double> eyeRevealAnimation;
  final Animation<double> floatAnimation;
  final Animation<double> eyeBaseAnimation;
  final double eyeIntensityBoost;
  final double parallaxX;

  const AiRobot({
    super.key,
    required this.revealAnimation,
    required this.eyeRevealAnimation,
    required this.floatAnimation,
    required this.eyeBaseAnimation,
    this.eyeIntensityBoost = 0.0,
    this.parallaxX = 0.0,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([
        revealAnimation, eyeRevealAnimation, floatAnimation, eyeBaseAnimation,
      ]),
      builder: (context, child) {
        final reveal = Curves.easeInOutCubic.transform(revealAnimation.value);
        final eyeReveal = Curves.easeOut.transform(eyeRevealAnimation.value);
        final floatVal = Curves.easeInOut.transform(floatAnimation.value);
        final floatOffset = (floatVal - 0.5) * 8;
        final revealOffset = (1.0 - reveal) * 20;

        return Stack(
          children: [
            AmbientCoreGlow(
              revealAnimation: revealAnimation,
              floatAnimation: floatAnimation,
            ),
            Transform.translate(
              offset: Offset(parallaxX, floatOffset + revealOffset),
              child: Opacity(
                opacity: reveal.clamp(0.0, 1.0),
                child: RobotMask(
                  reveal: reveal,
                  child: ColorFiltered(
                    colorFilter: ColorFilter.mode(
                      const Color(0xff1a1a2e).withValues(alpha: 0.6),
                      BlendMode.multiply,
                    ),
                    child: Opacity(
                      opacity: (reveal * 1.4).clamp(0.0, 0.18),
                      child: Image.asset(
                        'assets/images/robot.png',
                        fit: BoxFit.contain,
                        width: double.infinity,
                        height: double.infinity,
                      ),
                    ),
                  ),
                ),
              ),
            ),
            Opacity(
              opacity: eyeReveal,
              child: RobotFloat(
                floatAnimation: floatAnimation,
                parallaxX: parallaxX * 0.5,
                child: GlowingEyes(
                  eyeAnimation: AlwaysStoppedAnimation(0),
                  intensityBoost: eyeIntensityBoost,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
