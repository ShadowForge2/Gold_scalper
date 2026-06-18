import 'package:flutter/material.dart';

class AnimationDirector {
  final AnimationController master;

  late final Animation<double> bootProgress;
  late final Animation<double> robotReveal;
  late final Animation<double> eyeReveal;
  late final Animation<double> glowBuild;
  late final Animation<double> particleRamp;
  late final Animation<double> chartFade;
  late final Animation<double> sentimentEnable;
  late final Animation<double> logoAppear;
  late final Animation<double> buttonSlide;

  AnimationDirector(TickerProvider vsync)
      : master = AnimationController(
          vsync: vsync,
          duration: const Duration(seconds: 16),
        ) {
    _buildAnimations();
  }

  void _buildAnimations() {
    const d = 16.0;

    bootProgress = _interval(0.0 / d, 14.0 / d);
    robotReveal = _interval(1.2 / d, 3.2 / d, Curves.easeInOutCubic);
    eyeReveal = _interval(3.0 / d, 3.8 / d, Curves.easeOut);
    glowBuild = _interval(0.8 / d, 2.5 / d, Curves.easeOut);
    particleRamp = _interval(0.3 / d, 6.0 / d, Curves.easeOut);
    chartFade = _interval(4.0 / d, 6.0 / d, Curves.easeOut);
    sentimentEnable = _interval(4.5 / d, 6.0 / d, Curves.easeOut);
    logoAppear = _interval(10.0 / d, 11.0 / d, Curves.easeOut);
    buttonSlide = _interval(12.0 / d, 13.5 / d, Curves.easeOutCubic);
  }

  Animation<double> _interval(double begin, double end,
      [Curve curve = Curves.easeOut]) {
    return CurvedAnimation(
      parent: master,
      curve: Interval(begin, end, curve: curve),
    );
  }

  void start() {
    master.forward(from: 0);
  }

  void dispose() {
    master.dispose();
  }
}
