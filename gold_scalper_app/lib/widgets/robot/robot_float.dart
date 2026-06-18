import 'package:flutter/material.dart';

class RobotFloat extends StatelessWidget {
  final Animation<double> floatAnimation;
  final Widget child;
  final double parallaxX;

  const RobotFloat({
    super.key,
    required this.floatAnimation,
    required this.child,
    this.parallaxX = 0.0,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: floatAnimation,
      builder: (context, child) {
        final floatVal = Curves.easeInOut.transform(floatAnimation.value);
        final offset = (floatVal - 0.5) * 8;
        return Transform.translate(
          offset: Offset(parallaxX, offset),
          child: child,
        );
      },
      child: child,
    );
  }
}
