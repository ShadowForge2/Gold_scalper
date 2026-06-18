import 'package:flutter/material.dart';

class Vignette extends StatelessWidget {
  const Vignette({super.key});

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Container(
        decoration: BoxDecoration(
          gradient: RadialGradient(
            center: Alignment.center,
            radius: 0.7,
            colors: [
              Colors.transparent,
              Colors.black.withValues(alpha: 0.5),
              Colors.black.withValues(alpha: 0.85),
            ],
            stops: const [0.4, 0.8, 1.0],
          ),
        ),
      ),
    );
  }
}
