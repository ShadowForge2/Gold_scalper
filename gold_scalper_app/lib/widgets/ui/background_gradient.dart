import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class BackgroundGradient extends StatelessWidget {
  const BackgroundGradient({super.key});

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              const Color(0xff050005),
              AIColors.background,
              const Color(0xff000005),
            ],
          ),
        ),
      ),
    );
  }
}
