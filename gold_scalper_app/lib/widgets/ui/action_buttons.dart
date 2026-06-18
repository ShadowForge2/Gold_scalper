import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class ActionButtons extends StatelessWidget {
  final VoidCallback? onExplore;

  const ActionButtons({super.key, this.onExplore});

  @override
  Widget build(BuildContext context) {
    return _buildButton(
      'EXPLORE TERMINAL',
      AIColors.hologram.withValues(alpha: 0.6),
      onExplore ?? () {},
    );
  }

  Widget _buildButton(String text, Color color, VoidCallback onPressed) {
    return SizedBox(
      width: 260,
      height: 50,
      child: OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          foregroundColor: color,
          side: BorderSide(color: color.withValues(alpha: 0.6), width: 1.5),
          backgroundColor: color.withValues(alpha: 0.08),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        child: Text(
          text,
          style: TextStyle(
            fontFamily: 'monospace',
            fontSize: 13,
            letterSpacing: 5,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}
