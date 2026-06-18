import 'dart:async';
import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class LogoSection extends StatefulWidget {
  final void Function()? onTypingComplete;

  const LogoSection({super.key, this.onTypingComplete});

  @override
  State<LogoSection> createState() => _LogoSectionState();
}

class _LogoSectionState extends State<LogoSection>
    with SingleTickerProviderStateMixin {
  late AnimationController _fadeController;
  int _index = 0;
  static const _text = 'GOLD SCALPER PRO';
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..forward();

    _timer = Timer.periodic(const Duration(milliseconds: 120), (_) {
      if (_index < _text.length) {
        setState(() => _index++);
      } else {
        _timer?.cancel();
        widget.onTypingComplete?.call();
      }
    });
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final fade = _fadeController.value;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Hide',
          style: TextStyle(
            color: AIColors.white.withValues(alpha: fade),
            fontSize: 52,
            fontWeight: FontWeight.w700,
            letterSpacing: 20,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          _text.substring(0, _index),
          style: TextStyle(
            color: AIColors.hologram.withValues(alpha: 0.9 * fade),
            fontSize: 24,
            letterSpacing: 12,
            fontWeight: FontWeight.w300,
          ),
        ),
      ],
    );
  }
}
