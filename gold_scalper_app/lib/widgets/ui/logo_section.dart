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
  static const _text = 'QuantoraFX';
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
        ShaderMask(
          shaderCallback: (bounds) => LinearGradient(
            colors: [
              AIColors.white.withValues(alpha: fade),
              const Color(0xFFD4AF37).withValues(alpha: 0.9 * fade),
              AIColors.white.withValues(alpha: fade),
            ],
            stops: const [0.0, 0.5, 1.0],
          ).createShader(bounds),
          blendMode: BlendMode.srcIn,
          child: Text(
            'QuantoraFX',
            style: TextStyle(
              color: Colors.white,
              fontSize: 64,
              fontWeight: FontWeight.w900,
              letterSpacing: 24,
              height: 1.1,
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          _text.substring(0, _index),
          style: TextStyle(
            color: AIColors.hologram.withValues(alpha: 0.9 * fade),
            fontSize: 24,
            letterSpacing: 14,
            fontWeight: FontWeight.w400,
          ),
        ),
      ],
    );
  }
}
