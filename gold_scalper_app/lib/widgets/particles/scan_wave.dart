import 'dart:math';
import 'package:flutter/material.dart';
import '../../painters/scan_painter.dart';

class ScanWave extends StatefulWidget {
  final void Function()? onScan;
  const ScanWave({super.key, this.onScan});

  @override
  State<ScanWave> createState() => _ScanWaveState();
}

class _ScanWaveState extends State<ScanWave>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  final _rng = Random();
  double _progress = -1;
  double _elapsed = 0;
  double _interval = 8;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this, duration: const Duration(seconds: 30),
    )..repeat();
    _interval = 8 + _rng.nextDouble() * 4;
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        _elapsed += _controller.value * 30;
        if (_progress < 0 && _elapsed >= _interval) {
          _progress = 0;
          _elapsed = 0;
          _interval = 8 + _rng.nextDouble() * 4;
          widget.onScan?.call();
        }
        if (_progress >= 0) {
          _progress += _controller.value * 0.15;
          if (_progress >= 1.0) _progress = -1;
        }
        if (_progress < 0) return const SizedBox.shrink();
        return CustomPaint(
          size: Size.infinite,
          painter: ScanWavePainter(_progress),
        );
      },
    );
  }
}
