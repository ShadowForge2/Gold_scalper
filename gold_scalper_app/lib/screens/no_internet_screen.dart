import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../theme.dart';
import 'home_screen.dart';

class NoInternetScreen extends StatefulWidget {
  const NoInternetScreen({super.key});
  @override State<NoInternetScreen> createState() => _NoInternetScreenState();
}

class _NoInternetScreenState extends State<NoInternetScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulseCtrl = AnimationController(
    vsync: this,
    duration: const Duration(seconds: 2),
  )..repeat(reverse: true);

  Timer? _retryTimer;
  bool _checking = false;

  @override
  void initState() {
    super.initState();
    _startRetryTimer();
  }

  void _startRetryTimer() {
    _retryTimer?.cancel();
    _retryTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (!_checking) _checkAndNavigate();
    });
  }

  Future<void> _checkAndNavigate() async {
    if (_checking) return;
    setState(() => _checking = true);
    try {
      final result = await InternetAddress.lookup('google.com')
          .timeout(const Duration(seconds: 3));
      if (result.isNotEmpty && result[0].rawAddress.isNotEmpty) {
        _retryTimer?.cancel();
        _pulseCtrl.dispose();
        if (!context.mounted) return;
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (_) => const HomeScreen()),
        );
        return;
      }
    } catch (_) {}
    if (mounted) setState(() => _checking = false);
  }

  @override
  void dispose() {
    _retryTimer?.cancel();
    _pulseCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: AnimatedBuilder(
        animation: _pulseCtrl,
        builder: (context, _) {
          final pulse = _pulseCtrl.value;
          return Stack(
            fit: StackFit.expand,
            children: [
              Container(
                decoration: BoxDecoration(
                  gradient: RadialGradient(
                    radius: 1.2,
                    colors: [
                      const Color(0xFF1A1200).withValues(alpha: 0.6 + pulse * 0.3),
                      Colors.black,
                    ],
                  ),
                ),
              ),
              Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Transform.translate(
                      offset: Offset(0, -4 + pulse * 4),
                      child: Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: kDanger.withValues(alpha: 0.08),
                          boxShadow: [
                            BoxShadow(
                              color: kDanger.withValues(alpha: pulse * 0.2),
                              blurRadius: 20 + pulse * 20,
                              spreadRadius: pulse * 10,
                            ),
                          ],
                        ),
                        child: Icon(
                          Icons.wifi_off_rounded,
                          color: kDanger.withValues(alpha: 0.6 + pulse * 0.3),
                          size: 40,
                        ),
                      ),
                    ),
                    const SizedBox(height: 32),
                    ShaderMask(
                      shaderCallback: (bounds) => const LinearGradient(
                        colors: [kGold, Color(0xFFD4AF37), kGold],
                        stops: [0.0, 0.5, 1.0],
                      ).createShader(bounds),
                      blendMode: BlendMode.srcIn,
                      child: const Text('No Internet Connection',
                        style: TextStyle(fontSize: 22, fontWeight: FontWeight.w900,
                          color: Colors.white, letterSpacing: 1.0)),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'A stable internet connection is required\nto access the trading dashboard.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.5),
                        fontSize: 13, height: 1.5, letterSpacing: 0.3,
                      ),
                    ),
                    const SizedBox(height: 40),
                    ElevatedButton(
                      onPressed: _checking ? null : _checkAndNavigate,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: kGold,
                        foregroundColor: Colors.black,
                        padding: const EdgeInsets.symmetric(horizontal: 48, vertical: 14),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
                        elevation: 8 + pulse * 4,
                        shadowColor: kGold.withValues(alpha: pulse * 0.4),
                      ),
                      child: _checking
                          ? const SizedBox(
                              width: 18, height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black87),
                            )
                          : const Text('RETRY',
                              style: TextStyle(fontWeight: FontWeight.bold,
                                fontSize: 14, letterSpacing: 2)),
                    ),
                    if (_checking) ...[
                      const SizedBox(height: 16),
                      Text('Checking connection...',
                        style: TextStyle(color: Colors.white.withValues(alpha: 0.3), fontSize: 11)),
                    ],
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
