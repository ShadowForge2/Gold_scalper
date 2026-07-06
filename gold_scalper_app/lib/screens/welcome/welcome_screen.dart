import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../../theme.dart';
import '../../widgets/holographic_overlay.dart';
import '../../providers/device_provider.dart';
import '../../services/security_service.dart';
import '../home_screen.dart';

class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});
  @override State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen>
    with TickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(seconds: 6),
  )..repeat(reverse: true);

  late final AnimationController _taglineCtrl = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 800),
  );
  late final Animation<double> _taglineFade = CurvedAnimation(
    parent: _taglineCtrl,
    curve: Curves.easeIn,
  );

  static const _title = 'QuantoraFX';
  int _typewriterChars = 0;
  Timer? _twTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<DeviceProvider>().markWelcomeShown();
    });
    _twTimer = Timer.periodic(const Duration(milliseconds: 60), (_) {
      if (_typewriterChars < _title.length) {
        setState(() => _typewriterChars++);
      } else {
        _twTimer?.cancel();
        _taglineCtrl.forward();
      }
    });
  }

  Future<bool> _hasInternet() async {
    try {
      final result = await InternetAddress.lookup('google.com')
          .timeout(const Duration(seconds: 3));
      return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
    } catch (_) {
      return false;
    }
  }

  Future<void> _onGetStarted(BuildContext context) async {
    HapticFeedback.lightImpact();

    if (SecurityService.isJailbroken()) {
      if (!context.mounted) return;
      await showDialog(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => AlertDialog(
          backgroundColor: const Color(0xFF1A1A2E),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          title: const Row(
            children: [
              Icon(Icons.security, color: Colors.redAccent, size: 22),
              SizedBox(width: 10),
              Text('Security Risk',
                  style: TextStyle(color: Colors.white, fontSize: 16)),
            ],
          ),
          content: const Text(
            'This device appears to be rooted or jailbroken.\n'
            'Trading on a compromised device is not allowed.',
            style: TextStyle(color: Colors.white70, fontSize: 13),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(),
              child: const Text('EXIT',
                  style: TextStyle(color: Color(0xFFD4AF37))),
            ),
          ],
        ),
      );
      return;
    }

    if (!await _hasInternet()) {
      if (!context.mounted) return;
      await showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          backgroundColor: const Color(0xFF1A1A2E),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          title: const Row(
            children: [
              Icon(Icons.wifi_off, color: Colors.orange, size: 22),
              SizedBox(width: 10),
              Text('No Connection',
                  style: TextStyle(color: Colors.white, fontSize: 16)),
            ],
          ),
          content: const Text(
            'Internet connection required to access the dashboard.\n'
            'Please connect and try again.',
            style: TextStyle(color: Colors.white70, fontSize: 13),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(),
              child: const Text('TRY AGAIN',
                  style: TextStyle(color: Color(0xFFD4AF37))),
            ),
          ],
        ),
      );
      return;
    }
    if (!context.mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const HomeScreen()),
    );
  }

  @override
  void dispose() {
    _twTimer?.cancel();
    _c.dispose();
    _taglineCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: AnimatedBuilder(
        animation: _c,
        builder: (context, _) {
          final dy = (_c.value - 0.5) * 12;
          final glowOpacity = 0.3 + _c.value * 0.7;
          return Stack(fit: StackFit.expand, children: [
            Container(
              decoration: const BoxDecoration(
                gradient: RadialGradient(
                  radius: 1.2,
                  colors: [Color(0xFF1A1200), Colors.black],
                ),
              ),
            ),
            const HolographicOverlay(),
            Center(
              child: Transform.translate(
                offset: Offset(0, dy),
                child: Opacity(
                  opacity: 0.18 + glowOpacity * 0.12,
                  child: Container(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFF00FF88).withValues(alpha: glowOpacity * 0.25),
                          blurRadius: 20 + glowOpacity * 60,
                          spreadRadius: 10 + glowOpacity * 30,
                        ),
                      ],
                    ),
                    child: Image.asset(
                      'assets/images/robot.png',
                      fit: BoxFit.contain,
                      width: MediaQuery.of(context).size.width * 0.9,
                    ),
                  ),
                ),
              ),
            ),
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 80),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _title.substring(0, _typewriterChars),
                      style: const TextStyle(
                        color: kGold,
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 3,
                      ),
                    ),
                    if (_typewriterChars == _title.length) ...[
                      const SizedBox(height: 8),
                      FadeTransition(
                        opacity: _taglineFade,
                        child: const Text(
                          'Autonomous  •  Intelligent  •  Profitable',
                          style: TextStyle(
                            color: Colors.white54,
                            fontSize: 10,
                            letterSpacing: 2,
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                    ] else
                      const SizedBox(height: 12),
                    if (_typewriterChars == _title.length)
                      DecoratedBox(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(30),
                          boxShadow: [
                            BoxShadow(
                              color: kGold.withValues(alpha: glowOpacity * 0.4),
                              blurRadius: 8 + glowOpacity * 12,
                              spreadRadius: glowOpacity * 3,
                            ),
                          ],
                        ),
                        child: ElevatedButton(
                          onPressed: () => _onGetStarted(context),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: kGold,
                            foregroundColor: Colors.black,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 48,
                              vertical: 14,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          child: const Text(
                            'GET STARTED',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                              letterSpacing: 2,
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ]);
        },
      ),
    );
  }
}
