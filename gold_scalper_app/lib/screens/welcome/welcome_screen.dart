import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../utils/animation_director.dart';
import '../../widgets/effects/vignette.dart';
import '../../widgets/effects/grain_overlay.dart';
import '../../widgets/particles/particle_engine.dart';
import '../../widgets/particles/binary_stream.dart';
import '../../widgets/particles/market_symbols.dart';
import '../../widgets/particles/holographic_charts.dart';
import '../../widgets/particles/neural_network.dart';
import '../../widgets/particles/scan_wave.dart';
import '../../widgets/robot/ai_robot.dart';
import '../../widgets/sentiments/ai_sentiment_engine.dart';
import '../../widgets/ui/logo_section.dart';
import '../../widgets/ui/action_buttons.dart';
import '../../widgets/ui/background_gradient.dart';

class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen>
    with TickerProviderStateMixin {
  late AnimationDirector _director;
  late AnimationController _floatController;
  late AnimationController _eyeBreathController;
  double _eyeBoost = 0.0;
  bool _showButton = false;

  @override
  void initState() {
    super.initState();
    _director = AnimationDirector(this);
    _director.start();

    _floatController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 8),
    )..repeat(reverse: true);

    _eyeBreathController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 6),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _director.dispose();
    _floatController.dispose();
    _eyeBreathController.dispose();
    super.dispose();
  }

  void _onCriticalMessage() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      setState(() => _eyeBoost = 0.25);
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) setState(() => _eyeBoost = 0.0);
      });
    });
  }

  void _onTypingComplete() {
    setState(() => _showButton = true);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AIColors.black,
      body: Stack(
        children: [
          const BackgroundGradient(),
          const GrainOverlay(),
          const Vignette(),
          const ParticleEngine(),
          const BinaryStream(),
          const MarketSymbols(),
          const HolographicCharts(),
          const NeuralNetwork(),
          AiRobot(
            revealAnimation: _director.robotReveal,
            eyeRevealAnimation: _director.eyeReveal,
            floatAnimation: _floatController,
            eyeBaseAnimation: _eyeBreathController,
            eyeIntensityBoost: _eyeBoost,
          ),
          AiSentimentEngine(onCriticalMessage: _onCriticalMessage),
          const ScanWave(),
          Positioned(
            left: 0,
            right: 0,
            bottom: MediaQuery.of(context).size.height * 0.12,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                LogoSection(onTypingComplete: _onTypingComplete),
                const SizedBox(height: 24),
                AnimatedOpacity(
                  opacity: _showButton ? 1.0 : 0.0,
                  duration: const Duration(milliseconds: 800),
                  child: ActionButtons(),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
