import 'package:flutter/material.dart';
import '../theme.dart';
import '../widgets/ui/haptic.dart';

class OnboardingTutorial extends StatefulWidget {
  final VoidCallback onDismiss;
  const OnboardingTutorial({super.key, required this.onDismiss});

  @override
  State<OnboardingTutorial> createState() => _OnboardingTutorialState();
}

class _OnboardingTutorialState extends State<OnboardingTutorial> {
  final _ctrl = PageController();
  int _page = 0;

  static const _pages = [
    _TutorialPage(
      icon: Icons.auto_awesome_rounded,
      title: 'QuantoraFX AI Trading Core',
      body: 'Automate your Forex trading with precision.\n\n'
          'This quick guide will show you how to get started in 3 steps.',
      color: kGold,
    ),
    _TutorialPage(
      icon: Icons.settings_rounded,
      title: '1. Connect Your Account',
      body: 'Go to the Settings tab and enter your Capital.com API credentials.\n\n'
          'You can connect a Demo account to start risk-free, or a Live account for real trading.',
      color: kGold,
    ),
    _TutorialPage(
      icon: Icons.tune_rounded,
      title: '2. Configure & Start',
      body: 'Switch to the Controls tab to adjust trading parameters,\n'
          'then press Start Bot to begin automated trading.',
      color: Colors.green,
    ),
    _TutorialPage(
      icon: Icons.dashboard_rounded,
      title: '3. Monitor Performance',
      body: 'The Dashboard shows your balance, market bias, equity curve,\n'
          'and real-time trade logs at a glance.',
      color: kInfo,
    ),
    _TutorialPage(
      icon: Icons.check_circle_outline_rounded,
      title: 'You\'re All Set!',
      body: 'You can replay this tutorial anytime from the Settings page.\n\n'
          'Happy trading!',
      color: Colors.green,
    ),
  ];

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.black87,
      child: Stack(
        children: [
          PageView(
            controller: _ctrl,
            onPageChanged: (i) => setState(() => _page = i),
            children: _pages.map((p) => _buildPage(p)).toList(),
          ),
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            right: 16,
            child: TextButton(
              onPressed: hapt(widget.onDismiss),
              child: const Text('Skip', style: TextStyle(color: Colors.white54, fontSize: 14)),
            ),
          ),
          Positioned(
            bottom: MediaQuery.of(context).padding.bottom + 40,
            left: 0,
            right: 0,
            child: Column(
              children: [
                _buildDots(),
                const SizedBox(height: 24),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 40),
                  child: SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: hapt(() {
                        if (_page < _pages.length - 1) {
                          _ctrl.nextPage(
                            duration: const Duration(milliseconds: 300),
                            curve: Curves.easeInOut,
                          );
                        } else {
                          widget.onDismiss();
                        }
                      }),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: kGold,
                        foregroundColor: Colors.black,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      child: Text(
                        _page < _pages.length - 1 ? 'Next' : 'Get Started',
                        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPage(_TutorialPage page) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: page.color.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: Icon(page.icon, color: page.color, size: 48),
          ),
          const SizedBox(height: 32),
          Text(
            page.title,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            page.body,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 14,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDots() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(_pages.length, (i) {
        return AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          margin: const EdgeInsets.symmetric(horizontal: 4),
          width: _page == i ? 24 : 8,
          height: 8,
          decoration: BoxDecoration(
            color: _page == i ? kGold : Colors.white24,
            borderRadius: BorderRadius.circular(4),
          ),
        );
      }),
    );
  }
}

class _TutorialPage {
  final IconData icon;
  final String title;
  final String body;
  final Color color;
  const _TutorialPage({
    required this.icon,
    required this.title,
    required this.body,
    required this.color,
  });
}