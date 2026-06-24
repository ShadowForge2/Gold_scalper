import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/bot_provider.dart';
import '../providers/device_provider.dart';
import '../widgets/fade_in_scale.dart';
import '../widgets/onboarding_tutorial.dart';
import '../widgets/ui/haptic.dart';
import '../widgets/notification_bell.dart';
import '../theme.dart';
import 'dashboard_screen.dart';
import 'live_feed_screen.dart';
import 'performance_screen.dart';
import 'controls_screen.dart';
import 'settings_screen.dart';
import 'subscription_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;
  late PageController _pageController;

  final List<Widget> _screens = const [
    DashboardScreen(),
    LiveFeedScreen(),
    PerformanceScreen(),
    ControlsScreen(),
    SettingsScreen(),
  ];

  @override
  void initState() {
    super.initState();
    _pageController = PageController(initialPage: _currentIndex);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<BotProvider>().init();
    });
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bp = context.watch<BotProvider>();
    final inTrade = bp.state?.state == 'IN_TRADE';

    final navTab = bp.navigateToTab;
    if (navTab != null && navTab != _currentIndex) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        bp.clearNavigation();
        if (mounted) {
          setState(() => _currentIndex = navTab);
          _pageController.animateToPage(
            navTab,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
          );
        }
      });
    }

    if (bp.navigateToSubscription) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        bp.clearSubscriptionNavigation();
        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const SubscriptionScreen()),
          );
        }
      });
    }

    final device = context.watch<DeviceProvider>();
    final showTutorial = device.firstLaunch && !device.tutorialSeen && !device.loading;

    return Scaffold(
      backgroundColor: kDarkBg,
      appBar: AppBar(
        leading: const NotificationBell(),
        title: AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          child: Text(
            _titles[_currentIndex],
            key: ValueKey(_currentIndex),
            style: const TextStyle(fontSize: 17, fontWeight: FontWeight.bold, letterSpacing: -0.3),
          ),
        ),
        actions: [
          if (inTrade)
            FadeInScale(
              child: Container(
                margin: const EdgeInsets.only(right: 8),
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: Colors.green.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.green.withValues(alpha: 0.2)),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SizedBox(
                      width: 7, height: 7,
                      child: CircularProgressIndicator(strokeWidth: 1.5, color: Colors.green),
                    ),
                    SizedBox(width: 5),
                    Text('TRADING', style: TextStyle(color: Colors.green, fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 0.5)),
                  ],
                ),
              ),
            ),
          IconButton(
            icon: Icon(Icons.refresh_rounded, color: Colors.grey[500], size: 20),
            onPressed: hapt(() => bp.init()),
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: Stack(
        children: [
          PageView(
            controller: _pageController,
            physics: const NeverScrollableScrollPhysics(),
            children: _screens,
          ),
          if (showTutorial)
            OnboardingTutorial(
              onDismiss: () => device.markTutorialSeen(),
            ),
        ],
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildBottomNav() {
    return Container(
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: kDarkBorder.withValues(alpha: 0.3))),
      ),
      child: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: haptInt((i) {
          setState(() => _currentIndex = i);
          _pageController.animateToPage(
            i,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
          );
        }),
        backgroundColor: kDarkSurface,
        selectedItemColor: kGold,
        unselectedItemColor: Colors.grey[600],
        type: BottomNavigationBarType.fixed,
        selectedFontSize: 10,
        unselectedFontSize: 10,
        selectedLabelStyle: const TextStyle(letterSpacing: 0.3, fontWeight: FontWeight.w600),
        unselectedLabelStyle: const TextStyle(letterSpacing: 0.3),
        items: const [
          BottomNavigationBarItem(
            icon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.dashboard_rounded, size: 20)),
            activeIcon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.dashboard_rounded, size: 22)),
            label: 'Dashboard',
          ),
          BottomNavigationBarItem(
            icon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.timeline_rounded, size: 20)),
            activeIcon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.timeline_rounded, size: 22)),
            label: 'Live Feed',
          ),
          BottomNavigationBarItem(
            icon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.bar_chart_rounded, size: 20)),
            activeIcon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.bar_chart_rounded, size: 22)),
            label: 'Performance',
          ),
          BottomNavigationBarItem(
            icon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.tune_rounded, size: 20)),
            activeIcon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.tune_rounded, size: 22)),
            label: 'Controls',
          ),
          BottomNavigationBarItem(
            icon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.settings_rounded, size: 20)),
            activeIcon: Padding(padding: EdgeInsets.only(bottom: 4), child: Icon(Icons.settings_rounded, size: 22)),
            label: 'Settings',
          ),
        ],
      ),
    );
  }

  static const List<String> _titles = [
    'Dashboard',
    'Live Feed',
    'Performance',
    'Controls',
    'Settings',
  ];
}
