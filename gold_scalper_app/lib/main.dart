import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'providers/device_provider.dart';
import 'providers/bot_provider.dart';
import 'screens/home_screen.dart';
import 'screens/welcome/welcome_screen.dart';
import 'services/notification_service.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  NotificationService.instance.init();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ),
  );
  FlutterError.onError = (details) {
    FlutterError.presentError(details);
    debugPrint('FATAL: ${details.exception}\n${details.stack}');
  };
  runZonedGuarded(
    () => runApp(const GoldScalperApp()),
    (error, stack) => debugPrint('UNCAUGHT: $error\n$stack'),
  );
}

class GoldScalperApp extends StatelessWidget {
  const GoldScalperApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => DeviceProvider()..init()),
        ChangeNotifierProxyProvider<DeviceProvider, BotProvider>(
          create: (ctx) => BotProvider(ctx.read<DeviceProvider>()),
          update: (ctx, device, prev) => prev!,
        ),
      ],
      child: MaterialApp(
        title: 'Gold Scalper',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          scaffoldBackgroundColor: kDarkBg,
          colorScheme: const ColorScheme.dark(
            primary: kGold,
            secondary: kGold,
            surface: kDarkSurface,
          ),
          appBarTheme: const AppBarTheme(
            backgroundColor: kDarkSurface,
            foregroundColor: Colors.white,
            elevation: 0,
            centerTitle: true,
          ),
          cardTheme: CardThemeData(
            color: kDarkCard,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16)),
            elevation: 0,
          ),
          textTheme: const TextTheme(
            labelSmall: TextStyle(
                color: Color(0xFF888899), fontSize: 11, letterSpacing: 0.5),
            bodySmall: TextStyle(color: Color(0xFF888899), fontSize: 13),
            bodyMedium: TextStyle(color: Color(0xFFCCCCDD), fontSize: 14),
            bodyLarge: TextStyle(color: Colors.white, fontSize: 16),
            titleMedium: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold),
            titleLarge: TextStyle(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.bold),
          ),
          useMaterial3: true,
        ),
        home: const AppEntry(),
      ),
    );
  }
}

class AppEntry extends StatelessWidget {
  const AppEntry({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<DeviceProvider>(
      builder: (context, device, _) {
        if (device.loading) {
          return const Scaffold(
            backgroundColor: kDarkBg,
            body: Center(
              child: CircularProgressIndicator(color: kGold),
            ),
          );
        }
        if (device.firstLaunch) {
          return const WelcomeScreen();
        }
        return const HomeScreen();
      },
    );
  }
}
