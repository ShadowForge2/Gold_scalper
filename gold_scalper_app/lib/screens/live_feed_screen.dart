import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/bot_provider.dart';
import '../widgets/trade_tile.dart';
import '../theme.dart';

class LiveFeedScreen extends StatelessWidget {
  const LiveFeedScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<BotProvider>(
      builder: (context, bp, _) {
        if (bp.loading) {
          return const Center(child: CircularProgressIndicator(color: kGold));
        }

        if (bp.recentTrades.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.inbox, size: 64, color: Colors.grey[700]),
                const SizedBox(height: 16),
                Text('No trades yet', style: TextStyle(color: Colors.grey[500], fontSize: 18)),
              ],
            ),
          );
        }

        return ListView.builder(
          padding: const EdgeInsets.only(top: 8, bottom: 16),
          itemCount: bp.recentTrades.length,
          itemBuilder: (_, i) => TradeTile(trade: bp.recentTrades[i]),
        );
      },
    );
  }
}
