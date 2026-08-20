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
        if (bp.loading || !bp.dataLoaded) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(
                  width: 28,
                  height: 28,
                  child: CircularProgressIndicator(strokeWidth: 2, color: kGold),
                ),
                const SizedBox(height: 12),
                const Text('Loading trades...', style: TextStyle(color: Colors.white54, fontSize: 13)),
              ],
            ),
          );
        }

        if (bp.recentTrades.isEmpty) {
          return RefreshIndicator(
            onRefresh: () async => bp.refresh(),
            child: ListView(
              children: [
                SizedBox(height: MediaQuery.of(context).size.height * 0.2),
                Icon(Icons.inbox, size: 64, color: Colors.grey[700]),
                const SizedBox(height: 16),
                Center(child: Text('No trades yet', style: TextStyle(color: Colors.grey[500], fontSize: 18))),
              ],
            ),
          );
        }

        return RefreshIndicator(
          onRefresh: () async => bp.refresh(),
          child: ListView.builder(
            padding: const EdgeInsets.only(top: 8, bottom: 16),
            itemCount: bp.recentTrades.length,
            itemBuilder: (_, i) => TradeTile(trade: bp.recentTrades[i]),
          ),
        );
      },
    );
  }
}
