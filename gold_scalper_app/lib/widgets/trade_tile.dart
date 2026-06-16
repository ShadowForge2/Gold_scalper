import 'package:flutter/material.dart';
import '../models/trade.dart';
import '../theme.dart';

class TradeTile extends StatelessWidget {
  final Trade trade;

  const TradeTile({super.key, required this.trade});

  @override
  Widget build(BuildContext context) {
    final pnlColor = trade.pnl >= 0 ? Colors.green : Colors.red;
    final dirColor = trade.direction == 'BUY' ? Colors.green : Colors.red;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.3)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: dirColor.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    trade.direction,
                    style: TextStyle(color: dirColor, fontWeight: FontWeight.bold, fontSize: 12),
                  ),
                ),
                const SizedBox(width: 8),
                Text(trade.id, style: const TextStyle(color: kTextSecondary, fontSize: 11)),
                const Spacer(),
                Text(
                  '\$${trade.pnl.toStringAsFixed(2)}',
                  style: TextStyle(color: pnlColor, fontWeight: FontWeight.bold, fontSize: 15),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                _info('Entry', '\$${trade.entryPrice.toStringAsFixed(2)}'),
                const SizedBox(width: 16),
                _info('Exit', trade.exitPrice != null ? '\$${trade.exitPrice!.toStringAsFixed(2)}' : 'Open'),
                const SizedBox(width: 16),
                _info('Score', trade.score.toStringAsFixed(3)),
                const Spacer(),
                Text(trade.exitReason, style: const TextStyle(color: kTextSecondary, fontSize: 11)),
              ],
            ),
            const SizedBox(height: 4),
            Row(
              children: [
                _info('Time', '${trade.entryTime.hour.toString().padLeft(2, '0')}:${trade.entryTime.minute.toString().padLeft(2, '0')}'),
                const SizedBox(width: 16),
                _info('Lot', trade.lot.toStringAsFixed(3)),
                const SizedBox(width: 16),
                _info('Bal', '\$${trade.balance.toStringAsFixed(2)}'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _info(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 10)),
        Text(value, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }
}
