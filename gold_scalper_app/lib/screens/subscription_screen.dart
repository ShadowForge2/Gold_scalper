import 'dart:html' as html;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../providers/bot_provider.dart';
import '../theme.dart';

class SubscriptionScreen extends StatefulWidget {
  const SubscriptionScreen({super.key});

  @override
  State<SubscriptionScreen> createState() => _SubscriptionScreenState();
}

class _SubscriptionScreenState extends State<SubscriptionScreen> {
  String _selectedMethod = 'paystack';
  final _emailCtrl = TextEditingController();
  final _refCtrl = TextEditingController();

  @override
  void dispose() {
    _emailCtrl.dispose();
    _refCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<BotProvider>(
      builder: (context, bp, _) {
        return Scaffold(
          backgroundColor: kDarkBg,
          appBar: AppBar(
            title: const Text('Subscription'),
            backgroundColor: kDarkSurface,
          ),
          body: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _buildStatusCard(bp),
              const SizedBox(height: 16),
              _buildProfitCard(bp),
              const SizedBox(height: 16),
              _buildPeriodsCard(bp),
              const SizedBox(height: 16),
              _buildMethodCards(bp),
              if (_selectedMethod == 'paystack') ...[
                const SizedBox(height: 12),
                _buildPaystackForm(bp),
              ] else ...[
                const SizedBox(height: 12),
                _buildCryptoInfo(),
              ],
              const SizedBox(height: 24),
            ],
          ),
        );
      },
    );
  }

  Widget _buildStatusCard(BotProvider bp) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0F172A), kDarkSurface],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: kGold.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.credit_card_rounded, color: kGold, size: 24),
              ),
              const SizedBox(width: 14),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Status', style: TextStyle(color: kTextSecondary, fontSize: 11, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text(
                    bp.trialActive
                        ? 'Trial Active'
                        : bp.subscription['subscribed'] == true
                            ? 'Subscribed'
                            : 'Not Started',
                    style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
              const Spacer(),
              if (bp.trialActive)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: kGold.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: kGold.withValues(alpha: 0.2)),
                  ),
                  child: Text(
                    '${bp.daysRemaining}d left',
                    style: const TextStyle(color: kGold, fontWeight: FontWeight.bold, fontSize: 13),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 20),
          Container(height: 1, color: Colors.white.withValues(alpha: 0.04)),
          const SizedBox(height: 20),
          Row(
            children: [
              _statItem('Trial', bp.trialActive ? 'Active' : 'Expired'),
              _statItem('Subscribed', bp.subscription['subscribed'] == true ? 'Yes' : 'No'),
              _statItem('Can Trade', bp.canTrade ? 'Yes' : 'No'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _statItem(String label, String value) {
    return Expanded(
      child: Column(
        children: [
          Text(value, style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 10)),
        ],
      ),
    );
  }

  Widget _buildProfitCard(BotProvider bp) {
    final amount = bp.unpaidFees > 0 ? bp.unpaidFees : bp.currentMonthFee;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.04)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Current 30-Day Period', style: TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          _row('Starting Balance', '\$${(1000).toStringAsFixed(2)}'),
          const Divider(color: kDarkBorder, height: 20),
          _row('Current Balance', '\$${(bp.state?.balance ?? 0).toStringAsFixed(2)}'),
          const Divider(color: kDarkBorder, height: 20),
          _row('Profit', '\$${bp.currentMonthProfit.toStringAsFixed(2)}', valueColor: bp.currentMonthProfit >= 0 ? Colors.green : Colors.red),
          const Divider(color: kDarkBorder, height: 20),
          _row('15% Fee Due', '\$${bp.currentMonthFee.toStringAsFixed(2)}', valueColor: kGold),
          if (bp.unpaidFees > 0) ...[
            const Divider(color: kDarkBorder, height: 20),
            _row('Total Unpaid Fees', '\$${bp.unpaidFees.toStringAsFixed(2)}', valueColor: Colors.amberAccent),
          ],
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: kGold.withValues(alpha: 0.06),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: kGold.withValues(alpha: 0.12)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Amount to Pay', style: TextStyle(color: kTextSecondary, fontSize: 13)),
                Text('\$${amount.toStringAsFixed(2)}', style: const TextStyle(color: kGold, fontSize: 20, fontWeight: FontWeight.bold)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _row(String label, String value, {Color? valueColor}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 14)),
        Text(value, style: TextStyle(color: valueColor ?? Colors.white70, fontWeight: FontWeight.bold, fontSize: 14)),
      ],
    );
  }

  Widget _buildPeriodsCard(BotProvider bp) {
    final periods = bp.monthlyPeriods;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.04)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Billing History', style: TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          if (periods.isEmpty)
            const Text('No billing periods yet.', style: TextStyle(color: kTextSecondary, fontSize: 13))
          else
            ...periods.map((p) => _periodTile(p)),
        ],
      ),
    );
  }

  Widget _periodTile(Map<String, dynamic> p) {
    final start = p['period_start']?.toString().substring(0, 10) ?? '--';
    final end = p['period_end']?.toString().substring(0, 10);
    final profit = (p['cumulative_profit'] ?? 0).toDouble();
    final fee = (p['fee_15pct'] ?? 0).toDouble();
    final paid = p['fee_paid'] == true;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: kDarkBg,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.2)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('$start → ${end ?? 'current'}', style: const TextStyle(color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text('Profit: \$${profit.toStringAsFixed(2)}', style: TextStyle(color: profit >= 0 ? Colors.green : Colors.red, fontSize: 12)),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text('Fee: \$${fee.toStringAsFixed(2)}', style: const TextStyle(color: kGold, fontSize: 12, fontWeight: FontWeight.bold)),
              if (paid)
                const Text('PAID', style: TextStyle(color: Colors.green, fontSize: 10, fontWeight: FontWeight.bold))
              else if (fee > 0 && end != null)
                const Text('UNPAID', style: TextStyle(color: Colors.amberAccent, fontSize: 10, fontWeight: FontWeight.bold)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMethodCards(BotProvider bp) {
    final amount = bp.unpaidFees > 0 ? bp.unpaidFees : bp.currentMonthFee;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Payment Method', style: TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _methodCard(
                icon: Icons.payments_rounded,
                label: 'Paystack',
                desc: 'Pay with card, bank\nor USSD',
                selected: _selectedMethod == 'paystack',
                onTap: () => setState(() => _selectedMethod = 'paystack'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _methodCard(
                icon: Icons.currency_bitcoin_rounded,
                label: 'Crypto',
                desc: 'Pay with USDT,\nBTC or ETH',
                selected: _selectedMethod == 'crypto',
                onTap: () => setState(() => _selectedMethod = 'crypto'),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _methodCard({
    required IconData icon,
    required String label,
    required String desc,
    required bool selected,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: selected ? kGold.withValues(alpha: 0.1) : kDarkCard,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: selected ? kGold : kDarkBorder.withValues(alpha: 0.3),
            width: selected ? 1.5 : 1,
          ),
        ),
        child: Column(
          children: [
            Icon(icon, color: selected ? kGold : kTextSecondary, size: 28),
            const SizedBox(height: 8),
            Text(label, style: TextStyle(color: selected ? kGold : Colors.white70, fontWeight: FontWeight.bold, fontSize: 14)),
            const SizedBox(height: 4),
            Text(desc, textAlign: TextAlign.center, style: const TextStyle(color: kTextSecondary, fontSize: 10, height: 1.3)),
          ],
        ),
      ),
    );
  }

  Widget _buildPaystackForm(BotProvider bp) {
    final amount = bp.unpaidFees > 0 ? bp.unpaidFees : bp.currentMonthFee;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: kGold.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.payments_rounded, color: kGold, size: 18),
              const SizedBox(width: 8),
              const Text('Paystack Payment', style: TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _emailCtrl,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              labelText: 'Email Address',
              labelStyle: const TextStyle(color: kTextSecondary),
              hintText: 'you@email.com',
              hintStyle: TextStyle(color: kTextSecondary.withValues(alpha: 0.5)),
              filled: true,
              fillColor: kDarkBg,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: kGold),
              ),
            ),
            keyboardType: TextInputType.emailAddress,
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: kDarkBg,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Amount', style: TextStyle(color: kTextSecondary, fontSize: 13)),
                Text('\$${amount.toStringAsFixed(2)}', style: const TextStyle(color: kGold, fontSize: 16, fontWeight: FontWeight.bold)),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () async {
                final email = _emailCtrl.text.trim();
                if (email.isEmpty) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Enter your email')),
                  );
                  return;
                }
                final url = await bp.initializePayment(email);
                if (url != null && context.mounted) {
                  html.window.open(url, '_blank');
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Paystack opened in new tab')),
                  );
                }
              },
              icon: const Icon(Icons.payment_rounded),
              label: Text('Pay \$${amount.toStringAsFixed(2)} with Paystack'),
              style: ElevatedButton.styleFrom(
                backgroundColor: kGold,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Container(height: 1, color: kDarkBorder.withValues(alpha: 0.3)),
          const SizedBox(height: 16),
          const Text('Verify Payment', style: TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          TextField(
            controller: _refCtrl,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              hintText: 'Paste reference from Paystack',
              hintStyle: TextStyle(color: kTextSecondary.withValues(alpha: 0.5)),
              filled: true,
              fillColor: kDarkBg,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: kGold),
              ),
            ),
          ),
          const SizedBox(height: 8),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () async {
                final ref = _refCtrl.text.trim();
                if (ref.isEmpty) return;
                final ok = await bp.verifyPayment(ref);
                if (ok && context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Payment verified!')),
                  );
                  _refCtrl.clear();
                }
              },
              icon: const Icon(Icons.verified_rounded, size: 18),
              label: const Text('Verify Payment'),
              style: OutlinedButton.styleFrom(
                foregroundColor: kGold,
                side: const BorderSide(color: kGold),
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCryptoInfo() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.04)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.currency_bitcoin_rounded, color: kGold, size: 18),
              const SizedBox(width: 8),
              const Text('Crypto Payment', style: TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          _cryptoWallet('USDT (TRC-20)', 'TXx5RqaPzG8o1z9yL3...'),
          const SizedBox(height: 12),
          _cryptoWallet('BTC', 'bc1q5v8z2r9x...'),
          const SizedBox(height: 12),
          _cryptoWallet('ETH (ERC-20)', '0x742d35Cc6634C0...'),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: kGold.withValues(alpha: 0.06),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: kGold.withValues(alpha: 0.1)),
            ),
            child: const Row(
              children: [
                Icon(Icons.info_outline_rounded, color: kGold, size: 14),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'After sending crypto, contact support with transaction ID for verification.',
                    style: TextStyle(color: kTextSecondary, fontSize: 11, height: 1.3),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _cryptoWallet(String network, String address) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: kDarkBg,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.15)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(network, style: const TextStyle(color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text(address, style: const TextStyle(color: kTextSecondary, fontSize: 11)),
              ],
            ),
          ),
          GestureDetector(
            onTap: () {
              Clipboard.setData(ClipboardData(text: address));
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('$network address copied')),
              );
            },
            child: Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: kGold.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(6),
              ),
              child: const Icon(Icons.copy_rounded, color: kGold, size: 14),
            ),
          ),
        ],
      ),
    );
  }
}
