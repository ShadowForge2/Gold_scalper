import 'dart:js' as js;
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
  String _selectedMethod = 'card';
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
          backgroundColor: const Color(0xFFF0F0F0),
          appBar: AppBar(
            title: const Text('Subscription'),
            backgroundColor: const Color(0xFFF0F0F0),
            foregroundColor: Colors.black87,
            elevation: 0.5,
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
              if (_selectedMethod == 'card') ...[
                const SizedBox(height: 12),
                _buildPaymentForm(bp, channel: 'card'),
              ] else if (_selectedMethod == 'bank_transfer') ...[
                const SizedBox(height: 12),
                _buildPaymentForm(bp, channel: 'bank_transfer'),
              ] else if (_selectedMethod == 'cryptomus') ...[
                const SizedBox(height: 12),
                _buildCryptomusForm(bp),
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
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
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
                  Text('Status', style: TextStyle(color: Colors.grey.shade700, fontSize: 11, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text(
                    bp.trialActive
                        ? 'Trial Active'
                        : bp.subscription['subscribed'] == true
                            ? 'Subscribed'
                            : 'Not Started',
                    style: const TextStyle(color: Colors.black87, fontSize: 18, fontWeight: FontWeight.bold),
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
          Container(height: 1, color: Colors.grey.shade200),
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
          Text(value, style: const TextStyle(color: Colors.black87, fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text(label, style: TextStyle(color: Colors.grey.shade600, fontSize: 11)),
        ],
      ),
    );
  }

  Widget _buildProfitCard(BotProvider bp) {
    final amount = bp.unpaidFees > 0 ? bp.unpaidFees : bp.currentMonthFee;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Current 30-Day Period', style: TextStyle(color: Colors.black87, fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          _row('Current Balance', '\$${(bp.state?.balance ?? 0).toStringAsFixed(2)}'),
          const Divider(color: Colors.grey, height: 20),
          _row('Profit', '\$${bp.currentMonthProfit.toStringAsFixed(2)}', valueColor: bp.currentMonthProfit >= 0 ? Colors.green.shade700 : Colors.red.shade700),
          const Divider(color: Colors.grey, height: 20),
          _row('15% Fee Due', '\$${bp.currentMonthFee.toStringAsFixed(2)}', valueColor: kGold),
          if (bp.unpaidFees > 0) ...[
            const Divider(color: Colors.grey, height: 20),
            _row('Total Unpaid Fees', '\$${bp.unpaidFees.toStringAsFixed(2)}', valueColor: Colors.amber.shade700),
          ],
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: kGold.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: kGold.withValues(alpha: 0.12)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Amount to Pay', style: TextStyle(color: Colors.black87, fontSize: 13)),
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
        Text(label, style: const TextStyle(color: Colors.black87, fontSize: 14)),
        Text(value, style: TextStyle(color: valueColor ?? Colors.black87, fontWeight: FontWeight.bold, fontSize: 14)),
      ],
    );
  }

  Widget _buildPeriodsCard(BotProvider bp) {
    final periods = bp.monthlyPeriods;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Billing History', style: TextStyle(color: Colors.black87, fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          if (periods.isEmpty)
            Text('No billing periods yet.', style: TextStyle(color: Colors.grey.shade600, fontSize: 13))
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
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('$start → ${end ?? 'current'}', style: const TextStyle(color: Colors.black87, fontSize: 12, fontWeight: FontWeight.w600)),
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
                const Text('UNPAID', style: TextStyle(color: Colors.orange, fontSize: 10, fontWeight: FontWeight.bold)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMethodCards(BotProvider bp) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Payment Method', style: TextStyle(color: Colors.black87, fontSize: 15, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _methodCard(
                icon: Icons.credit_card_rounded,
                label: 'Card',
                desc: 'Pay with debit\nor credit card',
                selected: _selectedMethod == 'card',
                onTap: () => setState(() => _selectedMethod = 'card'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _methodCard(
                icon: Icons.account_balance_rounded,
                label: 'Bank Transfer',
                desc: 'Pay via bank\ntransfer',
                selected: _selectedMethod == 'bank_transfer',
                onTap: () => setState(() => _selectedMethod = 'bank_transfer'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _methodCard(
                icon: Icons.currency_bitcoin_rounded,
                label: 'Cryptomus',
                desc: 'Pay with USDT\nvia Cryptomus',
                selected: _selectedMethod == 'cryptomus',
                onTap: () => setState(() => _selectedMethod = 'cryptomus'),
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
          color: selected ? kGold.withValues(alpha: 0.1) : Colors.grey.shade100,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: selected ? kGold : Colors.grey.shade300,
            width: selected ? 1.5 : 1,
          ),
        ),
        child: Column(
          children: [
            Icon(icon, color: selected ? kGold : Colors.grey, size: 28),
            const SizedBox(height: 8),
            Text(label, style: TextStyle(color: selected ? kGold : Colors.black54, fontWeight: FontWeight.bold, fontSize: 14)),
            const SizedBox(height: 4),
            Text(desc, textAlign: TextAlign.center, style: TextStyle(color: Colors.grey.shade700, fontSize: 11, height: 1.3)),
          ],
        ),
      ),
    );
  }

  Widget _buildPaymentForm(BotProvider bp, {required String channel}) {
    final amount = bp.unpaidFees > 0 ? bp.unpaidFees : bp.currentMonthFee;
    final isCard = channel == 'card';
    final icon = isCard ? Icons.credit_card_rounded : Icons.account_balance_rounded;
    final title = isCard ? 'Card Payment' : 'Bank Transfer';
    final buttonLabel = isCard
        ? 'Pay \$${amount.toStringAsFixed(2)} with Card'
        : 'Pay \$${amount.toStringAsFixed(2)} via Bank Transfer';
    final channels = isCard ? ['card'] : ['bank_transfer'];
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: kGold.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: kGold, size: 18),
              const SizedBox(width: 8),
              Text(title, style: const TextStyle(color: Colors.black87, fontSize: 15, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _emailCtrl,
            style: const TextStyle(color: Colors.black87),
            decoration: InputDecoration(
              labelText: 'Email Address',
              labelStyle: TextStyle(color: Colors.grey.shade700),
              hintText: 'you@email.com',
              hintStyle: TextStyle(color: Colors.grey.shade500),
              filled: true,
              fillColor: Colors.grey.shade100,
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
              color: Colors.grey.shade100,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Amount', style: TextStyle(color: Colors.black87, fontSize: 13)),
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
                final result = await bp.initializePayment(email, channels: channels);
                if (result != null && context.mounted) {
                  final accessCode = result['access_code'] as String?;
                  if (accessCode != null) {
                    js.context.callMethod('paystackInline', [
                      accessCode,
                      (ref) {
                        bp.verifyPayment(ref as String);
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Payment successful!')),
                          );
                        }
                      },
                      () {
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Payment cancelled')),
                          );
                        }
                      },
                    ]);
                  } else {
                    final url = result['authorization_url'] as String?;
                    if (url != null) {
                      js.context.callMethod('open', [url, '_blank']);
                    }
                  }
                }
              },
              icon: const Icon(Icons.payment_rounded),
              label: Text(buttonLabel),
              style: ElevatedButton.styleFrom(
                backgroundColor: kGold,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCryptomusForm(BotProvider bp) {
    final amount = bp.unpaidFees > 0 ? bp.unpaidFees : bp.currentMonthFee;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.currency_bitcoin_rounded, color: kGold, size: 18),
              const SizedBox(width: 8),
              const Text('Cryptomus Crypto Payment', style: TextStyle(color: Colors.black87, fontSize: 15, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          const Text(
            'Pay with USDT, BTC, ETH or other crypto via Cryptomus.',
            style: TextStyle(color: Colors.black87, fontSize: 13),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _emailCtrl,
            style: const TextStyle(color: Colors.black87),
            decoration: InputDecoration(
              labelText: 'Email (optional)',
              labelStyle: TextStyle(color: Colors.grey.shade700),
              hintText: 'you@email.com',
              hintStyle: TextStyle(color: Colors.grey.shade500),
              filled: true,
              fillColor: Colors.grey.shade100,
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
              color: Colors.grey.shade100,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Amount', style: TextStyle(color: Colors.black87, fontSize: 13)),
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
                final result = await bp.initCryptomusPayment(amount, email);
                if (result != null && context.mounted) {
                  final payUrl = result['payment_url'] as String?;
                  if (payUrl != null) {
                    js.context.callMethod('open', [payUrl, '_blank']);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Cryptomus payment page opened')),
                    );
                  }
                } else if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Failed to create payment')),
                  );
                }
              },
              icon: const Icon(Icons.payment_rounded),
              label: Text('Pay \$${amount.toStringAsFixed(2)} with Crypto'),
              style: ElevatedButton.styleFrom(
                backgroundColor: kGold,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: kGold.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: kGold.withValues(alpha: 0.1)),
            ),
            child: const Row(
              children: [
                Icon(Icons.info_outline_rounded, color: kGold, size: 14),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'After payment, your subscription will be activated automatically.',
                    style: TextStyle(color: Colors.black87, fontSize: 12, height: 1.3),
                  ),
                ),
              ],
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
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.currency_bitcoin_rounded, color: kGold, size: 18),
              const SizedBox(width: 8),
              const Text('Crypto Payment', style: TextStyle(color: Colors.black87, fontSize: 15, fontWeight: FontWeight.bold)),
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
              color: kGold.withValues(alpha: 0.1),
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
                    style: TextStyle(color: Colors.black87, fontSize: 12, height: 1.3),
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
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(network, style: const TextStyle(color: Colors.black87, fontSize: 12, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text(address, style: TextStyle(color: Colors.grey.shade700, fontSize: 11)),
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
