import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/device_provider.dart';
import '../providers/bot_provider.dart';
import '../widgets/status_indicator.dart';
import '../widgets/fade_in_scale.dart';
import '../theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> with SingleTickerProviderStateMixin {
  final _apiKeyCtrl = TextEditingController();
  final _identifierCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  bool _isDemo = true;
  late AnimationController _pulseCtrl;
  late Animation<double> _pulseAnim;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 600));
    _pulseAnim = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  @override
  void dispose() {
    _apiKeyCtrl.dispose();
    _identifierCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  String _formatDuration(Duration? d) {
    if (d == null || d == Duration.zero) return '';
    final h = d.inHours;
    final m = d.inMinutes.remainder(60);
    return '${h}h ${m}m';
  }

  Future<void> _save() async {
    final apiKey = _apiKeyCtrl.text.trim();
    final identifier = _identifierCtrl.text.trim();
    final password = _passwordCtrl.text.trim();
    if (apiKey.isEmpty || identifier.isEmpty || password.isEmpty) {
      _snack('All fields are required');
      return;
    }

    final bp = context.read<BotProvider>();
    final ok = await bp.addAccount(apiKey, identifier, password, _isDemo);
    if (ok && mounted) {
      _apiKeyCtrl.clear();
      _identifierCtrl.clear();
      _passwordCtrl.clear();
      _snack('Account saved');
    }
  }

  void _snack(String msg) {
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    final bp = context.watch<BotProvider>();
    final device = context.watch<DeviceProvider>();

    if (bp.highlightCredentials) {
      _pulseCtrl.repeat(reverse: true);
      Future.delayed(const Duration(seconds: 3), () {
        if (mounted) {
          bp.clearHighlight();
          _pulseCtrl.stop();
          _pulseCtrl.reset();
        }
      });
    }

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        FadeInScale(
          child: _buildSection('Device', [
            _infoTile('Device ID', device.deviceId ?? '--'),
          ]),
        ),
        const SizedBox(height: 16),
        FadeInScale(
          delay: const Duration(milliseconds: 100),
          child: _buildCredentialsSection(bp, device),
            _field('API Key', _apiKeyCtrl),
            const SizedBox(height: 8),
            _field('Identifier (Email)', _identifierCtrl,
                keyboardType: TextInputType.emailAddress),
            const SizedBox(height: 8),
            _field('Password', _passwordCtrl, obscure: true),
            const SizedBox(height: 12),
            _demoToggle(),
            if (!device.canEditCredentials) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: device.accountTied
                      ? Colors.red.withValues(alpha: 0.1)
                      : kGold.withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: device.accountTied
                        ? Colors.red.withValues(alpha: 0.3)
                        : kGold.withValues(alpha: 0.2),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      device.accountTied
                          ? Icons.lock_rounded
                          : Icons.timer_outlined,
                      size: 14,
                      color: device.accountTied ? Colors.redAccent : kGold,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        device.accountTied
                            ? 'This account is tied to this device and cannot be changed.'
                            : 'Credentials can only be edited once every 24 hours.',
                        style: TextStyle(
                          color:
                              device.accountTied ? Colors.redAccent : kGold,
                          fontSize: 11,
                          height: 1.3,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: device.canEditCredentials ? _save : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor:
                      device.canEditCredentials ? kGold : kDarkBorder,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10)),
                ),
                child: Text(
                  device.accountTied
                      ? 'Account Locked'
                      : device.credentialsSavedAt != null && !device.canEditCredentials
                          ? 'Cooldown — ${_formatDuration(device.cooldownRemaining)}'
                          : 'Save Credentials',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ]),
        ),
        const SizedBox(height: 16),
        FadeInScale(
          delay: const Duration(milliseconds: 200),
          child: _buildSection('Subscription', [
            if (bp.trialActive)
              _infoTile('Trial Active', '${bp.daysRemaining} day(s) left')
            else if (bp.subscription['subscribed'] == true)
              _infoTile('Subscribed', 'Active')
            else
              _infoTile('Status', 'Not started'),
            _infoTile('Monthly Profit', '\$${bp.currentMonthProfit.toStringAsFixed(2)}'),
            _infoTile('15% Fee Due', '\$${bp.currentMonthFee.toStringAsFixed(2)}'),
            if (bp.unpaidFees > 0)
              _infoTile('Unpaid Fees (Total)', '\$${bp.unpaidFees.toStringAsFixed(2)}', 
                  valueColor: Colors.amberAccent),
            if (bp.dueAmount > 0)
              _infoTile('Due Amount', '\$${bp.dueAmount.toStringAsFixed(2)}',
                  valueColor: Colors.redAccent),
          ]),
        ),
        const SizedBox(height: 16),
        FadeInScale(
          delay: const Duration(milliseconds: 300),
          child: _buildSection('Info', [
            _infoTile('Broker', 'Capital.com'),
            _infoTile('Symbol', 'XAUUSD'),
            _infoTile('App', 'Gold Scalper v2.0'),
          ]),
        ),
        const SizedBox(height: 16),
        FadeInScale(
          delay: const Duration(milliseconds: 400),
          child: _buildFooter(),
        ),
      ],
    );
  }

  Widget _buildCredentialsSection(BotProvider bp, DeviceProvider device) {
    return AnimatedBuilder(
      animation: _pulseAnim,
      builder: (context, child) {
        final pulse = _pulseCtrl.isAnimating ? _pulseAnim.value : 0.0;
        final glow = kGold.withValues(alpha: (1.0 - pulse) * 0.4);
        return Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            boxShadow: [
              if (_pulseCtrl.isAnimating)
                BoxShadow(color: glow, blurRadius: 12 + pulse * 8, spreadRadius: pulse * 2),
            ],
          ),
          child: child,
        );
      },
      child: _buildSection('Capital.com Credentials', [
        _apiHelpTile(),
        const SizedBox(height: 8),
        _field('API Key', _apiKeyCtrl),
        const SizedBox(height: 8),
        _field('Identifier (Email)', _identifierCtrl,
            keyboardType: TextInputType.emailAddress),
        const SizedBox(height: 8),
        _field('Password', _passwordCtrl, obscure: true),
        const SizedBox(height: 12),
        _demoToggle(),
        if (!device.canEditCredentials) ...[
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: device.accountTied
                  ? Colors.red.withValues(alpha: 0.1)
                  : kGold.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: device.accountTied
                    ? Colors.red.withValues(alpha: 0.3)
                    : kGold.withValues(alpha: 0.2),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  device.accountTied
                      ? Icons.lock_rounded
                      : Icons.timer_outlined,
                  size: 14,
                  color: device.accountTied ? Colors.redAccent : kGold,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    device.accountTied
                        ? 'This account is tied to this device and cannot be changed.'
                        : 'Credentials can only be edited once every 24 hours.',
                    style: TextStyle(
                      color: device.accountTied ? Colors.redAccent : kGold,
                      fontSize: 11,
                      height: 1.3,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: device.canEditCredentials ? _save : null,
            style: ElevatedButton.styleFrom(
              backgroundColor: device.canEditCredentials ? kGold : kDarkBorder,
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            child: Text(
              device.accountTied
                  ? 'Account Locked'
                  : device.credentialsSavedAt != null && !device.canEditCredentials
                      ? 'Cooldown — ${_formatDuration(device.cooldownRemaining)}'
                      : 'Save Credentials',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
        ),
      ]),
    );
  }

  Widget _buildSection(String title, List<Widget> children) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  letterSpacing: -0.2)),
          const SizedBox(height: 8),
          ...children,
        ],
      ),
    );
  }

  Widget _buildFooter() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.3)),
      ),
      child: const Column(
        children: [
          StatusIndicator(active: true, label: 'Ready', size: 14),
          SizedBox(height: 12),
          Text('Gold Scalper v2.0',
              style: TextStyle(color: kTextSecondary, fontSize: 13)),
          Text('Multi-User | Capital.com',
              style: TextStyle(color: kTextSecondary, fontSize: 11)),
          SizedBox(height: 8),
          Text('Developer: Agni Kai',
              style: TextStyle(color: kTextSecondary, fontSize: 11)),
          Text('Company: Fire Star LTD',
              style: TextStyle(color: kTextSecondary, fontSize: 11)),
        ],
      ),
    );
  }

  void _switchMode(bool isDemo) {
    if (isDemo == _isDemo) return;
    setState(() {
      _isDemo = isDemo;
      _apiKeyCtrl.clear();
      _identifierCtrl.clear();
      _passwordCtrl.clear();
    });
    _snack(isDemo
        ? 'Input your demo credentials only'
        : 'Input your live credentials only');
  }

  Widget _demoToggle() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Text('Account Type',
              style: TextStyle(color: kTextSecondary, fontSize: 14)),
          Container(
            decoration: BoxDecoration(
              color: kDarkBg,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: kDarkBorder),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                GestureDetector(
                  onTap: () => _switchMode(true),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 6),
                    decoration: BoxDecoration(
                      color: _isDemo ? kGold : Colors.transparent,
                      borderRadius: const BorderRadius.horizontal(
                          left: Radius.circular(7)),
                    ),
                    child: Text(
                      'Demo',
                      style: TextStyle(
                        color: _isDemo ? Colors.black : kTextSecondary,
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ),
                GestureDetector(
                  onTap: () => _switchMode(false),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 6),
                    decoration: BoxDecoration(
                      color: !_isDemo ? kGold : Colors.transparent,
                      borderRadius: const BorderRadius.horizontal(
                          right: Radius.circular(7)),
                    ),
                    child: Text(
                      'Live',
                      style: TextStyle(
                        color: !_isDemo ? Colors.black : kTextSecondary,
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
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

  Widget _field(String label, TextEditingController ctrl,
      {bool obscure = false, TextInputType? keyboardType}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: const TextStyle(color: kTextSecondary, fontSize: 13)),
        const SizedBox(height: 4),
        TextField(
          controller: ctrl,
          obscureText: obscure,
          keyboardType: keyboardType,
          style: const TextStyle(color: Colors.white, fontSize: 14),
          decoration: InputDecoration(
            isDense: true,
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            filled: true,
            fillColor: kDarkBg,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: const BorderSide(color: kDarkBorder),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: const BorderSide(color: kDarkBorder),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: const BorderSide(color: kGold),
            ),
          ),
        ),
      ],
    );
  }

  void _showApiHelp() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1A1A1A),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Row(
          children: [
            Icon(Icons.help_outline_rounded, color: kGold, size: 22),
            const SizedBox(width: 10),
            const Text('Get Your API Key',
                style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _helpStep('1', 'Open Capital.com in Chrome.'),
            const SizedBox(height: 10),
            _helpStep('2', 'Log in to your account.'),
            const SizedBox(height: 10),
            _helpStep('3', 'Go to Settings → API integrations.'),
            const SizedBox(height: 10),
            _helpStep('4', 'Tap Generate API key.'),
            const SizedBox(height: 10),
            _helpStep('5', 'Enable 2FA if asked.'),
            const SizedBox(height: 10),
            _helpStep('6', 'Copy and save the key right away.'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Got it', style: TextStyle(color: kGold, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  Widget _helpStep(String num, String text) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 22,
          height: 22,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: kGold.withValues(alpha: 0.2),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Text(num,
              style: const TextStyle(color: kGold, fontSize: 11, fontWeight: FontWeight.bold)),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Text(text,
              style: const TextStyle(color: Colors.white70, fontSize: 13, height: 1.3)),
        ),
      ],
    );
  }

  Widget _apiHelpTile() {
    return GestureDetector(
      onTap: _showApiHelp,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
        decoration: BoxDecoration(
          color: kGold.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: kGold.withValues(alpha: 0.2)),
        ),
        child: Row(
          children: [
            Icon(Icons.help_outline_rounded, color: kGold, size: 16),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                'How to get your API key',
                style: TextStyle(color: kGold, fontSize: 12, fontWeight: FontWeight.w600),
              ),
            ),
            Icon(Icons.chevron_right_rounded, color: kGold.withValues(alpha: 0.6), size: 18),
          ],
        ),
      ),
    );
  }

  Widget _infoTile(String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style:
                  const TextStyle(color: kTextSecondary, fontSize: 14)),
          Text(value,
              style: TextStyle(
                  color: valueColor ?? Colors.white70,
                  fontSize: 14,
                  fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}
