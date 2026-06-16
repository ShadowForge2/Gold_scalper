import 'package:flutter/material.dart';
import '../widgets/status_indicator.dart';
import '../widgets/fade_in_scale.dart';
import '../theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _emailController = TextEditingController(text: '');
  final _passwordController = TextEditingController(text: '');
  bool _isDemo = true;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        FadeInScale(
          child: _buildSection('Connection', [
            _inputField('Email', _emailController, keyboardType: TextInputType.emailAddress),
            const SizedBox(height: 8),
            _inputField('Password', _passwordController, obscure: true),
            const SizedBox(height: 12),
            _saveButton(),
            const SizedBox(height: 12),
            _settingTile('Broker', 'Capital.com'),
            _accountTypeToggle(),
            _settingTile('API Endpoint', 'https://gold-scalper.onrender.com'),
            _settingTile('Symbol', 'XAUUSD'),
            _settingTile('Leverage', '1:100'),
          ]),
        ),
        const SizedBox(height: 16),
        FadeInScale(
          delay: const Duration(milliseconds: 100),
          child: _buildSection('Notifications', [
            _switchTile('Push notifications', true),
            _switchTile('Trade alerts', true),
            _switchTile('Daily summary', false),
            _switchTile('Error alerts', true),
          ]),
        ),
        const SizedBox(height: 16),
        FadeInScale(
          delay: const Duration(milliseconds: 200),
          child: _buildSection('Data', [
            _settingTile('Storage', 'Local'),
            _settingTile('Auto-refresh', '5 seconds'),
            _settingTile('Trade history', '385 trades'),
            _settingTile('Backtest data', '2025 Full Year'),
          ]),
        ),
        const SizedBox(height: 16),
        FadeInScale(
          delay: const Duration(milliseconds: 300),
          child: _buildFooter(),
        ),
      ],
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
          Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white, letterSpacing: -0.2)),
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
          StatusIndicator(active: true, label: 'API Connected', size: 14),
          SizedBox(height: 12),
          Text('App v1.3.0', style: TextStyle(color: kTextSecondary, fontSize: 13)),
          Text('Developer: Agni Kai', style: TextStyle(color: kTextSecondary, fontSize: 11)),
          Text('Fire Star LTD', style: TextStyle(color: kTextSecondary, fontSize: 11)),
        ],
      ),
    );
  }

  Widget _saveButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: _saveCredentials,
        style: ElevatedButton.styleFrom(
          backgroundColor: kGold,
          foregroundColor: Colors.black,
          padding: const EdgeInsets.symmetric(vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
        child: const Text('Save Credentials', style: TextStyle(fontWeight: FontWeight.bold)),
      ),
    );
  }

  void _saveCredentials() {
    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();
    if (email.isEmpty || password.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Email and password are required')),
      );
      return;
    }
    final accountType = _isDemo ? 'Demo' : 'Live';
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Credentials saved for $accountType')),
    );
  }

  Widget _accountTypeToggle() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Text('Account Type', style: TextStyle(color: kTextSecondary, fontSize: 14)),
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
                  onTap: () => setState(() => _isDemo = true),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                    decoration: BoxDecoration(
                      color: _isDemo ? kGold : Colors.transparent,
                      borderRadius: const BorderRadius.horizontal(left: Radius.circular(7)),
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
                  onTap: () => setState(() => _isDemo = false),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                    decoration: BoxDecoration(
                      color: !_isDemo ? kGold : Colors.transparent,
                      borderRadius: const BorderRadius.horizontal(right: Radius.circular(7)),
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

  Widget _inputField(String label, TextEditingController controller, {bool obscure = false, TextInputType? keyboardType}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 13)),
        const SizedBox(height: 4),
        TextField(
          controller: controller,
          obscureText: obscure,
          keyboardType: keyboardType,
          style: const TextStyle(color: Colors.white, fontSize: 14),
          decoration: InputDecoration(
            isDense: true,
            contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
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

  Widget _settingTile(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 14)),
          Text(value, style: const TextStyle(color: Colors.white70, fontSize: 14, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  Widget _switchTile(String label, bool value) {
    return SwitchListTile(
      title: Text(label, style: const TextStyle(color: Colors.white70, fontSize: 14)),
      value: value,
      onChanged: (_) {},
      activeTrackColor: kGold,
      contentPadding: EdgeInsets.zero,
      dense: true,
    );
  }
}
