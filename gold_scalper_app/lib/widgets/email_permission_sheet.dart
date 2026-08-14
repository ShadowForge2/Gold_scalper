import 'package:flutter/material.dart';
import '../theme.dart';
import '../widgets/ui/haptic.dart';

class EmailPermissionResult {
  final bool allowEmail;
  final bool allowPush;
  final bool allowMarketing;

  const EmailPermissionResult({
    this.allowEmail = false,
    this.allowPush = true,
    this.allowMarketing = false,
  });
}

class EmailPermissionSheet extends StatefulWidget {
  final String? email;

  const EmailPermissionSheet({super.key, this.email});

  @override
  State<EmailPermissionSheet> createState() => _EmailPermissionSheetState();
}

class _EmailPermissionSheetState extends State<EmailPermissionSheet> {
  bool _allowPush = true;
  bool _allowEmail = true;
  bool _allowMarketing = false;

  Future<void> _dismiss(EmailPermissionResult? result) async {
    Navigator.of(context).pop(result);
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 24, 24, 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 40,
              height: 4,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: kDarkBorder,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: kGold.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: kGold.withValues(alpha: 0.25)),
                  ),
                  child: const Icon(Icons.notifications_active_rounded,
                      color: kGold, size: 24),
                ),
                const SizedBox(width: 14),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Stay in the loop',
                          style: TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              letterSpacing: -0.3)),
                      SizedBox(height: 2),
                      Text('Choose how QuantoraFX reaches you',
                          style: TextStyle(color: kTextSecondary, fontSize: 13)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            _toggleTile(
              icon: Icons.notifications_rounded,
              title: 'Push notifications',
              subtitle: 'Trade alerts, bot status and daily recaps — right on this device.',
              value: _allowPush,
              onChanged: (v) => setState(() => _allowPush = v),
            ),
            const SizedBox(height: 8),
            _toggleTile(
              icon: Icons.mark_email_read_rounded,
              title: 'Email updates',
              subtitle: widget.email != null
                  ? 'Billing notices, payment receipts, PnL recaps and product news to ${widget.email}.'
                  : 'Billing notices, payment receipts, PnL recaps and product news to your email.',
              value: _allowEmail,
              onChanged: (v) => setState(() => _allowEmail = v),
            ),
            const SizedBox(height: 8),
            _toggleTile(
              icon: Icons.local_offer_rounded,
              title: 'Occasional offers',
              subtitle: 'Special promotions, once in a while. Never spam.',
              value: _allowMarketing,
              onChanged: (v) => setState(() => _allowMarketing = v),
            ),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: hapt(() => _dismiss(EmailPermissionResult(
                      allowPush: _allowPush,
                      allowEmail: _allowEmail,
                      allowMarketing: _allowMarketing,
                    ))),
                style: ElevatedButton.styleFrom(
                  backgroundColor: kGold,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Continue',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
              ),
            ),
            const SizedBox(height: 8),
            Center(
              child: TextButton(
                onPressed: hapt(() => _dismiss(null)),
                child: const Text('Not now',
                    style: TextStyle(color: kTextSecondary, fontSize: 13)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _toggleTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.4)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 22, color: value ? kGold : kTextMuted),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 2),
                Text(subtitle,
                    style: const TextStyle(
                        color: kTextSecondary, fontSize: 12, height: 1.35)),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Switch(
            value: value,
            onChanged: onChanged,
            activeTrackColor: kGold.withValues(alpha: 0.35),
            activeColor: kGold,
          ),
        ],
      ),
    );
  }
}
