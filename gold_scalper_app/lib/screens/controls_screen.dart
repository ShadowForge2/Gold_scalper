import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/bot_provider.dart';
import '../widgets/status_indicator.dart';
import '../widgets/fade_in_scale.dart';
import '../theme.dart';

class ControlsScreen extends StatelessWidget {
  const ControlsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<BotProvider>(
      builder: (context, bp, _) {
        final s = bp.state;
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            FadeInScale(
              child: _buildBotControlCard(context, s, bp),
            ),
            const SizedBox(height: 16),
            FadeInScale(
              delay: const Duration(milliseconds: 100),
              child: _buildConfigCard(bp),
            ),
            const SizedBox(height: 16),
            FadeInScale(
              delay: const Duration(milliseconds: 200),
              child: _buildSessionsCard(bp),
            ),
            const SizedBox(height: 16),
            FadeInScale(
              delay: const Duration(milliseconds: 300),
              child: _buildDangerZone(context, bp),
            ),
          ],
        );
      },
    );
  }

  Widget _buildBotControlCard(BuildContext context, s, BotProvider bp) {
    final isRunning = s != null && s.status == 'running';
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isRunning
              ? [Colors.green.withValues(alpha: 0.08), kDarkCard]
              : [Colors.red.withValues(alpha: 0.08), kDarkCard],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              StatusIndicator(active: isRunning, label: isRunning ? 'Running' : 'Stopped', size: 14),
              const Spacer(),
              if (s != null) Text(s.state.replaceAll('_', ' '), style: const TextStyle(color: kTextSecondary)),
            ],
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (!isRunning)
                ElevatedButton.icon(
                  onPressed: () => bp.startBot(),
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Start Bot'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                )
              else
                OutlinedButton.icon(
                  onPressed: () => _confirmStopBot(context, bp),
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop Bot'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.red,
                    side: const BorderSide(color: Colors.red),
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildConfigCard(BotProvider bp) {
    final cfg = bp.config;
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
          const Text('Parameters', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white, letterSpacing: -0.2)),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: kGold.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: kGold.withValues(alpha: 0.15)),
            ),
            child: const Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline_rounded, color: kGold, size: 14),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Keep all settings as default to ensure system efficiency.',
                    style: TextStyle(color: kGold, fontSize: 11, height: 1.3),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          _sliderSetting('Lot Multiplier', cfg.lotMultiplier, 1, 10, 1, (v) {
            bp.updateConfig(cfg.copyWith(lotMultiplier: v));
          }),
          _sliderSetting('Entry Threshold', cfg.signalEntryThreshold, 0.1, 1.0, 0.05, (v) {
            bp.updateConfig(cfg.copyWith(signalEntryThreshold: v));
          }),
          _sliderSetting('Max Daily Loss (\$)', cfg.maxDailyLoss, 1, 50, 1, (v) {
            bp.updateConfig(cfg.copyWith(maxDailyLoss: v));
          }),
          _sliderSetting('Max Trades/Event', cfg.maxTradesPerEvent.toDouble(), 1, 20, 1, (v) {
            bp.updateConfig(cfg.copyWith(maxTradesPerEvent: v.round()));
          }),
          _sliderSetting('Max Consecutive Losses', cfg.maxConsecutiveLosses.toDouble(), 1, 10, 1, (v) {
            bp.updateConfig(cfg.copyWith(maxConsecutiveLosses: v.round()));
          }),
          _sliderSetting('Cooldown (sec)', cfg.reEntryCooldownSec.toDouble(), 10, 600, 10, (v) {
            bp.updateConfig(cfg.copyWith(reEntryCooldownSec: v.round()));
          }),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () => bp.saveConfig(),
              icon: const Icon(Icons.save_rounded, size: 16),
              label: const Text('Save Settings'),
              style: ElevatedButton.styleFrom(
                backgroundColor: kGold,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sliderSetting(String label, double value, double min, double max, double step, Function(double) onChanged) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 13)),
              Text(
                step >= 1 ? value.toInt().toString() : value.toStringAsFixed(2),
                style: const TextStyle(color: kGold, fontWeight: FontWeight.bold, fontSize: 14),
              ),
            ],
          ),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: kGold,
              inactiveTrackColor: Colors.white.withValues(alpha: 0.08),
              thumbColor: kGold,
              overlayColor: kGold.withValues(alpha: 0.1),
            ),
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              divisions: ((max - min) / step).round(),
              onChanged: onChanged,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSessionsCard(BotProvider bp) {
    final cfg = bp.config;
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
          const Text('Trading Sessions', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white, letterSpacing: -0.2)),
          const SizedBox(height: 12),
          _sessionToggle('Asia (00:00-08:00 UTC)', cfg.asiaSession, (v) {
            bp.updateConfig(cfg.copyWith(asiaSession: v));
          }),
          _sessionToggle('London (07:00-16:00 UTC)', cfg.londonSession, (v) {
            bp.updateConfig(cfg.copyWith(londonSession: v));
          }),
          _sessionToggle('New York (12:00-21:00 UTC)', cfg.newYorkSession, (v) {
            bp.updateConfig(cfg.copyWith(newYorkSession: v));
          }),
        ],
      ),
    );
  }

  Widget _sessionToggle(String label, bool value, Function(bool) onChanged) {
    return SwitchListTile(
      title: Text(label, style: const TextStyle(color: Colors.white70, fontSize: 13)),
      value: value,
      onChanged: onChanged,
      activeTrackColor: kGold,
      contentPadding: EdgeInsets.zero,
      dense: true,
    );
  }

  Future<void> _confirmStopBot(BuildContext context, BotProvider bp) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1A1A1A),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        title: const Row(
          children: [
            Icon(Icons.stop_circle_outlined, color: Colors.redAccent, size: 22),
            SizedBox(width: 10),
            Text('Stop Bot', style: TextStyle(color: Colors.redAccent, fontSize: 16)),
          ],
        ),
        content: const Text(
          'Are you sure you want to stop the bot?\n\n'
          'Active positions will remain open.',
          style: TextStyle(color: Colors.white70, fontSize: 13, height: 1.4),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel', style: TextStyle(color: Colors.white54)),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Stop', style: TextStyle(color: Colors.redAccent, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await bp.stopBot();
    }
  }

  Future<void> _confirmCloseAll(BuildContext context, BotProvider bp) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1A1A1A),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        title: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.redAccent, size: 22),
            SizedBox(width: 10),
            Text('Close All Positions', style: TextStyle(color: Colors.redAccent, fontSize: 16)),
          ],
        ),
        content: const Text(
          'This will close ALL open positions — both profitable and losing.\n\n'
          'This action cannot be undone.',
          style: TextStyle(color: Colors.white70, fontSize: 13, height: 1.4),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel', style: TextStyle(color: Colors.white54)),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Close All', style: TextStyle(color: Colors.redAccent, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await bp.closeAllPositions();
    }
  }

  Widget _buildDangerZone(BuildContext context, BotProvider bp) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.red.withValues(alpha: 0.15)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Danger Zone', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.redAccent)),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => _confirmCloseAll(context, bp),
              icon: const Icon(Icons.close),
              label: const Text('Close All Positions'),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.red,
                side: const BorderSide(color: Colors.red),
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
