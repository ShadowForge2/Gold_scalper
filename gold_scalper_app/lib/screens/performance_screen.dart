import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/bot_provider.dart';
import '../models/performance.dart';
import '../widgets/metric_card.dart';
import '../widgets/equity_chart.dart';
import '../widgets/fade_in_scale.dart';
import '../widgets/ui/haptic.dart';
import '../theme.dart';

class PerformanceScreen extends StatefulWidget {
  const PerformanceScreen({super.key});

  @override
  State<PerformanceScreen> createState() => _PerformanceScreenState();
}

class _PerformanceScreenState extends State<PerformanceScreen> {
  bool _isYearlyView = true;

  @override
  Widget build(BuildContext context) {
    return Consumer<BotProvider>(
      builder: (context, bp, _) {
        if (bp.loading) {
          return const Center(child: CircularProgressIndicator(color: kGold));
        }

        final perf = bp.performance;
        if (perf == null) {
          return Center(
            child: Text('No data', style: TextStyle(color: kTextSecondary)),
          );
        }

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            FadeInScale(
              child: _buildSummaryGrid(perf),
            ),
            const SizedBox(height: 20),
            if (bp.equityCurve.isNotEmpty) ...[
              FadeInScale(
                delay: const Duration(milliseconds: 100),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const _SectionTitle('Equity Curve'),
                    Text(
                      _isYearlyView ? 'YEARLY (MONTHS)' : 'DAILY GROWTH',
                      style: TextStyle(color: kGold.withValues(alpha: 0.6), fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 1.0),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 10),
              FadeInScale(
                delay: const Duration(milliseconds: 150),
                child: _buildCard(
                  child: EquityChart(data: bp.equityCurve),
                ),
              ),
              const SizedBox(height: 12),
              FadeInScale(
                delay: const Duration(milliseconds: 200),
                child: _buildToggle(),
              ),
              const SizedBox(height: 20),
            ],
            const FadeInScale(
              delay: Duration(milliseconds: 250),
              child: _SectionTitle('Monthly Breakdown'),
            ),
            const SizedBox(height: 10),
            ...perf.monthly.map((m) => FadeInScale(
              delay: const Duration(milliseconds: 300),
              child: _breakdownRow(m.month, m.trades, m.pnl, m.winRate),
            )),
            const SizedBox(height: 8),
            FadeInScale(
              delay: const Duration(milliseconds: 400),
              child: _totalRow(perf),
            ),
            const SizedBox(height: 24),
          ],
        );
      },
    );
  }

  Widget _buildToggle() {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Row(
        children: [
          Expanded(
            child: _toggleItem('Yearly Curve', _isYearlyView, () => setState(() => _isYearlyView = true)),
          ),
          Expanded(
            child: _toggleItem('Daily Curve', !_isYearlyView, () => setState(() => _isYearlyView = false)),
          ),
        ],
      ),
    );
  }

  Widget _toggleItem(String label, bool active, VoidCallback onTap) {
    return GestureDetector(
      onTap: hapt(onTap),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          color: active ? kGold : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          label,
          textAlign: TextAlign.center,
          style: TextStyle(
            color: active ? Colors.black : kTextSecondary,
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  Widget _breakdownRow(String label, int trades, double pnl, double winRate) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.03)),
      ),
      child: Row(
        children: [
          SizedBox(
            width: 90,
            child: Text(
              label.toUpperCase(),
              style: const TextStyle(color: kTextPrimary, fontWeight: FontWeight.w800, fontSize: 12, letterSpacing: 0.5),
            ),
          ),
          Expanded(
            child: Text(
              '$trades trades',
              style: const TextStyle(color: kTextSecondary, fontSize: 12, fontWeight: FontWeight.w500),
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                '${pnl >= 0 ? '+' : ''}\$${pnl.toStringAsFixed(2)}',
                style: TextStyle(
                  color: pnl >= 0 ? kSuccess : kDanger,
                  fontWeight: FontWeight.w900,
                  fontSize: 14,
                ),
              ),
              Text(
                '${winRate.toStringAsFixed(1)}% WR',
                style: TextStyle(color: kTextSecondary.withValues(alpha: 0.7), fontSize: 10, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCard({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.04)),
      ),
      child: child,
    );
  }

  Widget _buildSummaryGrid(Performance p) {
    return GridView.count(
      crossAxisCount: 3,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisSpacing: 10,
      mainAxisSpacing: 10,
      childAspectRatio: 1.1,
      children: [
        MetricCard(title: 'TRADES', value: '${p.totalTrades}', valueColor: kInfo, icon: Icons.swap_horiz_rounded),
        MetricCard(title: 'WIN RATE', value: '${p.winRate}%', valueColor: kSuccess, icon: Icons.check_circle_rounded),
        MetricCard(title: 'PF', value: '${p.profitFactor}', valueColor: kGold, icon: Icons.trending_up_rounded),
        MetricCard(title: 'NET P&L', value: '\$${p.netPnl.toStringAsFixed(0)}', valueColor: kSuccess, icon: Icons.payments_rounded),
        MetricCard(title: 'MAX DD', value: '\$${p.maxDrawdown.toStringAsFixed(0)}', valueColor: kDanger, icon: Icons.arrow_downward_rounded),
        MetricCard(title: 'RETURN', value: '${(p.returnPct / 100).toStringAsFixed(0)}x', valueColor: kGold, icon: Icons.auto_graph_rounded),
      ],
    );
  }

  Widget _totalRow(Performance p) {
    final totalPnl = p.monthly.fold(0.0, (sum, m) => sum + m.pnl);
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      decoration: BoxDecoration(
        color: kGold.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: kGold.withValues(alpha: 0.15)),
      ),
      child: Row(
        children: [
          const Text(
            'TOTAL PERFORMANCE',
            style: TextStyle(color: kGold, fontWeight: FontWeight.w900, fontSize: 12, letterSpacing: 1.0),
          ),
          const Spacer(),
          Text(
            '${totalPnl >= 0 ? '+' : ''}\$${totalPnl.toStringAsFixed(2)}',
            style: const TextStyle(color: kGold, fontWeight: FontWeight.w900, fontSize: 18, letterSpacing: -0.5),
          ),
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String title;
  const _SectionTitle(this.title);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.bold,
          color: kTextPrimary,
          letterSpacing: -0.2,
        ),
      ),
    );
  }
}
