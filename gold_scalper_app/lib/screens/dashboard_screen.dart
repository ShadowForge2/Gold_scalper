import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../providers/bot_provider.dart';
import '../widgets/status_indicator.dart';
import '../widgets/metric_card.dart';
import '../widgets/equity_chart.dart';
import '../widgets/skeletons/dashboard_skeleton.dart';
import '../widgets/fade_in_scale.dart';
import '../widgets/terminal_log.dart';
import '../widgets/ui/haptic.dart';
import 'subscription_screen.dart';
import '../theme.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  bool _isYearlyView = true;
  bool _dismissedWithdrawNotice = false;
  bool _dismissedSubscriptionNotice = false;
  bool _dismissedDemoNotice = false;

  @override
  void initState() {
    super.initState();
    _loadDismissedDemoNotice();
  }

  Future<void> _loadDismissedDemoNotice() async {
    final prefs = await SharedPreferences.getInstance();
    final ts = prefs.getInt('_dismissedDemoNotice');
    if (ts != null && DateTime.now().millisecondsSinceEpoch - ts < 86400000) {
      if (mounted) setState(() => _dismissedDemoNotice = true);
    }
  }

  Future<void> _dismissDemoNotice() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('_dismissedDemoNotice', DateTime.now().millisecondsSinceEpoch);
    if (mounted) setState(() => _dismissedDemoNotice = true);
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<BotProvider>(
      builder: (context, bp, _) {
        final s = bp.state;
        if (bp.loading || s == null) {
          return const DashboardSkeleton();
        }

        return RefreshIndicator(
          onRefresh: () async => bp.refresh(),
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              FadeInScale(
                child: _buildHeader(s, bp),
              ),
              if (bp.botRunning && !_dismissedWithdrawNotice)
                _buildWithdrawNotice(),
              if (!bp.canTrade && !bp.isDemo && !bp.hasNoAccounts && !bp.botRunning && !_dismissedSubscriptionNotice && bp.subscription.isNotEmpty)
                _buildSubscriptionBanner(context),
              if (bp.isDemo && !_dismissedDemoNotice)
                _buildDemoNotice(),
              const SizedBox(height: 16),
              FadeInScale(
                delay: const Duration(milliseconds: 100),
                child: _buildPriceRow(s),
              ),
              const SizedBox(height: 20),
              const FadeInScale(
                delay: Duration(milliseconds: 200),
                child: _SectionTitle('Performance Metrics'),
              ),
              const SizedBox(height: 10),
              FadeInScale(
                delay: const Duration(milliseconds: 250),
                child: _buildStatsGrid(s, bp),
              ),
              const SizedBox(height: 20),
              if (bp.equityCurve.isNotEmpty || bp.yearlyCurve.isNotEmpty || bp.monthlyCurve.isNotEmpty) ...[
                FadeInScale(
                  delay: const Duration(milliseconds: 350),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const _SectionTitle('Equity Curve'),
                      Text(
                        _isYearlyView ? 'YEARLY' : 'MONTHLY',
                        style: TextStyle(color: kGold.withValues(alpha: 0.6), fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 1.0),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 10),
                FadeInScale(
                  delay: const Duration(milliseconds: 400),
                  child: _buildChartCard(bp),
                ),
                const SizedBox(height: 12),
                FadeInScale(
                  delay: const Duration(milliseconds: 450),
                  child: _buildToggle(),
                ),
                const SizedBox(height: 20),
              ],
              const FadeInScale(
                delay: Duration(milliseconds: 500),
                child: _SectionTitle('Detailed Stats'),
              ),
              const SizedBox(height: 10),
              if (bp.performance != null)
                FadeInScale(
                  delay: const Duration(milliseconds: 550),
                  child: _buildQuickStats(bp.performance!),
                ),
              const SizedBox(height: 16),
              FadeInScale(
                delay: const Duration(milliseconds: 600),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const _SectionTitle('Terminal'),
                    const SizedBox(height: 8),
                    TerminalLog(logs: bp.logs, height: 200),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        );
      },
    );
  }

  Widget _buildHeader(s, BotProvider bp) {
    final isBullish = s.bias == 'BULLISH' || s.bias == 'BUY';
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0F172A), kDarkSurface],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.4),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              _buildLogo(),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    FittedBox(
                      fit: BoxFit.scaleDown,
                      alignment: Alignment.centerLeft,
                      child: ShaderMask(
                        shaderCallback: (bounds) => const LinearGradient(
                          colors: [kGold, Color(0xFFD4AF37), kGold],
                          stops: [0.0, 0.5, 1.0],
                        ).createShader(bounds),
                        blendMode: BlendMode.srcIn,
                        child: const Text('QuantoraFX',
                          style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900,
                            color: Colors.white, letterSpacing: 1.5, height: 1.2)),
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text('AI-Powered Forex Trading',
                      style: TextStyle(
                        fontSize: 11,
                        fontStyle: FontStyle.italic,
                        color: Colors.white.withValues(alpha: 0.6),
                        letterSpacing: 1.2,
                      ),
                    ),
                    const SizedBox(height: 2),
                    StatusIndicator(active: s.connected, label: s.broker),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              _buildStateBadge(s),
            ],
          ),
          const SizedBox(height: 12),
          Row(mainAxisAlignment: MainAxisAlignment.end, children: [_buildSubscriptionBtn()]),
          const SizedBox(height: 12),
          Container(height: 1, color: Colors.white.withValues(alpha: 0.04)),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('MARKET BIAS', style: TextStyle(color: kTextSecondary, fontSize: 10, fontWeight: FontWeight.w600, letterSpacing: 0.8)),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        Icon(
                          isBullish ? Icons.trending_up_rounded : Icons.trending_down_rounded,
                          color: isBullish ? kSuccess : kDanger,
                          size: 20,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          '${isBullish ? "BULLISH" : "BEARISH"}',
                          style: TextStyle(
                            color: isBullish ? kSuccess : kDanger,
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                            letterSpacing: 0.2,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'STR: ${s.biasStrength}%',
                          style: TextStyle(
                            color: (isBullish ? kSuccess : kDanger).withValues(alpha: 0.5),
                            fontSize: 11,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  const Text('BALANCE', style: TextStyle(color: kTextSecondary, fontSize: 10, fontWeight: FontWeight.w600, letterSpacing: 0.8)),
                  const SizedBox(height: 2),
                  Text(
                    '\$${s.balance.toStringAsFixed(2)}',
                    style: const TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w900,
                      color: kTextPrimary,
                      letterSpacing: -1.0,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                    decoration: BoxDecoration(
                      color: (s.dailyPnl >= 0 ? kSuccess : kDanger).withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '${s.dailyPnl >= 0 ? '+' : ''}\$${s.dailyPnl.toStringAsFixed(2)} today',
                      style: TextStyle(
                        color: s.dailyPnl >= 0 ? kSuccess : kDanger,
                        fontWeight: FontWeight.w700,
                        fontSize: 11,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              const Text('OPEN POSITIONS', style: TextStyle(color: kTextSecondary, fontSize: 10, fontWeight: FontWeight.w600, letterSpacing: 0.8)),
              const SizedBox(width: 10),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: kGold.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  '${s.openPositions}',
                  style: const TextStyle(
                    color: kGold,
                    fontWeight: FontWeight.w900,
                    fontSize: 14,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLogo() {
    return Container(
      width: 48, height: 48,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [kGold.withValues(alpha: 0.3), kGold.withValues(alpha: 0.05)],
          begin: Alignment.topLeft, end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: kGold.withValues(alpha: 0.2)),
        boxShadow: [BoxShadow(color: kGold.withValues(alpha: 0.1), blurRadius: 8, spreadRadius: 1)],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: Image.asset('assets/images/logo.png', width: 48, height: 48, fit: BoxFit.cover),
      ),
    );
  }

  Widget _buildStateBadge(s) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: _stateColor(s.state).withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: _stateColor(s.state).withValues(alpha: 0.25)),
      ),
      child: Text(
        s.state.replaceAll('_', ' '),
        style: TextStyle(color: _stateColor(s.state), fontWeight: FontWeight.w700, fontSize: 9, letterSpacing: 0.6),
      ),
    );
  }

  Widget _buildSubscriptionBtn() {
    return GestureDetector(
      onTap: hapt(() => Navigator.push(context, MaterialPageRoute(builder: (_) => const SubscriptionScreen()))),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: kGold.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: kGold.withValues(alpha: 0.2)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.credit_card_rounded, color: kGold.withValues(alpha: 0.7), size: 12),
            const SizedBox(width: 4),
            Text('SUBSCRIPTION', style: TextStyle(color: kGold.withValues(alpha: 0.7), fontSize: 9, fontWeight: FontWeight.w600, letterSpacing: 0.4)),
          ],
        ),
      ),
    );
  }

  Widget _buildPriceRow(s) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.03)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _priceItem('BID', s.bid.toStringAsFixed(s.priceDecimals), kTextPrimary),
          _divider(),
          _priceItem('ASK', s.ask.toStringAsFixed(s.priceDecimals), kTextPrimary),
          _divider(),
          _priceItem('SPREAD', s.spread.toStringAsFixed(s.priceDecimals), kGold),
        ],
      ),
    );
  }

  Widget _divider() {
    return Container(
      height: 24,
      width: 1,
      margin: const EdgeInsets.symmetric(horizontal: 24),
      color: Colors.white.withValues(alpha: 0.06),
    );
  }

  Widget _priceItem(String label, String value, Color valueColor) {
    return Column(
      children: [
        Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 10, fontWeight: FontWeight.w600, letterSpacing: 1.0)),
        const SizedBox(height: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w800,
            color: valueColor,
            letterSpacing: -0.5,
          ),
        ),
      ],
    );
  }

  Widget _buildStatsGrid(s, BotProvider bp) {
    final p = bp.performance;
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 1.6,
      children: [
        MetricCard(
          title: 'WIN RATE',
          value: p != null ? '${p.winRate.toStringAsFixed(1)}%' : '--',
          valueColor: kSuccess,
          icon: Icons.percent_rounded,
        ),
        MetricCard(
          title: 'PROFIT FACTOR',
          value: p != null ? p.profitFactor.toStringAsFixed(2) : '--',
          valueColor: kGold,
          icon: Icons.insights_rounded,
        ),
        MetricCard(
          title: 'MAX DRAWDOWN',
          value: p != null ? '\$${_fmt(p.maxDrawdown)}' : '--',
          valueColor: kDanger,
          icon: Icons.south_east_rounded,
        ),
        MetricCard(
          title: 'TOTAL TRADES',
          value: p != null ? '${p.totalTrades}' : '--',
          valueColor: kInfo,
          icon: Icons.swap_vert_rounded,
        ),
      ],
    );
  }

  String _fmt(double v) {
    if (v >= 1000) return '${(v / 1000).toStringAsFixed(1)}k';
    return v.toStringAsFixed(0);
  }

  Widget _buildWithdrawNotice() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.amber.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.amber.withValues(alpha: 0.2)),
        ),
        child: Row(
          children: [
            Icon(Icons.info_outline_rounded, color: Colors.amber, size: 18),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'For optimal performance, avoid withdrawing from your broker account until the next subscription cycle.',
                style: TextStyle(
                  color: Colors.amber.shade200,
                  fontSize: 12,
                  height: 1.3,
                ),
              ),
            ),
            GestureDetector(
              onTap: hapt(() => setState(() => _dismissedWithdrawNotice = true)),
              child: Padding(
                padding: const EdgeInsets.only(left: 8),
                child: Icon(Icons.close_rounded, color: Colors.amber.shade400, size: 18),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSubscriptionBanner(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: GestureDetector(
        onTap: hapt(() {
          _dismissedSubscriptionNotice = true;
          Navigator.push(context, MaterialPageRoute(builder: (_) => const SubscriptionScreen()));
        }),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: Colors.orange.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.orange.withValues(alpha: 0.2)),
          ),
          child: Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 18),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Your free instance has been exhausted. Subscribe to continue using the service.',
                  style: TextStyle(
                    color: Colors.orange.shade200,
                    fontSize: 12,
                    height: 1.3,
                  ),
                ),
              ),
              GestureDetector(
                onTap: hapt(() => setState(() => _dismissedSubscriptionNotice = true)),
                child: Padding(
                  padding: const EdgeInsets.only(left: 8),
                  child: Icon(Icons.close_rounded, color: Colors.orange.shade400, size: 18),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDemoNotice() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.blue.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.blue.withValues(alpha: 0.2)),
        ),
        child: Row(
          children: [
            Icon(Icons.info_outline_rounded, color: Colors.blue, size: 18),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Demo account connection is always free. Live accounts have a monthly quota with subscription.',
                style: TextStyle(
                  color: Colors.blue.shade200,
                  fontSize: 12,
                  height: 1.3,
                ),
              ),
            ),
            GestureDetector(
              onTap: hapt(() => _dismissDemoNotice()),
              child: Padding(
                padding: const EdgeInsets.only(left: 8),
                child: Icon(Icons.close_rounded, color: Colors.blue.shade400, size: 18),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChartCard(BotProvider bp) {
    final displayData = _isYearlyView 
        ? bp.yearlyCurve 
        : bp.monthlyCurve;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.04)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.2),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: EquityChart(data: displayData),
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
            child: _toggleItem('Yearly', _isYearlyView, () => setState(() => _isYearlyView = true)),
          ),
          Expanded(
            child: _toggleItem('Monthly', !_isYearlyView, () => setState(() => _isYearlyView = false)),
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

  Widget _buildQuickStats(perf) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: kDarkBorder.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          _statRow('Net P&L', '\$${perf.netPnl.toStringAsFixed(2)}', Colors.green),
          const Divider(color: kDarkBorder, height: 20),
          _statRow('Average Win', '\$${perf.avgWin.toStringAsFixed(2)}', Colors.green),
          const Divider(color: kDarkBorder, height: 20),
          _statRow('Average Loss', '\$${perf.avgLoss.toStringAsFixed(2)}', Colors.red),
          const Divider(color: kDarkBorder, height: 20),
          _statRow('Total Return', '${perf.returnPct.toStringAsFixed(1)}%', kGold),
          const Divider(color: kDarkBorder, height: 20),
          _statRow('Wins / Losses', '${perf.wins} / ${perf.losses}', Colors.white),
        ],
      ),
    );
  }

  Widget _statRow(String label, String value, Color valueColor) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text(label, style: const TextStyle(color: kTextSecondary, fontSize: 13)),
          const Spacer(),
          Text(
            value,
            style: TextStyle(color: valueColor, fontWeight: FontWeight.bold, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Color _stateColor(String state) {
    switch (state) {
      case 'IN_TRADE':
        return Colors.green;
      case 'AWAITING_SIGNAL':
        return Colors.amber;
      case 'STOPPED':
        return Colors.red;
      default:
        return Colors.grey;
    }
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
          fontSize: 15,
          fontWeight: FontWeight.bold,
          color: Colors.white,
          letterSpacing: -0.2,
        ),
      ),
    );
  }
}
