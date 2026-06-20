import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/performance.dart';
import '../theme.dart';

class EquityChart extends StatelessWidget {
  final List<EquityPoint> data;

  const EquityChart({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    if (data.isEmpty) return const SizedBox();

    final minTime = data.first.time.millisecondsSinceEpoch.toDouble();
    final spots = data.map((p) {
      return FlSpot(
        (p.time.millisecondsSinceEpoch.toDouble() - minTime) / 1000,
        p.balance,
      );
    }).toList();

    return SizedBox(
      height: 200,
      child: LineChart(
        LineChartData(
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            getDrawingHorizontalLine: (v) => FlLine(
              color: Colors.white.withValues(alpha: 0.04),
              strokeWidth: 1,
            ),
          ),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
            bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          ),
          borderData: FlBorderData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              color: kGold,
              barWidth: 2.5,
              dotData: FlDotData(show: false),
              belowBarData: BarAreaData(
                show: true,
                color: kGold.withValues(alpha: 0.08),
              ),
            ),
          ],
          lineTouchData: LineTouchData(
            handleBuiltInTouches: true,
            touchTooltipData: LineTouchTooltipData(
              getTooltipColor: (spot) => kDarkSurface,
              tooltipRoundedRadius: 8,
              getTooltipItems: (spots) => spots.map((s) {
                return LineTooltipItem(
                  '\$${s.y.toStringAsFixed(2)}',
                  const TextStyle(color: kGold, fontWeight: FontWeight.bold, fontSize: 13),
                );
              }).toList(),
            ),
            getTouchedSpotIndicator: (LineChartBarData barData, List<int> spotIndexes) {
              return spotIndexes.map((index) {
                return TouchedSpotIndicatorData(
                  FlLine(color: kGold.withValues(alpha: 0.2), strokeWidth: 2, dashArray: [5, 5]),
                  FlDotData(
                    show: true,
                    getDotPainter: (spot, percent, barData, index) => FlDotCirclePainter(
                      radius: 6,
                      color: kGold,
                      strokeWidth: 2,
                      strokeColor: Colors.white,
                    ),
                  ),
                );
              }).toList();
            },
          ),
        ),
      ),
    );
  }
}
