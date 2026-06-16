import 'package:flutter/material.dart';
import 'package:shimmer/shimmer.dart';
import '../../theme.dart';

class DashboardSkeleton extends StatelessWidget {
  const DashboardSkeleton({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _shimmerHeader(),
        const SizedBox(height: 16),
        _shimmerPriceRow(),
        const SizedBox(height: 20),
        _shimmerSectionTitle(),
        const SizedBox(height: 10),
        _shimmerStatsGrid(),
        const SizedBox(height: 20),
        _shimmerSectionTitle(),
        const SizedBox(height: 10),
        _shimmerChartCard(),
      ],
    );
  }

  Widget _shimmerHeader() {
    return Shimmer.fromColors(
      baseColor: kDarkCard,
      highlightColor: kDarkBorder.withValues(alpha: 0.2),
      child: Container(
        height: 180,
        decoration: BoxDecoration(
          color: kDarkCard,
          borderRadius: BorderRadius.circular(24),
        ),
      ),
    );
  }

  Widget _shimmerPriceRow() {
    return Shimmer.fromColors(
      baseColor: kDarkCard,
      highlightColor: kDarkBorder.withValues(alpha: 0.2),
      child: Container(
        height: 60,
        decoration: BoxDecoration(
          color: kDarkCard,
          borderRadius: BorderRadius.circular(16),
        ),
      ),
    );
  }

  Widget _shimmerSectionTitle() {
    return Shimmer.fromColors(
      baseColor: kDarkCard,
      highlightColor: kDarkBorder.withValues(alpha: 0.2),
      child: Container(
        height: 20,
        width: 150,
        decoration: BoxDecoration(
          color: kDarkCard,
          borderRadius: BorderRadius.circular(6),
        ),
      ),
    );
  }

  Widget _shimmerStatsGrid() {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 1.6,
      children: List.generate(4, (index) => Shimmer.fromColors(
        baseColor: kDarkCard,
        highlightColor: kDarkBorder.withValues(alpha: 0.2),
        child: Container(
          decoration: BoxDecoration(
            color: kDarkCard,
            borderRadius: BorderRadius.circular(20),
          ),
        ),
      )),
    );
  }

  Widget _shimmerChartCard() {
    return Shimmer.fromColors(
      baseColor: kDarkCard,
      highlightColor: kDarkBorder.withValues(alpha: 0.2),
      child: Container(
        height: 230,
        decoration: BoxDecoration(
          color: kDarkCard,
          borderRadius: BorderRadius.circular(20),
        ),
      ),
    );
  }
}
