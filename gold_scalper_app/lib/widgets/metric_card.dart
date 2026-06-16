import 'package:flutter/material.dart';
import '../theme.dart';

class MetricCard extends StatelessWidget {
  final String title;
  final String value;
  final Color? valueColor;
  final IconData? icon;
  final String? subtitle;

  const MetricCard({
    super.key,
    required this.title,
    required this.value,
    this.valueColor,
    this.icon,
    this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    final color = valueColor ?? kTextPrimary;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kDarkCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.04)),
        gradient: LinearGradient(
          colors: [kDarkCard, kDarkSurface.withValues(alpha: 0.5)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.2),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Row(
            children: [
              if (icon != null) ...[
                Container(
                  padding: const EdgeInsets.all(6),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(icon, size: 12, color: color),
                ),
                const SizedBox(width: 10),
              ],
              Text(
                title,
                style: const TextStyle(
                  color: kTextSecondary,
                  fontSize: 10,
                  letterSpacing: 1.0,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 24,
              fontWeight: FontWeight.w900,
              letterSpacing: -0.8,
            ),
          ),
          if (subtitle != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                subtitle!,
                style: const TextStyle(color: kTextSecondary, fontSize: 11),
              ),
            ),
        ],
      ),
    );
  }
}
