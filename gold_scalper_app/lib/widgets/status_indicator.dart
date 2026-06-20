import 'package:flutter/material.dart';

class StatusIndicator extends StatelessWidget {
  final bool active;
  final String label;
  final double size;

  const StatusIndicator({
    super.key,
    required this.active,
    required this.label,
    this.size = 12,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            color: active ? Colors.green : Colors.grey,
            shape: BoxShape.circle,
            boxShadow: active
                ? [BoxShadow(color: Colors.green.withValues(alpha: 0.5), blurRadius: 6)]
                : null,
          ),
        ),
        const SizedBox(width: 6),
        Text(label, style: TextStyle(fontSize: 13, color: Colors.grey[300])),
      ],
    );
  }
}

class BiasIndicator extends StatelessWidget {
  final String bias;
  final double strength;

  const BiasIndicator({super.key, required this.bias, required this.strength});

  @override
  Widget build(BuildContext context) {
    final isBullish = bias == 'BULLISH';
    final isBearish = bias == 'BEARISH';
    final color = isBullish
        ? Colors.green
        : isBearish
            ? Colors.red
            : Colors.grey;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            isBullish ? Icons.trending_up : isBearish ? Icons.trending_down : Icons.trending_flat,
            color: color,
            size: 18,
          ),
          const SizedBox(width: 6),
          Text(
            bias,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
          if (strength > 0) ...[
            const SizedBox(width: 4),
            Text(
              '(${strength.toInt()}%)',
              style: TextStyle(color: color.withValues(alpha: 0.7), fontSize: 12),
            ),
          ],
        ],
      ),
    );
  }
}
