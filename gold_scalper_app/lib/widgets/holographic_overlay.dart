import 'dart:math';
import 'dart:ui' show lerpDouble;
import 'package:flutter/material.dart';

enum HoloType {
  candle,
  buyArrow,
  sellArrow,
  trendLine,
  supportResistance,
  fibonacci,
  radarScan,
  hudRect,
  neuralNode,
  neuralLine,
  binaryNumber,
  hexString,
  aiCalc,
  marketStructure,
  worldGrid,
  circuitPattern,
  codeSnippet,
  confidencePct,
  tradingStat,
  coordMarker,
  signalDot,
  rotatingHexagon,
  scanCrosshair,
}

class _HoloObject {
  final HoloType type;
  double x;
  double y;
  final double orbitRadius;
  final double orbitSpeed;
  final double orbitPhase;
  final double driftXSpeed;
  final double driftYSpeed;
  final double rotationSpeed;
  final double scaleSpeed;
  final double scalePhase;
  final double opacityPhase;
  final double opacitySpeed;
  final double size;
  final String? label;
  double _life;

  _HoloObject({
    required this.type,
    required this.x,
    required this.y,
    this.orbitRadius = 0,
    this.orbitSpeed = 0,
    this.orbitPhase = 0,
    this.driftXSpeed = 0,
    this.driftYSpeed = 0,
    this.rotationSpeed = 0,
    this.scaleSpeed = 0,
    this.scalePhase = 0,
    this.opacityPhase = 0,
    this.opacitySpeed = 1.0,
    this.size = 14,
    this.label,
    double life = 1.0,
  }) : _life = life;
}

class HolographicOverlay extends StatefulWidget {
  const HolographicOverlay({super.key});
  @override State<HolographicOverlay> createState() => _HolographicOverlayState();
}

class _HolographicOverlayState extends State<HolographicOverlay> with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  final _rand = Random(42);
  late List<_HoloObject> _objects;
  final _objectPool = <_HoloObject>[];

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(seconds: 60))..repeat();
    _objects = [];
    for (int i = 0; i < 60; i++) {
      _objects.add(_spawn());
    }
  }

  _HoloObject _spawn({double? x, double? y}) {
    final types = HoloType.values;
    final type = types[_rand.nextInt(types.length)];
    return _HoloObject(
      type: type,
      x: x ?? 0.2 + _rand.nextDouble() * 0.6,
      y: y ?? 0.15 + _rand.nextDouble() * 0.7,
      orbitRadius: 20 + _rand.nextDouble() * 80,
      orbitSpeed: (0.08 + _rand.nextDouble() * 0.25) * (_rand.nextBool() ? 1 : -1),
      orbitPhase: _rand.nextDouble() * 2 * pi,
      driftXSpeed: (-0.3 + _rand.nextDouble() * 0.6) * 0.02,
      driftYSpeed: (-0.3 + _rand.nextDouble() * 0.6) * 0.02,
      rotationSpeed: (-0.5 + _rand.nextDouble()) * 0.03,
      scaleSpeed: 0.3 + _rand.nextDouble() * 0.7,
      scalePhase: _rand.nextDouble() * 2 * pi,
      opacityPhase: _rand.nextDouble() * 2 * pi,
      opacitySpeed: 0.2 + _rand.nextDouble() * 0.5,
      size: 8 + _rand.nextDouble() * 20,
      label: _randomLabel(type),
      life: 0.0,
    );
  }

  String _randomLabel(HoloType type) {
    const bin = ['0', '1'];
    const hex = ['0x3F', '0xA2', '0xCC', '0x7B', '0x1E', '0xF9', '0x4D', '0x80', '0x5A', '0xE6', '0x2C', '0x91'];
    const codes = ['if(bias){buy()}', 'calc:score=0.78', 'await signal;', 'entry=2345.6', 'lot=0.15', 'SL=23.4', 'TP=46.8', 'risk=1.2%', 'conf=0.83', 'trend=UP'];
    const stats = <String>['W:266', 'L:119', 'WR:69%', 'PF:4.2', r'+$3.6k', 'DD:8.2%', r'AVG:+$21.9', r'MAX:-$18.4'];
    switch (type) {
      case HoloType.binaryNumber: return bin[_rand.nextInt(bin.length)];
      case HoloType.hexString: return hex[_rand.nextInt(hex.length)];
      case HoloType.codeSnippet: return codes[_rand.nextInt(codes.length)];
      case HoloType.tradingStat: return stats[_rand.nextInt(stats.length)];
      case HoloType.confidencePct: return '${50 + _rand.nextInt(45)}%';
      case HoloType.aiCalc: return 'σ=${(0.1 + _rand.nextDouble() * 0.8).toStringAsFixed(2)}';
      case HoloType.coordMarker: return '(${_rand.nextInt(100)},${_rand.nextInt(100)})';
      case HoloType.signalDot: return _rand.nextBool() ? 'BUY' : 'SELL';
      default: return '';
    }
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: AnimatedBuilder(
        animation: _ctrl,
        builder: (context, _) => CustomPaint(
          painter: _HoloPainter(_ctrl.value, _rand, _objects, _objectPool, _spawn),
          size: Size.infinite,
        ),
      ),
    );
  }
}

class _HoloPainter extends CustomPainter {
  final double t;
  final Random rand;
  final List<_HoloObject> objects;
  final List<_HoloObject> pool;
  final _HoloObject Function({double? x, double? y}) spawn;

  _HoloPainter(this.t, this.rand, this.objects, this.pool, this.spawn);

  @override
  void paint(Canvas canvas, Size size) {
    for (int i = objects.length - 1; i >= 0; i--) {
      final obj = objects[i];
      obj._life = (obj._life + 0.005).clamp(0.0, 1.0);
      if (obj._life >= 1.0 && rand.nextDouble() < 0.002) {
        pool.add(obj);
        objects.removeAt(i);
        _replenish();
        continue;
      }

      final ox = cos(t * obj.orbitSpeed * 2 * pi + obj.orbitPhase) * obj.orbitRadius;
      final oy = sin(t * obj.orbitSpeed * 2 * pi + obj.orbitPhase) * obj.orbitRadius;
      final dx = sin(t * obj.driftXSpeed * 2 * pi) * 40;
      final dy = cos(t * obj.driftYSpeed * 2 * pi) * 40;

      final px = obj.x * size.width + ox + dx;
      final py = obj.y * size.height + oy + dy;

      if (px < -50 || px > size.width + 50 || py < -50 || py > size.height + 50) {
        pool.add(obj);
        objects.removeAt(i);
        _replenish();
        continue;
      }

      final opacity = lerpDouble(0.0, 0.08 + sin(t * obj.opacitySpeed * 2 * pi + obj.opacityPhase) * 0.04, obj._life)!;
      if (opacity <= 0) continue;

      final scale = 0.95 + sin(t * obj.scaleSpeed * 2 * pi + obj.scalePhase) * 0.05;
      final rot = t * obj.rotationSpeed * 2 * pi;
      final color = Colors.cyan.withValues(alpha: opacity);
      final paint = Paint()
        ..color = color
        ..strokeWidth = 1.0
        ..style = PaintingStyle.stroke;

      canvas.save();
      canvas.translate(px, py);
      canvas.scale(scale);
      canvas.rotate(rot);

      _drawObject(canvas, obj, color, paint, size);

      canvas.restore();
    }

    while (objects.length < 60) {
      objects.add(pool.isNotEmpty ? (() { final r = pool.removeLast(); final n = spawn(x: r.x, y: r.y); n.x = rand.nextDouble(); n.y = rand.nextDouble(); return n; })() : spawn());
    }
  }

  void _replenish() {
    if (pool.isNotEmpty) {
      final recycled = pool.removeLast();
      final newObj = spawn(x: recycled.x, y: recycled.y);
      newObj.x = rand.nextDouble();
      newObj.y = rand.nextDouble();
      objects.add(newObj);
    } else {
      objects.add(spawn());
    }
  }

  void _drawObject(Canvas canvas, _HoloObject obj, Color color, Paint paint, Size size) {
    switch (obj.type) {
      case HoloType.candle:
        paint.strokeWidth = 2.0;
        canvas.drawLine(const Offset(0, -6), const Offset(0, 6), paint);
        paint.strokeWidth = 3.5;
        canvas.drawRect(Rect.fromCenter(center: Offset.zero, width: 4, height: 8), paint);
        break;

      case HoloType.buyArrow:
        final path = Path()
          ..moveTo(0, -8)
          ..lineTo(-5, 2)
          ..lineTo(5, 2)
          ..close();
        canvas.drawPath(path, paint);
        canvas.drawLine(const Offset(0, 2), Offset(0, 10), paint);
        break;

      case HoloType.sellArrow:
        final path = Path()
          ..moveTo(0, 8)
          ..lineTo(-5, -2)
          ..lineTo(5, -2)
          ..close();
        canvas.drawPath(path, paint);
        canvas.drawLine(const Offset(0, -2), const Offset(0, -10), paint);
        break;

      case HoloType.trendLine:
        canvas.drawLine(const Offset(-8, 6), const Offset(8, -6), paint);
        canvas.drawCircle(const Offset(8, -6), 2, paint);
        break;

      case HoloType.supportResistance:
        canvas.drawLine(const Offset(-8, 0), const Offset(8, 0), paint);
        paint.strokeWidth = 0.5;
        canvas.drawLine(const Offset(-8, 2), const Offset(-4, 0), paint);
        canvas.drawLine(const Offset(8, 2), const Offset(4, 0), paint);
        if (obj.label != null) {
          _drawText(canvas, obj.label!, const Offset(10, -2), color, 8);
        }
        break;

      case HoloType.fibonacci:
        const levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
        for (int i = 0; i < levels.length; i++) {
          final yOff = -8 + i * 2.5;
          canvas.drawLine(Offset(-8, yOff), Offset(8, yOff), paint);
          if (i % 2 == 0) {
            _drawText(canvas, '${(levels[i] * 100).toInt()}%', Offset(10, yOff - 4), color, 6);
          }
        }
        break;

      case HoloType.radarScan:
        paint.strokeWidth = 0.8;
        canvas.drawCircle(Offset.zero, 10, paint);
        canvas.drawCircle(Offset.zero, 6, paint);
        canvas.drawLine(const Offset(-10, 0), const Offset(10, 0), paint);
        canvas.drawLine(const Offset(0, -10), const Offset(0, 10), paint);
        final angle = t * 2 * pi;
        canvas.drawLine(Offset.zero, Offset(cos(angle) * 10, sin(angle) * 10), paint);
        break;

      case HoloType.hudRect:
        final r = 10.0;
        final gap = 4.0;
        canvas.drawLine(Offset(-r, -r), Offset(-r, -r + gap), paint);
        canvas.drawLine(Offset(-r, -r), Offset(-r + gap, -r), paint);
        canvas.drawLine(Offset(r, -r), Offset(r, -r + gap), paint);
        canvas.drawLine(Offset(r, -r), Offset(r - gap, -r), paint);
        canvas.drawLine(Offset(-r, r), Offset(-r, r - gap), paint);
        canvas.drawLine(Offset(-r, r), Offset(-r + gap, r), paint);
        canvas.drawLine(Offset(r, r), Offset(r, r - gap), paint);
        canvas.drawLine(Offset(r, r), Offset(r - gap, r), paint);
        canvas.drawLine(Offset(0, -r), Offset(0, -r - 3), paint);
        canvas.drawLine(Offset(0, r), Offset(0, r + 3), paint);
        canvas.drawLine(Offset(-r, 0), Offset(-r - 3, 0), paint);
        canvas.drawLine(Offset(r, 0), Offset(r + 3, 0), paint);
        break;

      case HoloType.neuralNode:
        paint.style = PaintingStyle.fill;
        canvas.drawCircle(Offset.zero, 2, paint..color = color.withValues(alpha: color.opacity * 0.5));
        paint.style = PaintingStyle.stroke;
        canvas.drawCircle(Offset.zero, 2, paint..color = color);
        break;

      case HoloType.neuralLine:
        canvas.drawLine(Offset(-4, -3), Offset(4, 3), paint);
        canvas.drawLine(Offset(-4, 3), Offset(4, -3), paint);
        break;

      case HoloType.binaryNumber:
        _drawText(canvas, obj.label ?? '0', Offset.zero, color, obj.size);
        break;

      case HoloType.hexString:
        _drawText(canvas, obj.label ?? '0xFF', Offset.zero, color, obj.size);
        break;

      case HoloType.aiCalc:
        _drawText(canvas, obj.label ?? 'calc()', Offset.zero, color, obj.size);
        break;

      case HoloType.marketStructure:
        final path = Path()
          ..moveTo(-8, 4)
          ..lineTo(-4, -2)
          ..lineTo(0, 3)
          ..lineTo(4, -4)
          ..lineTo(8, 2);
        canvas.drawPath(path, paint);
        canvas.drawCircle(const Offset(-8, 4), 1.5, paint);
        canvas.drawCircle(const Offset(8, 2), 1.5, paint);
        break;

      case HoloType.worldGrid:
        paint.strokeWidth = 0.5;
        for (int i = -2; i <= 2; i++) {
          canvas.drawLine(Offset(-8 + i * 4, -6), Offset(-8 + i * 4, 6), paint);
          canvas.drawLine(Offset(-8, -6 + i * 3), Offset(8, -6 + i * 3), paint);
        }
        break;

      case HoloType.circuitPattern:
        paint.strokeWidth = 0.8;
        canvas.drawLine(const Offset(-6, -4), const Offset(-2, -4), paint);
        canvas.drawLine(const Offset(-2, -4), const Offset(-2, 4), paint);
        canvas.drawLine(const Offset(-2, 4), const Offset(3, 4), paint);
        canvas.drawLine(const Offset(3, 4), const Offset(3, 0), paint);
        canvas.drawLine(const Offset(3, 0), const Offset(7, 0), paint);
        canvas.drawCircle(const Offset(-6, -4), 1.5, paint);
        canvas.drawCircle(const Offset(7, 0), 1.5, paint);
        break;

      case HoloType.codeSnippet:
        _drawText(canvas, obj.label ?? 'code()', Offset.zero, color, obj.size);
        break;

      case HoloType.confidencePct:
        final pct = obj.label ?? '85%';
        _drawText(canvas, pct, Offset.zero, color, obj.size);
        paint.strokeWidth = 1.5;
        canvas.drawArc(Rect.fromCenter(center: Offset.zero, width: 14, height: 14), -pi / 2, pi * 2 * 0.83, false, paint);
        break;

      case HoloType.tradingStat:
        _drawText(canvas, obj.label ?? 'W:266', Offset.zero, color, obj.size);
        break;

      case HoloType.coordMarker:
        _drawText(canvas, obj.label ?? '(0,0)', Offset.zero, color, obj.size);
        break;

      case HoloType.signalDot:
        paint.style = PaintingStyle.fill;
        canvas.drawCircle(Offset.zero, 2.5, paint..color = Colors.red.withValues(alpha: color.opacity * 0.6));
        paint.style = PaintingStyle.stroke;
        canvas.drawCircle(Offset.zero, 3.5, paint..color = color);
        if (obj.label != null) {
          _drawText(canvas, obj.label!, const Offset(5, -4), color, 7);
        }
        break;

      case HoloType.rotatingHexagon:
        final path = Path();
        for (int i = 0; i < 6; i++) {
          final a = pi / 3 * i - pi / 6;
          final px2 = cos(a) * 8;
          final py2 = sin(a) * 8;
          if (i == 0) path.moveTo(px2, py2);
          else path.lineTo(px2, py2);
        }
        path.close();
        canvas.drawPath(path, paint);
        canvas.drawLine(const Offset(-4, 0), const Offset(4, 0), paint..strokeWidth = 0.5);
        canvas.drawLine(const Offset(0, -4), const Offset(0, 4), paint);
        break;

      case HoloType.scanCrosshair:
        paint.strokeWidth = 0.8;
        canvas.drawLine(const Offset(-12, 0), const Offset(-3, 0), paint);
        canvas.drawLine(const Offset(3, 0), const Offset(12, 0), paint);
        canvas.drawLine(const Offset(0, -12), const Offset(0, -3), paint);
        canvas.drawLine(const Offset(0, 3), const Offset(0, 12), paint);
        canvas.drawCircle(Offset.zero, 4, paint);
        canvas.drawCircle(Offset.zero, 2, paint..style = PaintingStyle.fill..color = color.withValues(alpha: color.opacity * 0.3));
        break;
    }
  }

  void _drawText(Canvas canvas, String text, Offset offset, Color color, double size) {
    final tp = TextPainter(
      text: TextSpan(text: text, style: TextStyle(color: color, fontSize: size, fontFamily: 'monospace')),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, offset);
  }

  @override
  bool shouldRepaint(_HoloPainter old) => old.t != t;
}
