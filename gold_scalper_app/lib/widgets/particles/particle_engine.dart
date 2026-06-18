import 'dart:math';
import 'package:flutter/material.dart';
import '../../models/particle.dart';
import '../../painters/particle_painter.dart';

class ParticleEngine extends StatefulWidget {
  final double intensity;
  const ParticleEngine({super.key, this.intensity = 1.0});

  @override
  State<ParticleEngine> createState() => _ParticleEngineState();
}

class _ParticleEngineState extends State<ParticleEngine>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  final _particles = <Particle>[];
  final _rng = Random();
  var _initialized = false;
  double _w = 0;
  double _h = 0;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this, duration: const Duration(seconds: 30),
    )..repeat();
  }

  void _ensureParticles(double w, double h) {
    if (!_initialized || _w != w || _h != h) {
      _w = w;
      _h = h;
      _particles.clear();
      for (var i = 0; i < 220; i++) {
        _particles.add(Particle.random(_rng, _w, _h));
      }
      _initialized = true;
    }
  }

  void _update(double dt, double w, double h) {
    _ensureParticles(w, h);
    for (final p in _particles) {
      _updateParticle(p, dt, w, h);
    }
  }

  void _updateParticle(Particle p, double dt, double w, double h) {
    final speedFactor = p.speed * dt * 120;
    switch (p.type) {
      case 'star':
        final cx = w / 2, cy = h / 2;
        final dx = p.x - cx, dy = p.y - cy;
        final dist = sqrt(dx * dx + dy * dy);
        if (dist > 0) {
          p.x += (dx / dist) * speedFactor;
          p.y += (dy / dist) * speedFactor;
        }
        if (p.x < -50 || p.x > w + 50 || p.y < -50 || p.y > h + 50) {
          p.respawn(_rng, w, h);
        }
        break;
      case 'binary':
        p.y += speedFactor * 0.6;
        p.x += speedFactor * 0.25;
        p.rotation += 0.002;
        if (p.y > h + 30) p.respawn(_rng, w, h);
        break;
      case 'symbol':
        p.y -= speedFactor * 0.4;
        p.x += speedFactor * 0.15 * sin(dt * 3 + p.z);
        p.rotation += 0.003;
        if (p.y < -40) p.respawn(_rng, w, h);
        break;
      case 'chart':
        p.y += speedFactor * 0.15;
        p.x += speedFactor * 0.05;
        p.rotation += 0.001;
        if (p.y > h + 50) p.respawn(_rng, w, h);
        break;
      case 'neural':
        p.opacity *= 0.995;
        if (p.opacity < 0.01) p.respawn(_rng, w, h);
        break;
      case 'dust':
        p.x += sin(dt * 0.5 + p.z * 10) * 0.15;
        p.y += cos(dt * 0.3 + p.z * 15) * 0.15;
        break;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return LayoutBuilder(
            builder: (context, constraints) {
              final w = constraints.maxWidth;
              final h = constraints.maxHeight;
              _update(_controller.value, w, h);
              return Opacity(
                opacity: widget.intensity,
                child: CustomPaint(
                  size: Size(w, h),
                  painter: ParticlePainter(_particles),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
