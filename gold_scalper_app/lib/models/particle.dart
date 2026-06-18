import 'dart:math';

class Particle {
  double x;
  double y;
  double z;
  double speed;
  double opacity;
  double rotation;
  double scale;
  String type;
  String? label;
  List<double>? chartData;
  double neuralAngle;
  double neuralLength;
  bool active;

  Particle({
    required this.x,
    required this.y,
    required this.z,
    required this.speed,
    required this.opacity,
    required this.rotation,
    required this.scale,
    required this.type,
    this.label,
    this.chartData,
    this.neuralAngle = 0,
    this.neuralLength = 0,
    this.active = true,
  });

  factory Particle.random(Random rng, double w, double h) {
    final type = _pickType(rng);
    final z = rng.nextDouble();
    return Particle(
      x: rng.nextDouble() * w,
      y: rng.nextDouble() * h,
      z: z,
      speed: _layerSpeed(type) * (0.5 + z * 1.5),
      opacity: _baseOpacity(type) * (0.3 + z * 0.7),
      rotation: rng.nextDouble() * 0.3 - 0.15,
      scale: 0.5 + z * 1.5,
      type: type,
      label: type == 'symbol'
          ? _symbols[rng.nextInt(_symbols.length)]
          : type == 'binary'
              ? _binarySegments[rng.nextInt(_binarySegments.length)]
              : null,
      chartData: type == 'chart' ? List.generate(8, (_) => rng.nextDouble()) : null,
      neuralAngle: type == 'neural' ? rng.nextDouble() * pi * 2 : 0,
      neuralLength: type == 'neural' ? 20 + rng.nextDouble() * 60 : 0,
    );
  }

  void respawn(Random rng, double w, double h) {
    type = _pickType(rng);
    z = rng.nextDouble();
    speed = _layerSpeed(type) * (0.5 + z * 1.5);
    opacity = _baseOpacity(type) * (0.3 + z * 0.7);
    scale = 0.5 + z * 1.5;

    final edge = rng.nextInt(4);
    x = edge == 0 ? -20 - rng.nextDouble() * 40
        : edge == 1 ? w + 20 + rng.nextDouble() * 40
        : rng.nextDouble() * w;
    y = edge == 2 ? -20 - rng.nextDouble() * 40
        : edge == 3 ? h + 20 + rng.nextDouble() * 40
        : rng.nextDouble() * h;

    label = type == 'symbol'
        ? _symbols[rng.nextInt(_symbols.length)]
        : type == 'binary'
            ? _binarySegments[rng.nextInt(_binarySegments.length)]
            : null;
    chartData = type == 'chart' ? List.generate(8, (_) => rng.nextDouble()) : null;

    if (type == 'neural') {
      neuralAngle = rng.nextDouble() * pi * 2;
      neuralLength = 20 + rng.nextDouble() * 60;
    }
    rotation = rng.nextDouble() * 0.3 - 0.15;
  }

  static String _pickType(Random rng) {
    final roll = rng.nextDouble();
    if (roll < 0.35) return 'star';
    if (roll < 0.50) return 'binary';
    if (roll < 0.65) return 'symbol';
    if (roll < 0.78) return 'chart';
    if (roll < 0.88) return 'neural';
    return 'dust';
  }

  static const _symbols = [
    'EURUSD', 'BTCUSD', 'XAUUSD', 'NASDAQ',
    'SP500', 'US30', 'GBPJPY', 'ETHUSD',
  ];

  static const _binarySegments = [
    '101101', '010011', '111000', '001011',
    '110101', '100110', '011100', '101010',
  ];

  static double _layerSpeed(String type) {
    switch (type) {
      case 'star': return 2.5;
      case 'binary': return 1.8;
      case 'symbol': return 1.2;
      case 'chart': return 0.6;
      case 'neural': return 0.3;
      case 'dust': return 0.15;
      default: return 1.0;
    }
  }

  static double _baseOpacity(String type) {
    switch (type) {
      case 'star': return 0.6;
      case 'binary': return 0.06;
      case 'symbol': return 0.08;
      case 'chart': return 0.05;
      case 'neural': return 0.04;
      case 'dust': return 0.03;
      default: return 0.1;
    }
  }
}
