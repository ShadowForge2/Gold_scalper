import 'package:flutter/services.dart';

VoidCallback hapt(VoidCallback cb) {
  return () {
    HapticFeedback.lightImpact();
    cb();
  };
}

void Function(int) haptInt(void Function(int) cb) {
  return (int i) {
    HapticFeedback.lightImpact();
    cb(i);
  };
}
