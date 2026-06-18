enum SentimentCategory { boot, market, intelligence, trading, protection, aggressive }

enum SentimentZone { leftHud, rightHud, chest, leftArm, rightArm, lowerHud }

class AiMessage {
  final String text;
  final SentimentCategory category;
  final SentimentZone zone;
  final bool isCritical;
  final int priority;

  const AiMessage({
    required this.text,
    required this.category,
    this.zone = SentimentZone.leftHud,
    this.isCritical = false,
    this.priority = 0,
  });
}
