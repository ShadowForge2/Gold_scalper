class TradingSymbol {
  final String pair;
  final String price;
  final String change;
  final bool isPositive;

  const TradingSymbol({
    required this.pair,
    required this.price,
    required this.change,
    required this.isPositive,
  });
}
