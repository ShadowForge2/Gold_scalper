import 'package:flutter_test/flutter_test.dart';
import 'package:gold_scalper_app/main.dart';

void main() {
  testWidgets('App renders welcome screen', (WidgetTester tester) async {
    await tester.pumpWidget(const GoldScalperApp());
    await tester.pump();
    expect(find.byType(GoldScalperApp), findsOneWidget);
  });
}
