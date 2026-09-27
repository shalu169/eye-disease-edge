import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/screens/session_id_screen.dart';

void main() {
  testWidgets('submitting a non-empty ID calls onSubmit with trimmed value',
      (tester) async {
    String? submitted;
    await tester.pumpWidget(MaterialApp(
      home: SessionIdScreen(onSubmit: (id) => submitted = id),
    ));

    await tester.enterText(find.byType(TextField), '  patient-07  ');
    await tester.tap(find.byType(ElevatedButton));
    await tester.pump();

    expect(submitted, 'patient-07');
  });

  testWidgets('submit button is disabled for an empty ID', (tester) async {
    var called = false;
    await tester.pumpWidget(MaterialApp(
      home: SessionIdScreen(onSubmit: (id) => called = true),
    ));

    await tester.tap(find.byType(ElevatedButton));
    await tester.pump();

    expect(called, isFalse);
  });
}
