import 'package:flutter/material.dart';

class SessionIdScreen extends StatefulWidget {
  final void Function(String sessionId) onSubmit;

  const SessionIdScreen({super.key, required this.onSubmit});

  @override
  State<SessionIdScreen> createState() => _SessionIdScreenState();
}

class _SessionIdScreenState extends State<SessionIdScreen> {
  final _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Patient / Session ID')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: const InputDecoration(labelText: 'Session ID'),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                final trimmed = _controller.text.trim();
                if (trimmed.isNotEmpty) {
                  widget.onSubmit(trimmed);
                }
              },
              child: const Text('Continue to capture'),
            ),
          ],
        ),
      ),
    );
  }
}
