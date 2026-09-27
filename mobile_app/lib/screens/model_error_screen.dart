import 'package:flutter/material.dart';

class ModelErrorScreen extends StatelessWidget {
  final String message;

  const ModelErrorScreen({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text('Could not load the on-device model:\n$message',
                  textAlign: TextAlign.center),
            ],
          ),
        ),
      ),
    );
  }
}
