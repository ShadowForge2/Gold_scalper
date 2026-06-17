import 'package:flutter/material.dart';
import '../theme.dart';

class LogEntry {
  final String time;
  final String message;
  final String level;

  LogEntry({required this.time, required this.message, this.level = 'INFO'});

  factory LogEntry.fromJson(Map<String, dynamic> json) {
    return LogEntry(
      time: json['time'] ?? '',
      message: json['message'] ?? '',
      level: json['level'] ?? 'INFO',
    );
  }

  bool get isError => level == 'ERROR';
  bool get isWarning => level == 'WARNING';
  bool get isSignal => message.contains('[SIGNAL]');
  bool get isTrade => message.contains('[TRADE]');
  bool get isBias => message.contains('[BIAS]');
}

class TerminalLog extends StatefulWidget {
  final List<LogEntry> logs;
  final double height;

  const TerminalLog({super.key, required this.logs, this.height = 180});

  @override
  State<TerminalLog> createState() => _TerminalLogState();
}

class _TerminalLogState extends State<TerminalLog> {
  final ScrollController _scrollCtrl = ScrollController();

  @override
  void didUpdateWidget(TerminalLog oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.logs.length > oldWidget.logs.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scrollCtrl.hasClients) {
          _scrollCtrl.animateTo(
            _scrollCtrl.position.maxScrollExtent,
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  @override
  void dispose() {
    _scrollCtrl.dispose();
    super.dispose();
  }

  Color _colorFor(LogEntry entry) {
    if (entry.isError) return const Color(0xFFFF4444);
    if (entry.isWarning) return const Color(0xFFFFAA00);
    if (entry.isTrade) return const Color(0xFF66BB6A);
    if (entry.isSignal) return const Color(0xFF42A5F5);
    if (entry.isBias) return const Color(0xFFAB47BC);
    return const Color(0xFFCCCCCC);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: widget.height,
      decoration: BoxDecoration(
        color: const Color(0xFF0A0A0A),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF1A1A1A)),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: const BoxDecoration(
              color: Color(0xFF111111),
              border: Border(
                bottom: BorderSide(color: Color(0xFF1A1A1A)),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    color: Color(0xFF66BB6A),
                    boxShadow: [
                      BoxShadow(
                        color: Color(0x3366BB6A),
                        blurRadius: 6,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 6),
                const Text(
                  'TERMINAL',
                  style: TextStyle(
                    color: Color(0xFF66BB6A),
                    fontSize: 9,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5,
                    fontFamily: 'monospace',
                  ),
                ),
                Text(
                  '${widget.logs.length} entries',
                  style: const TextStyle(
                    color: Color(0xFF555555),
                    fontSize: 8,
                    fontWeight: FontWeight.w500,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: widget.logs.isEmpty
                ? const Center(
                    child: Text(
                      '~ no output ~',
                      style: TextStyle(
                        color: Color(0xFF444444),
                        fontSize: 11,
                        fontFamily: 'monospace',
                      ),
                    ),
                  )
                : ListView.builder(
                    controller: _scrollCtrl,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    itemCount: widget.logs.length,
                    itemBuilder: (context, i) {
                      final entry = widget.logs[i];
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 1),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              entry.time,
                              style: const TextStyle(
                                color: Color(0xFF555555),
                                fontSize: 10,
                                fontFamily: 'monospace',
                              ),
                            ),
                            const SizedBox(width: 6),
                            Expanded(
                              child: Text(
                                entry.message,
                                style: TextStyle(
                                  color: _colorFor(entry),
                                  fontSize: 10,
                                  fontFamily: 'monospace',
                                  height: 1.3,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
