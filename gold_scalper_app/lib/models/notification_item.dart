import 'dart:convert';

class NotificationItem {
  final int id;
  final String type;
  final String title;
  final String message;
  final Map<String, dynamic>? data;
  final bool isRead;
  final DateTime createdAt;

  NotificationItem({
    required this.id,
    required this.type,
    required this.title,
    required this.message,
    this.data,
    required this.isRead,
    required this.createdAt,
  });

  factory NotificationItem.fromJson(Map<String, dynamic> json) {
    Map<String, dynamic>? parsedData;
    if (json['data'] != null && json['data'] is String) {
      try {
        parsedData = Map<String, dynamic>.from(jsonDecode(json['data']));
      } catch (_) {}
    } else if (json['data'] is Map) {
      parsedData = Map<String, dynamic>.from(json['data']);
    }

    return NotificationItem(
      id: json['id'] is int ? json['id'] : int.tryParse(json['id'].toString()) ?? 0,
      type: json['type'] ?? '',
      title: json['title'] ?? '',
      message: json['message'] ?? '',
      data: parsedData,
      isRead: json['is_read'] == 1 || json['is_read'] == true,
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
    );
  }
}
