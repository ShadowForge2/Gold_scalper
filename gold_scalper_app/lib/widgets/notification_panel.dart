import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/bot_provider.dart';
import '../models/notification_item.dart';
import '../theme.dart';

class NotificationPanel extends StatelessWidget {
  const NotificationPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<BotProvider>(
      builder: (context, bp, _) {
        final notifs = bp.notifications;
        return Container(
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.5,
          ),
          decoration: const BoxDecoration(
            color: Color(0xFF1A1A2E),
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                decoration: BoxDecoration(
                  border: Border(bottom: BorderSide(color: Colors.white.withValues(alpha: 0.06))),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.notifications_rounded, color: kGold, size: 18),
                    const SizedBox(width: 8),
                    const Text(
                      'Notifications',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const Spacer(),
                    if (bp.unreadCount > 0)
                      GestureDetector(
                        onTap: () => bp.markNotificationsRead(),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: kGold.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Text(
                            'Mark all read',
                            style: TextStyle(color: kGold, fontSize: 11, fontWeight: FontWeight.w600),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              Flexible(
                child: notifs.isEmpty
                    ? const Padding(
                        padding: EdgeInsets.all(40),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.notifications_off_outlined, color: Colors.white24, size: 40),
                            SizedBox(height: 12),
                            Text(
                              'No notifications yet',
                              style: TextStyle(color: Colors.white38, fontSize: 14),
                            ),
                            SizedBox(height: 4),
                            Text(
                              'Trade open/close alerts will appear here',
                              style: TextStyle(color: Colors.white24, fontSize: 12),
                            ),
                          ],
                        ),
                      )
                    : ListView.separated(
                        shrinkWrap: true,
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        itemCount: notifs.length,
                        separatorBuilder: (_, __) => const SizedBox(height: 6),
                        itemBuilder: (context, index) {
                          return _NotificationTile(notification: notifs[index]);
                        },
                      ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _NotificationTile extends StatelessWidget {
  final NotificationItem notification;

  const _NotificationTile({required this.notification});

  @override
  Widget build(BuildContext context) {
    final isTradeOpen = notification.type == 'trade_open';
    final isTradeClose = notification.type == 'trade_close';
    final isPositive = notification.data?['pnl'] != null
        ? (notification.data!['pnl'] as num) >= 0
        : null;

    IconData icon;
    Color iconColor;
    if (isTradeOpen) {
      icon = Icons.arrow_upward_rounded;
      iconColor = kSuccess;
    } else if (isTradeClose && isPositive == true) {
      icon = Icons.check_circle_rounded;
      iconColor = kSuccess;
    } else if (isTradeClose && isPositive == false) {
      icon = Icons.cancel_rounded;
      iconColor = kDanger;
    } else if (isTradeClose) {
      icon = Icons.swap_horiz_rounded;
      iconColor = kGold;
    } else {
      icon = Icons.info_outline_rounded;
      iconColor = kInfo;
    }

    final timeStr = _formatTime(notification.createdAt);

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: notification.isRead
            ? Colors.white.withValues(alpha: 0.03)
            : kGold.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: notification.isRead
              ? Colors.white.withValues(alpha: 0.04)
              : kGold.withValues(alpha: 0.12),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: iconColor.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: iconColor, size: 16),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        notification.title,
                        style: TextStyle(
                          color: notification.isRead ? Colors.white54 : Colors.white,
                          fontSize: 13,
                          fontWeight: notification.isRead ? FontWeight.normal : FontWeight.w600,
                        ),
                      ),
                    ),
                    Text(
                      timeStr,
                      style: const TextStyle(color: Colors.white24, fontSize: 10),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  notification.message,
                  style: TextStyle(
                    color: notification.isRead ? Colors.white38 : Colors.white60,
                    fontSize: 12,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
          if (!notification.isRead)
            Padding(
              padding: const EdgeInsets.only(left: 4, top: 4),
              child: Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: kGold,
                  shape: BoxShape.circle,
                ),
              ),
            ),
        ],
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    if (diff.inSeconds < 60) return 'now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m';
    if (diff.inHours < 24) return '${diff.inHours}h';
    return '${diff.inDays}d';
  }
}
