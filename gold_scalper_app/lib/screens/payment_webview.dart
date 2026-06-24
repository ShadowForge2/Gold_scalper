import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

class PaymentWebView extends StatefulWidget {
  final String url;
  final String title;
  final VoidCallback? onSuccess;

  const PaymentWebView({
    super.key,
    required this.url,
    this.title = 'Payment',
    this.onSuccess,
  });

  @override
  State<PaymentWebView> createState() => _PaymentWebViewState();
}

class _PaymentWebViewState extends State<PaymentWebView> {
  WebViewController? _controller;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    final uri = Uri.tryParse(widget.url);
    if (uri == null || !uri.hasScheme || !uri.hasAuthority) {
      _loading = false;
      _error = 'Invalid payment link.';
      return;
    }
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setNavigationDelegate(NavigationDelegate(
        onPageStarted: (_) => setState(() => _loading = true),
        onPageFinished: (_) => setState(() => _loading = false),
        onWebResourceError: (error) {
          setState(() {
            _loading = false;
            _error = 'Payment page failed to load. Check your connection and try again.';
          });
        },
        onUrlChange: (change) {
          final url = change.url ?? '';
          if (url.contains('success') || url.contains('callback')) {
            widget.onSuccess?.call();
          }
        },
      ))
      ..loadRequest(uri);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          if (_loading)
            const Padding(
              padding: EdgeInsets.only(right: 16),
              child: SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
        ],
      ),
      body: _error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white70),
                ),
              ),
            )
          : WebViewWidget(controller: _controller!),
    );
  }
}
