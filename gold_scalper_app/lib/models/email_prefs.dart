class EmailPrefs {
  final bool configured;
  final String? email;
  final bool allowEmail;
  final bool allowPush;
  final bool allowMarketing;
  final bool isNew;

  const EmailPrefs({
    this.configured = false,
    this.email,
    this.allowEmail = false,
    this.allowPush = true,
    this.allowMarketing = false,
    this.isNew = true,
  });

  factory EmailPrefs.fromJson(Map<String, dynamic> json) {
    return EmailPrefs(
      configured: json['configured'] == true,
      email: json['email'] as String?,
      allowEmail: json['allow_email'] == true,
      allowPush: json['allow_push'] != false,
      allowMarketing: json['allow_marketing'] == true,
      isNew: json['is_new'] == true,
    );
  }

  EmailPrefs copyWith({
    bool? configured,
    String? email,
    bool? allowEmail,
    bool? allowPush,
    bool? allowMarketing,
    bool? isNew,
  }) {
    return EmailPrefs(
      configured: configured ?? this.configured,
      email: email ?? this.email,
      allowEmail: allowEmail ?? this.allowEmail,
      allowPush: allowPush ?? this.allowPush,
      allowMarketing: allowMarketing ?? this.allowMarketing,
      isNew: isNew ?? this.isNew,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'configured': configured,
      'email': email,
      'allow_email': allowEmail,
      'allow_push': allowPush,
      'allow_marketing': allowMarketing,
      'is_new': isNew,
    };
  }
}
