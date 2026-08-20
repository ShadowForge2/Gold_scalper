import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from app import database as db_mod

logger = logging.getLogger("GoldScalper")

# ── Resend configuration ────────────────────────────────────────────
# Everything is dormant until at least one Resend API key + a verified
# sending domain (RESEND_FROM_EMAIL) are present in the environment.
#
# ROTATING KEYS: multiple keys are supported via RESEND_API_KEYS as a
# comma-separated list, e.g.
#   RESEND_API_KEYS=re_aaa...,re_bbb...,re_ccc...
# Sends rotate round-robin across every key, and on a rate-limit / server
# failure the next key in the list is tried before giving up. The legacy
# single-key RESEND_API_KEY still works (and is used when RESEND_API_KEYS
# is absent).
RESEND_API_KEYS = [
    k.strip() for k in os.getenv("RESEND_API_KEYS", "").split(",") if k.strip()
]
_legacy_resend_key = os.getenv("RESEND_API_KEY", "").strip()
if RESEND_API_KEYS and not _legacy_resend_key:
    _resend_keys = list(RESEND_API_KEYS)
elif not RESEND_API_KEYS and _legacy_resend_key:
    _resend_keys = [_legacy_resend_key]
elif _legacy_resend_key:
    _resend_keys = list(RESEND_API_KEYS) + [_legacy_resend_key]
else:
    _resend_keys = []

RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "QuantoraFX <onboarding@resend.dev>")
RESEND_BASE = "https://api.resend.com/emails"

# Brand logo used in every email header. Served by the backend itself at
# /static/logo.png (PUBLIC_BASE_URL points at this deployment); override with
# BRAND_LOGO_URL for a fully external logo URL.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://gold-scalper-qyhg.onrender.com")
BRAND_LOGO_URL = os.getenv("BRAND_LOGO_URL", f"{PUBLIC_BASE_URL.rstrip('/')}/static/logo.png")

# Round-robin state. Sends rotate through _resend_keys; a key that fails
# with a retryable error is penalized (skipped) until _KEY_PENALTY_SEC
# elapses. _KEY_FAILURES keeps the penalty timestamp per key index.
_resend_key_lock = asyncio.Lock()
_resend_key_index = 0
_KEY_PENALTY_SEC = 300.0
_KEY_FAILURES: Dict[int, float] = {}


def _key_available(idx: int, now: float) -> bool:
    pen = _KEY_FAILURES.get(idx, 0.0)
    return now - pen >= _KEY_PENALTY_SEC


async def _next_resend_key() -> Optional[str]:
    """Return the next available Resend API key (round-robin)."""
    global _resend_key_index
    keys = _resend_keys
    if not keys:
        return None
    now = __import__("time").time()
    async with _resend_key_lock:
        for _ in range(len(keys)):
            idx = _resend_key_index % len(keys)
            _resend_key_index += 1
            if _key_available(idx, now):
                return keys[idx]
        # Every key is under penalty — fall back to plain round-robin anyway.
        idx = (_resend_key_index - 1) % len(keys)
        return keys[idx]


async def _mark_key_failure(key: str):
    try:
        idx = _resend_keys.index(key)
        _KEY_FAILURES[idx] = __import__("time").time()
    except ValueError:
        pass

# Email throttle windows per (identifier, email_type). Mirrors the push
# notification model: messages are mirrored to email only when the user
# has granted email permission, and are rate-limited so we never spam.
# Trial communication is milestone-driven (14 days / 7 days / expired) by the
# scheduler, so its cooldown is only a safety net against duplicate sends.
_EMAIL_TYPE_COOLDOWNS = {
    "billing_fee_due": timedelta(hours=24),
    "billing_trial_ending": timedelta(hours=24),
    "billing_trial_expired": timedelta(days=30),  # the "your trial has ended" notice fires once
    "billing_payment_received": timedelta(hours=1),
    "billing_welcome": timedelta(hours=24),
    "trade_alert": timedelta(minutes=15),
    "daily_pnl": timedelta(hours=23),
    "promo": timedelta(days=14),  # at most one promo every 2 weeks
}

_EMAIL_TYPE_DEFAULT = timedelta(hours=1)


def _enabled() -> bool:
    return bool(_resend_keys and RESEND_FROM_EMAIL)


def email_configured() -> bool:
    """True when Resend keys + a from-address are set (emails can be sent)."""
    return _enabled()


# ── Email permission store ──────────────────────────────────────────

async def get_email_prefs(identifier: str) -> Dict:
    row = await db_mod.fetch_one(
        "SELECT allow_email, allow_push, allow_marketing, updated_at "
        "FROM email_prefs WHERE identifier = :id",
        {"id": identifier},
    )
    if row is None:
        return {
            "allow_email": False,
            "allow_push": True,
            "allow_marketing": False,
            "updated_at": None,
        }
    return {
        "allow_email": bool(row["allow_email"]),
        "allow_push": bool(row["allow_push"]),
        "allow_marketing": bool(row["allow_marketing"]),
        "updated_at": row["updated_at"],
    }


async def set_email_prefs(identifier: str, allow_email: bool, allow_push: bool, allow_marketing: bool):
    await db_mod.execute(
        """INSERT INTO email_prefs (identifier, allow_email, allow_push, allow_marketing, updated_at)
           VALUES (:id, :ae, :ap, :am, :ua)
           ON CONFLICT (identifier)
           DO UPDATE SET allow_email = :ae2, allow_push = :ap2,
                         allow_marketing = :am2, updated_at = :ua2""",
        {
            "id": identifier,
            "ae": int(allow_email),
            "ap": int(allow_push),
            "am": int(allow_marketing),
            "ua": datetime.utcnow().isoformat(),
            "ae2": int(allow_email),
            "ap2": int(allow_push),
            "am2": int(allow_marketing),
            "ua2": datetime.utcnow().isoformat(),
        },
    )


async def grant_first_connection_prefs(identifier: str, allow_email: bool, allow_push: bool, allow_marketing: bool = False):
    """Merge with any existing preferences (first-time consent is additive)."""
    cur = await get_email_prefs(identifier)
    await set_email_prefs(
        identifier,
        allow_email=allow_email or cur["allow_email"],
        allow_push=allow_push or cur["allow_push"],
        allow_marketing=allow_marketing or cur["allow_marketing"],
    )


# ── Send throttle (email_log table) ─────────────────────────────────

async def _last_sent(identifier: str, email_type: str) -> Optional[datetime]:
    row = await db_mod.fetch_one(
        "SELECT sent_at FROM email_log WHERE identifier = :id AND email_type = :t ORDER BY sent_at DESC LIMIT 1",
        {"id": identifier, "t": email_type},
    )
    if row is None or not row.get("sent_at"):
        return None
    try:
        return datetime.fromisoformat(row["sent_at"])
    except Exception:
        return None


async def _record_sent(identifier: str, email_type: str):
    try:
        await db_mod.execute(
            "INSERT INTO email_log (identifier, email_type, sent_at) VALUES (:id, :t, :sa)",
            {"id": identifier, "t": email_type, "sa": datetime.utcnow().isoformat()},
        )
    except Exception as e:
        logger.debug("email_log insert failed: %s", e)


async def _can_send(identifier: str, email_type: str) -> bool:
    if not _enabled():
        return False
    last = await _last_sent(identifier, email_type)
    if last is None:
        return True
    cooldown = _EMAIL_TYPE_COOLDOWNS.get(email_type, _EMAIL_TYPE_DEFAULT)
    return datetime.utcnow() - last >= cooldown


async def send_email(identifier: str, email_type: str, subject: str, html: str, text: str = "", marketing: bool = False) -> bool:
    """Send a transactional email to the account's Capital.com-registered email.

    Rotates round-robin across every configured Resend key and retries with
    the next key on rate-limit / server errors. Returns True only when an
    email was actually dispatched (permission granted AND outside the type
    throttle window).
    """
    if not _enabled():
        return False
    prefs = await get_email_prefs(identifier)
    if not prefs["allow_email"]:
        return False
    if marketing and not prefs["allow_marketing"]:
        return False
    if not await _can_send(identifier, email_type):
        return False

    to_addr = identifier
    payload = {
        "from": RESEND_FROM_EMAIL,
        "to": [to_addr],
        "subject": subject,
        "html": html,
    }
    if text:
        payload["text"] = text

    # Try up to len(_resend_keys) distinct keys (round-robin + failover).
    attempts = len(_resend_keys)
    tried_keys: List[str] = []
    for attempt in range(attempts):
        api_key = await _next_resend_key()
        if api_key is None:
            return False
        if api_key in tried_keys:
            continue
        tried_keys.append(api_key)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    RESEND_BASE,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
            if r.is_success:
                await _record_sent(identifier, email_type)
                logger.info("Email sent: %s -> %s (%s)", email_type, to_addr, subject)
                return True
            # Any non-success tries the NEXT key (a revoked/rate-limited key
            # must not prevent the rest of the rotation from working).
            logger.warning("Resend returned %s for %s (attempt %d/%d): %s",
                           r.status_code, to_addr, attempt + 1, attempts, r.text[:200])
            await _mark_key_failure(api_key)
            continue
        except Exception as e:
            logger.warning("Email send failed (%s for %s, attempt %d/%d): %s",
                           email_type, to_addr, attempt + 1, attempts, e)
            await _mark_key_failure(api_key)
            continue
    return False


# ── HTML templates ──────────────────────────────────────────────────

def _shell(inner_html: str, title: str = "QuantoraFX") -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title></head>
<body style="margin:0;padding:0;background:#020408;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#020408;padding:32px 16px;">
<tr><td align="center">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#0A0C14;border:1px solid #1E2433;border-radius:16px;overflow:hidden;">
<tr><td style="padding:32px 32px 24px;text-align:center;">
<img src="{BRAND_LOGO_URL}" width="96" height="96" alt="QuantoraFX" style="display:inline-block;border:0;outline:none;text-decoration:none;border-radius:50%;background:#0A0C14;"/>
<div style="color:#FFD700;font-size:20px;font-weight:900;letter-spacing:3px;margin-top:14px;">QUANTORAFX</div>
<div style="color:#94A3B8;font-size:11px;letter-spacing:2px;margin-top:6px;">AI-POWERED TRADING AUTOMATION</div>
</td></tr>
<tr><td style="height:1px;background:#1E2433;font-size:0;line-height:0;">&nbsp;</td></tr>
<tr><td style="padding:28px 32px;color:#F1F5F9;font-size:14px;line-height:1.7;">{inner_html}</td></tr>
<tr><td style="height:1px;background:#1E2433;font-size:0;line-height:0;">&nbsp;</td></tr>
<tr><td style="padding:20px 32px;text-align:center;color:#475569;font-size:11px;line-height:1.6;">
You are receiving this because you connected a Capital.com account to QuantoraFX.<br>
Manage your preferences anytime in the app under Settings &rarr; Notifications &amp; Email.
</td></tr>
</table></td></tr></table></body></html>"""


def _btn(href: str, label: str) -> str:
    return (
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="margin:24px auto;">'
        f'<tr><td style="border-radius:10px;background:#FFD700;box-shadow:0 4px 14px rgba(255,215,0,0.25);">'
        f'<a href="{href}" style="display:inline-block;padding:13px 30px;color:#000;font-weight:800;'
        f'text-decoration:none;font-size:14px;border-radius:10px;">{label}</a>'
        f'</td></tr></table>'
    )


def _li(text: str) -> str:
    return (
        f'<tr><td style="padding:5px 0 5px 14px;border-left:3px solid #FFD700;color:#F1F5F9;'
        f'font-size:13.5px;line-height:1.6;">{text}</td></tr>'
    )


def _money(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


# ── Template builders (billing / status) ────────────────────────────

def render_billing_welcome(identifier: str, trial_end: str) -> str:
    return _shell(
        f"""<p style="margin:0 0 14px;font-size:16px;">Welcome to <b style="color:#FFD700;">QuantoraFX</b> — your
        Capital.com account (<b>{identifier}</b>) is now connected to our AI trading engine.</p>
        <p style="margin:0 0 18px;">Your <b>30-day free trial</b> is live and runs until
        <b style="color:#FFD700;">{trial_end}</b>. Here&rsquo;s what happens now:</p>
        <table role="presentation" cellpadding="0" cellspacing="0" style="margin:0 0 18px;">
        {_li("Our bot scans the markets 24/7 for pull-back entries in the direction of the last completed H1 candle.")}
        {_li("Every position is managed automatically &mdash; entry, trailing stop, and exit.")}
        {_li("No profit, no fee &mdash; after the trial you pay only <b>15% of profit</b> per 30-day period.")}
        </table>
        <p style="margin:0;">Sit back &mdash; the automation is running.</p>""",
        title="Welcome to QuantoraFX",
    )


def render_billing_fee_due(identifier: str, amount: float, payment_url: str = "") -> str:
    body = (
        f'<p style="margin:0 0 14px;">Hi <b style="color:#FFD700;">{identifier}</b>,</p>'
        f'<p style="margin:0 0 16px;">Your QuantoraFX 30-day fee of '
        f'<b style="color:#FFD700;font-size:16px;">{_money(amount)}</b> is due.</p>'
        f'<p style="margin:0 0 16px;">Your bot keeps trading while this fee is outstanding, but settling it keeps '
        f'your service fully uninterrupted.</p>'
    )
    if payment_url:
        body += _btn(payment_url, "Pay Now")
    body += (
        f'<p style="margin:16px 0 0;">Once paid, your subscription is extended by <b>30 days</b> instantly.</p>'
    )
    return _shell(body, title="Your QuantoraFX fee is due")


def render_billing_trial_ending(identifier: str, days_left: int) -> str:
    return _shell(
        f"""<p style="margin:0 0 14px;">Hi <b style="color:#FFD700;">{identifier}</b>,</p>
        <p style="margin:0 0 16px;">Your <b style="color:#FFD700;">QuantoraFX trial</b> ends in
        <b>{days_left} day(s)</b>.</p>
        <p style="margin:0 0 16px;">After the trial, you pay only <b>15% of profit</b> each 30-day period.
        If the bot made no profit, there is nothing to pay.</p>
        <p style="margin:0;">Keep your subscription active so your automation never stops.</p>""",
        title="Your QuantoraFX trial is ending soon",
    )


def render_billing_trial_expired(identifier: str, payment_url: str = "") -> str:
    body = (
        f'<p style="margin:0 0 14px;">Hi <b style="color:#FFD700;">{identifier}</b>,</p>'
        f'<p style="margin:0 0 16px;">Your <b style="color:#FFD700;">QuantoraFX trial</b> has ended and your bot '
        f'is now paused.</p>'
        f'<p style="margin:0 0 16px;">Reactivate in one tap and pick up right where you left off &mdash; '
        f'your next 30 days of fully automated trading are waiting.</p>'
    )
    if payment_url:
        body += _btn(payment_url, "Reactivate Now")
    return _shell(body, title="Reactivate your QuantoraFX subscription")


def render_billing_payment_received(identifier: str, amount: float, subscription_end: str) -> str:
    return _shell(
        f"""<p style="margin:0 0 14px;">Hi <b style="color:#FFD700;">{identifier}</b>,</p>
        <p style="margin:0 0 16px;">We received <b style="color:#FFD700;font-size:16px;">{_money(amount)}</b>
        for your QuantoraFX subscription.</p>
        <p style="margin:0 0 16px;">Your subscription is now active until
        <b style="color:#FFD700;">{subscription_end}</b>. If your bot was paused because the subscription lapsed,
        it has automatically resumed.</p>
        <p style="margin:0;">Thank you for trading with <b style="color:#FFD700;">QuantoraFX</b>.</p>""",
        title="Payment received — thank you",
    )


# ── Template builders (trading / promos) ────────────────────────────

def render_trade_alert(identifier: str, title: str, message: str) -> str:
    return _shell(
        f"""<p style="margin:0 0 14px;font-size:16px;"><b style="color:#FFD700;">{title}</b></p>
        <p style="margin:0;">{message}</p>""",
        title=title,
    )


def render_daily_pnl(identifier: str, pnl: float, day_label: str, trades: int = 0, balance: float = 0.0) -> str:
    color = "#10B981" if pnl >= 0 else "#EF4444"
    trade_line = f"{trades} closed trade(s)" if trades else "no closed trades"
    bal_line = f"Balance: <b>{_money(balance)}</b>" if balance else ""
    return _shell(
        f"""<p style="margin:0 0 16px;">Your QuantoraFX recap for <b>{day_label}</b>:</p>
        <p style="margin:0 0 6px;font-size:12px;letter-spacing:1px;color:#94A3B8;">TODAY'S P&amp;L</p>
        <p style="margin:0 0 18px;font-size:36px;font-weight:900;color:{color};">{_money(pnl)}</p>
        <p style="margin:0 0 6px;color:#94A3B8;">{trade_line}</p>
        {('<p style="margin:0 0 6px;color:#94A3B8;">' + bal_line + '</p>') if bal_line else ''}
        <p style="margin:18px 0 0;color:#94A3B8;font-size:12px;">Automation runs 24/7. Keep your subscription active
        to never miss a session.</p>""",
        title="Your QuantoraFX daily recap",
    )


def render_promo(identifier: str, headline: str = "", body_text: str = "") -> str:
    headline = headline or "Opportunities come once in a while."
    body_text = body_text or (
        "Your QuantoraFX subscription has lapsed. Activate now and continue enjoying seamless, "
        "fully automated trading — the bot never sleeps, so you don't have to."
    )
    return _shell(
        f"""<p style="margin:0 0 16px;color:#FFD700;font-size:17px;font-weight:800;">{headline}</p>
        <p style="margin:0;">{body_text}</p>""",
        title="QuantoraFX",
    )


# ── High-level senders ──────────────────────────────────────────────

async def email_billing_welcome(identifier: str, trial_end: str):
    await send_email(identifier, "billing_welcome", "Welcome to QuantoraFX 🚀",
                     render_billing_welcome(identifier, trial_end))


async def email_billing_fee_due(identifier: str, amount: float, payment_url: str = ""):
    await send_email(identifier, "billing_fee_due", "Your QuantoraFX fee is due",
                     render_billing_fee_due(identifier, amount, payment_url))


async def email_billing_trial_ending(identifier: str, days_left: int):
    await send_email(identifier, "billing_trial_ending",
                     f"Your QuantoraFX trial ends in {days_left} day(s)",
                     render_billing_trial_ending(identifier, days_left))


async def email_billing_trial_expired(identifier: str, payment_url: str = ""):
    await send_email(identifier, "billing_trial_expired", "Reactivate your QuantoraFX subscription",
                     render_billing_trial_expired(identifier, payment_url))


async def email_billing_payment_received(identifier: str, amount: float, subscription_end: str):
    await send_email(identifier, "billing_payment_received", "Payment received — thank you",
                     render_billing_payment_received(identifier, amount, subscription_end))


async def email_trade_alert(identifier: str, title: str, message: str):
    await send_email(identifier, "trade_alert", title, render_trade_alert(identifier, title, message))


async def email_daily_pnl(identifier: str, pnl: float, day_label: str, trades: int = 0, balance: float = 0.0):
    await send_email(identifier, "daily_pnl", "Your QuantoraFX daily recap",
                     render_daily_pnl(identifier, pnl, day_label, trades, balance))


async def email_promo(identifier: str, headline: str = "", body_text: str = ""):
    await send_email(identifier, "promo", "QuantoraFX — why wait?",
                     render_promo(identifier, headline, body_text), marketing=True)
