# QuantoraFX — AI-Powered Forex Scalping Bot

> Hackathon portfolio / project submission
> Project owner: Agni Kai (Fire Star studio)

---

## 1. Overview

QuantoraFX is a fully automated, AI-driven forex/CFD scalping bot that trades
gold (XAUUSD), US equity indices (US100, US30) and more on **Capital.com** via
its REST API — 24/7, with institutional-grade risk management.

It is a complete product, not just an algorithm: live trading engine, honest
backtesting pipeline, ML signal filtering, a Flutter mobile app, subscription
billing, push/email notifications, and a production deployment with automatic
failover.

**Deployed live**: `https://gold-scalper-qyhg.onrender.com`
**Broker**: Capital.com (demo + live, MT5 removed)
**Stack**: Python (FastAPI) + Flutter + Render

---

## 2. The Problem

- Retail forex trading is dominated by emotion: fear, greed, overtrading.
- Most "automated" bots are black boxes that never survive honest backtesting
  (fill assumptions, ignored costs, look-ahead bias).
- Existing tools charge high fees, offer no transparency, and cannot adapt to
  changing market conditions.

## 3. The Solution

A quantitive trading system where **every edge is quantified before it touches
real money**:

1. **Honest backtesting pipeline** — bar-close fills only, real spreads +
   commissions from day one. Strategies that only "win" under ideal fills are
   rejected. This discipline removed several appealing-but-fake strategies.
2. **Live engine that exactly mirrors the backtest** — the same state machine
   that is validated on history runs live (verified trade-for-trade on real
   data), so backtest P&L is what live P&L should be.
3. **Adaptive, multi-symbol** — one bot scans the whole market board, ranks
   every pair with the same proven edge, and trades the strongest setup first.

---

## 4. Current Strategy (LIVE): Pull-into-H1 Scalper

The only strategy that survived honest backtests (bar-close fills + real costs)
is live. It trades the **M5 pullback in the direction of the last completed H1
candle body**, trailing a giveback fraction of the wave, and force-closing at a
fixed horizon.

**Out-of-sample validation 2026 (Profit Factor ≥ 1.3):**

| Pair   | Pull | Trail | Hold | PF train | PF valid | PF OOS |
|--------|------|-------|------|----------|----------|--------|
| US30   | 0.30 | 0.35  | 24   | 1.73     | 1.83     | **1.81** |
| XAUUSD | 0.30 | 0.15  | 12   | 1.22     | 1.96     | **1.40** |
| US100  | 0.30 | 0.50  | 6    | 1.38     | 1.86     | **1.34** |

Engine: `app/pull_h1_scalper.py` — a persistent per-symbol state machine that
replicates the backtest trade-for-trade.

**Profit lock-in**: a live trade never gives back more than 30% of the peak
profit it has touched (`PULL_GIVEBACK_CAP`); a single "sudden pump" M5 candle
exits at the tip (`PULL_PUMP_ATR`).

### Whole-board scanning
- The bot pulls the **full ~4000-market Capital.com board** and screens it for
  tradability (status, spread, volatility band).
- A **universal pull scanner** (`app/universal_pull_scanner.py`) ranks the
  universe by the proven pull-into-H1 edge and trades the strongest setup.
- Pairs without hand-tuned params **self-calibrate** their pull/trail/hold from
  their own M5/H1 structure (`app/pull_auto_tune.py`).

### Daily "profit bot" guards
Per symbol (UTC rollover): stop new entries once the day's net R hits a target
(lock in the day) or a max loss (stop the bleed). Open positions are never
force-closed by these guards.

---

## 5. Risk Management

- **Per-symbol state machines** — XAUUSD and US100 trade independently with no
  cross-symbol race conditions.
- **Event loss as % of balance** (default 5%), with **correlated-symbol
  grouping** (US100/US500/US30 share one budget — a single US-equity drawdown
  can't hit both).
- **Volatility-regime filters** — during high vol: 50% lot reduction, tight
  filters; skip a regime after N consecutive losses.
- **Global margin guard** — pauses all new entries when margin is insufficient,
  auto-resumes when free margin recovers.
- **Friday close gate** — no new positions in the last 60 min before the weekly
  close (existing positions are never orphaned).
- **News-aware trading** — economic calendar with pre/spike/post windows.
- Hard safety caps on lot multipliers (backtest-verified: over-leverage blows up
  a small account).

---

## 6. Machine Learning

- **XGBoost direction predictor** (trained on ~1.1M M5 bars, 26 features,
  ~75% directional accuracy) filters entries — only trades when ML agrees with
  the H1 bias at confidence ≥ threshold.
- The ML layer is optional (default on) and **fails closed** — any ML exception
  is caught and the trade proceeds on the mechanical rule, never crashes the
  bot.

---

## 7. Architecture

```
Capital.com REST API
    │
    ├── Whole-board scanner  ── screen ~4000 markets → top-K leaders
    │
    ├── Universal pull scanner ── rank by proven pull-into-H1 edge
    │
    └── Per-symbol PullPrevH1Scalper engines
            │
            ├── Enter on M5 pullback in H1 direction
            └── Exit on trailing giveback / max-hold / pump-tip
                    │
                    └── Position closed → daily guards updated → re-enter
```

**Backend** (`app/`):
| Module | Role |
|---|---|
| `bot.py` | Main loop, per-symbol state machine, entry/exit execution |
| `pull_h1_scalper.py` | Live pull-into-H1 strategy engine |
| `universal_pull_scanner.py` | Whole-universe ranking by the proven edge |
| `pull_auto_tune.py` | Per-symbol self-calibration |
| `pair_scanner.py` | Whole-board market screening (momentum pre-filter) |
| `capital_client.py` | Capital.com REST client (auth, prices, positions, margin) |
| `risk_manager.py` | EquityScaler, limits, spread/session filters |
| `position_manager.py` | Position tracking, recovery, PnL |
| `trade_executor.py` | Order execution |
| `subscription.py` | Trial + billing (Paystack, MaxelPay), notifications |
| `email_service.py` | Resend-powered transactional emails |
| `failover.py` | Primary/backup leader election (dual Render instances) |
| `api.py` | FastAPI dashboard API |

**Mobile app** (`gold_scalper_app/`): Flutter — dashboard, live feed, bot
controls, performance/equity charts, subscription, settings, onboarding.

**Website** (`website/`): landing page with download + legal pages.

**Deployment**: Render (uvicorn + FastAPI), PostgreSQL/SQLite DB, `render.yaml`,
two instances with automatic failover.

---

## 8. Backtesting & Validation Discipline

This is what separates the project from typical "backtest porn":

- **Bar-close fills only** — no intrabar fills at ideal levels. The Wave
  scalper showed PF 4.17 under "level" fills but was a **loser (PF 0.53,
  −286R) under honest market fills** — it was removed entirely. Only honest
  strategies stay live.
- **Real costs from day one** — live spread + commissions priced into every
  backtest and every live trade.
- **Out-of-sample validation** — train → validate → OOS windows; only pairs
  with OOS 2026 PF ≥ 1.3 are live.
- **Live == backtest** — verified trade-for-trade, R-for-R on real data before
  going live.

---

## 9. Business Model

- **Trial**: 7- or 14-day free trial per account.
- **Live billing**: 15% of profit per 30-day period (profit-sharing, not a flat
  fee) — aligns the product with the user.
- **Payments**: Paystack (cards, bank transfer) + MaxelPay (crypto), with
  USD→NGN conversion, webhooks with signature verification and idempotency.
- **Auto-restart after payment**: a subscription-lapse stop auto-resumes the bot
  when payment is received; a manual stop stays stopped.
- **Notifications**: Firebase Cloud Messaging push + Resend email (welcome,
  trial reminders at 14/7 days, fee-due, daily PnL recap, promos — throttled so
  users are never spammed).

---

## 10. Key Technologies

Python · FastAPI · asyncio · pandas · numpy · XGBoost · joblib · Capital.com
REST API · Flutter (Dart) · Firebase (FCM) · Paystack · MaxelPay · Resend ·
PostgreSQL / SQLite · Render · httpx · psutil

---

## 11. What Makes It Stand Out

1. **Transparency over hype** — every strategy is validated with honest fills
   and real costs; losing strategies are publicly removed, not hidden.
2. **Whole product, not a script** — trading engine + ML + mobile app + billing
   + notifications + failover deployment.
3. **Adaptive by design** — volatility regimes, per-symbol tuning, auto-
   calibration, daily profit guards.
4. **Engineered for small accounts** — starts at $10, lot sizes scale with
   equity, hard caps prevent over-leverage blowups.

---

## 12. Roadmap

- Broker expansion (MT5/MT4, OANDA, IBKR).
- Equity-index basket rebalancing for even lower drawdown.
- Multi-timeframe confluence (M15 confirmations on top of H1).
- Portfolio-level capital allocation across symbols.
- Live dashboard web app (beyond the mobile app).

---

*Past performance is not indicative of future results. Trading forex on margin
carries high risk of loss.*
