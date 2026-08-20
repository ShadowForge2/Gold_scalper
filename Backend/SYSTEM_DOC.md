# Gold Scalper — Multi-TF H1 Breakout + Peak Harvest Exit

> **Strategy**: H1 range breakout + M1 momentum entry with peak-harvest exit (no hard SL).  
> **Broker**: Capital.com REST API (primary), MT5 fallback.  
> **Exit philosophy**: No stop loss. Let trades breathe. Trail only after significant profit.  
>   Close when directional confidence breaks down. Capture peak profit per event.

---

## 1. Architecture

```
Capital.com REST API / MT5
    │
    ├── H1 candles ──→ BiasEngine ──→ Trend direction (BULLISH/BEARISH/NEUTRAL)
    │
    └── M1 candles ──→ SignalEngine ──→ Entry (H1 breakout + momentum)
                                          │
                                          └──→ Exit (peak harvest — no SL, late trail,
                                                    directional confidence decay)
                                                    │
                                                    └── Position closed, cooldown, re-enter
```

**Files**:
| File | Purpose |
|---|---|
| `.env` | All credentials & config |
| `config.py` | Env loader, all parameters |
| `main.py` | Entry point (uvicorn FastAPI) |
| `backtest.py` | Historical simulation (MT5/CAPITAL/YAHOO/DUKASCOPY) |
| `app/capital_client.py` | Capital.com REST API client |
| `app/mt5_client.py` | MT5 connection wrapper |
| `app/bot.py` | Main loop: bias → signal → entry → exit |
| `app/bias_engine.py` | H1 trend detection |
| `app/signal_engine.py` | Entry/exit signal generation (exit_mode=5 default) |
| `app/risk_manager.py` | EquityScaler + RiskManager |
| `app/trade_executor.py` | Order execution (delegates to client) |
| `app/position_manager.py` | Position tracking & PnL |
| `app/logger.py` | Logging |
| `app/api.py` | FastAPI dashboard |

---

## 2. BiasEngine (`app/bias_engine.py`)

### Input
- 96 H1 candles (last 4 days)

### Logic: EMA Cross + Swing Point Majority Vote

1. **EMA trend**: Fast EMA(20) vs Slow EMA(50) slope → ±1 vote
2. **Price vs Slow EMA**: Close above/below EMA(50) with slope → ±0.5 vote
3. **Swing points** (lookback=5): Count higher highs/lows vs lower highs/lows → normalized swing score added

**Decision**:
- ≥ 0.75 → BULLISH
- ≤ -0.75 → BEARISH
- else → NEUTRAL

**Strength** = `abs(votes) / 1.5`, capped at 1.0. Tradeable when strength ≥ 0.30.

---

## 3. SignalEngine (`app/signal_engine.py`)

### 3a. Entry — `evaluate(m1_data, bias, price, h1_high, h1_low)`

**Conditions** (ALL must pass):
1. Bias is BULLISH or BEARISH with strength ≥ 0.30
2. H1 range valid (h1_high > h1_low)
3. **Breakout**: BUY if price > h1_high, SELL if price < h1_low
4. **Recent range test** (optional): At least 1 of last 2-5 M1 candles was inside H1 range
5. **Minimum breakout distance**: `breakout_dist / range_size ≥ MIN_BREAKOUT_SCORE` (0.02)

**Score** = `min(breakout_dist / h1_range, 1.0)`, entry if ≥ SIGNAL_ENTRY_THRESHOLD (default 0.70)

### 3b. Exit — `evaluate_exit()` — Peak Harvest (exit_mode=5)

The peak harvest exit has **no hard stop loss**. It trusts the entry signal and lets the trade develop:

| Condition | Action | Reason |
|-----------|--------|--------|
| Normal price fluctuation | Hold | No SL — trust signal |
| Peak profit < ATR × 2.0 | Hold | Not enough profit to trail yet |
| Peak profit ≥ trigger × patience, retrace > 50% | Close | `trail_stop` — locked in peak profit |
| 3+ consecutive counter-directional candles | Close | `direction_loss` — conviction broken |
| Bars held ≥ 10, momentum decay > 0.85 | Close | `momentum_decay` — momentum exhausted |
| Bars held > 48 (safety) | Close | `max_hold` — emergency exit |

**Patience factor**: Higher entry score → more patience before trail activates.  
Formula: `trail_trigger = ATR × PEAK_HARVEST_TRAIL_TRIGGER × (1.0 + (1.0 - entry_score))`

**Other exit modes available** (for backtest comparison):
- Mode 1: Trailing SL + anti-candle + momentum decay (original tight exit)
- Mode 4: ATR-based stop + double counter-candle (for H1-only data)

---

## 4. Configuration (`.env` / `config.py`)

### Core Trading
| Parameter | Default | Description |
|---|---|---|
| `BROKER` | CAPITAL | "CAPITAL" or "MT5" |
| `SYMBOL` | XAUUSD | Trading symbol |
| `LOT_SIZE` | 0.01 | Base micro lot |
| `LOT_MULTIPLIER` | 5 | Multiply scaler lot by this |
| `MIN_LOT` | 0.01 | Minimum lot |
| `MAX_LOT` | 1.0 | Maximum lot |
| `MIN_BALANCE` | 20.0 | Min balance to start trading |

### Entry
| Parameter | Default | Description |
|---|---|---|
| `SIGNAL_ENTRY_THRESHOLD` | 0.70 | Min signal score for entry |
| `MIN_BREAKOUT_SCORE` | 0.02 | Min breakout distance / H1 range |
| `REQUIRE_RECENT_PULLBACK` | false | Require recent candle inside H1 range |
| `BIAS_STRENGTH_MIN` | 0.30 | Min bias strength to trade |

### Peak Harvest Exit (mode 5)
| Parameter | Default | Description |
|---|---|---|
| `PEAK_HARVEST_TRAIL_TRIGGER` | 2.0 | ATR multiple to activate trailing (higher = later) |
| `PEAK_HARVEST_TRAIL_RETRACE` | 0.50 | Retrace % of peak before trail closes |
| `PEAK_HARVEST_MIN_BARS_EXIT` | 10 | Min bars held before momentum-based exit |
| `PEAK_HARVEST_MOMENTUM_THRESHOLD` | 0.85 | Momentum decay threshold to exit |
| `PEAK_HARVEST_MAX_HOLD_BARS` | 48 | Max bars held (safety exit) |

### Risk Controls
| Parameter | Default | Description |
|---|---|---|
| `MAX_DAILY_LOSS_USD` | 2.00 | Hard daily stop (stops bot, does not close trades) |
| `MAX_EVENT_LOSS_USD` | 1.00 | Per-event loss limit (closes all positions) |
| `MAX_TRADES_PER_EVENT` | 1 | Max positions per event |
| `MAX_TRADES_PER_SESSION` | 3 | Max trades per session/day |
| `MAX_CONSECUTIVE_LOSSES` | 2 | Max consecutive losses before stop |
| `RE_ENTRY_COOLDOWN_SEC` | 600 | Cooldown after exit (× consecutive losses) |
| `MAX_SPREAD_PIPS` | 35.0 | Max spread to allow entry |
| `MAX_DRIFT_XAUUSD` | 0.50 | Max price drift from signal (USD) |
| `MAX_DRIFT_US100` | 5.00 | Max price drift from signal (points) |
| `ALLOWED_SESSIONS` | LONDON,NEW_YORK | Sessions to trade |

---

## 5. Bot Loop (`app/bot.py`)

```
IDLE → (bias updated every 60s) → AWAITING_SIGNAL → ENTERING → IN_TRADE
  ↑                                                           │
  └──────────────────── COOLDOWN ←────────────────────────────┘
```

**Tick cycle (~1s)**:
1. Refresh positions & PnL via client
2. Check subscription status (every 30s)
3. Check daily loss limit
4. Reconnect on connection loss (exponential backoff 2^n, max 60s)
5. Market hours check (Sun 23:00 UTC - Fri 22:00 UTC)
6. State dispatch:
   - **IN_TRADE**: Evaluate exit via peak harvest (mode 5). Check event loss limit.
   - **COOLDOWN**: Wait until cooldown expires, then → IDLE
   - **IDLE/AWAITING_SIGNAL**: Update H1 bias (every 60s), fetch 50 M1 candles + H1 range, check entry
7. On entry: `lot = scaler.get_lot(balance) × LOT_MULTIPLIER`, open 1+ trades, margin check
8. On exit: close all positions, cooldown = `RE_ENTRY_COOLDOWN_SEC × (1 + consecutive_losses)`

---

## 6. Position Sizing

`EquityScaler`:
```
lot = BASE_LOT (0.01) × (balance / 20.0)     # scales with equity
lot = min(lot, balance / 5000)                 # equity cap
lot = lot × LOT_MULTIPLIER                     # e.g. 5×
```

**Tiers** (balance-based):
| Balance | Tier | Trade Mult | Max Trades |
|---------|------|------------|------------|
| < $50   | 1    | 1.0×       | 1          |
| $50-100 | 2    | 1.5×       | 2          |
| ≥ $100  | 3    | 2.0×       | 3          |

**Trades per event** = `base_trades × tier_mult × confidence_mult`
- Confidence mult: 2.0 if score ≥ 0.85, 1.5 if ≥ 0.75, else 1.0

---

## 7. Key Properties

1. **Breakout-based**: Only trades when price breaks H1 range with bias alignment
2. **No hard stop loss**: Lets trades breathe through pullbacks — trusts high-confidence signals
3. **Late trailing**: Only activates after significant profit (≥2 ATR peak)
4. **Peak harvesting**: Closes at/near peak profit when directional confidence breaks
5. **Self-correcting**: Event loss limit ($1), daily loss limit ($2), cooldown after losses
6. **All-session**: Trades LONDON, NEW_YORK (ASIA optional, wider spreads)
7. **Broker-agnostic**: Same strategy via Capital.com REST API or MT5

---

## 8. Capital.com API Notes

- Auth: `POST /api/v1/session` → CST + X-SECURITY-TOKEN headers
- Session expires after ~10min inactivity (auto-refresh on 401)
- Prices: `GET /api/v1/prices/{epic}` (epic in path, not query)
- Range query broken (400), count-based fallback (max=1000, paginate with `to`)
- XAUUSD epic = "GOLD", decimalPlacesFactor=2
- Demo: `demo-api-capital.backend-capital.com`
- Live: `api-capital.backend-capital.com`

---

## 9. Broker Switching

```env
BROKER=CAPITAL   # Capital.com REST API (no MT5 needed)
BROKER=MT5       # Requires MetaTrader5 terminal
```

`trade_executor.py` delegates to whichever client is active.

---

## 10. Gemini AI Integration

An optional **Google Gemini API** advisory layer (`app/gemini_advisor.py`) can be enabled to provide AI-driven oversight on trading decisions.

### Configuration (`.env`)
| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | `""` | Google Gemini API key |
| `GEMINI_ENABLED` | `false` | Enable/disable Gemini advisor |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `GEMINI_ADVICE_WEIGHT` | `1.0` | Weight multiplier for advice |

### How it works
- **Entry gate**: After the mechanical signal engine detects a valid setup (score >= 0.85), the current market context (bias, score, ATR, spread, momentum, consecutive losses, session) is sent to Gemini. It returns a JSON decision — `"proceed"` (trade as planned), `"skip"` (block the trade), or `"caution"` (adjust take-profit targets). If Gemini fails or is disabled, the trade proceeds on mechanical rules alone.
- **Exit observer**: While in a trade, Gemini is periodically consulted with live PnL, run duration, and momentum data. Its advice is logged for reference but does not trigger any action — all exits are determined by the mechanical exit engine.
- Gemini is **disabled by default** and requires explicit opt-in via `GEMINI_ENABLED=true` and a valid API key.

## 11. Dependencies

```
requests          # Capital.com REST API
pandas            # OHLC data
numpy             # Numerical ops
python-dotenv     # .env loading
uvicorn           # FastAPI server
fastapi           # Dashboard
google-generativeai # Gemini API
yfinance          # Backtest data (optional, Yahoo)
MetaTrader5       # Backtest data (optional, MT5)
```
