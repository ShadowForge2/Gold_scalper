# Gold Scalper — Multi-TF H1 Breakout System

> **Strategy**: H1 range breakout + M1 momentum entry with ATR-based exit.  
> **Broker**: Capital.com REST API (primary), MT5 fallback.  
> **Backtest validation**: 2023 ($20→$64k), 2024 ($20→$79k), 2025 ($20→$30k), 2026 OOS ($20→$81).  
> **Live risk**: Aggressive scalper. Lot multiplier amplifies both wins and losses. Start small.

---

## 1. Architecture

```
Capital.com REST API / MT5
    │
    ├── H1 candles ──→ BiasEngine ──→ Trend direction (BULLISH/BEARISH/NEUTRAL)
    │
    └── M1 candles ──→ SignalEngine ──→ Entry (H1 breakout + momentum)
                                          │
                                          └──→ Exit (ATR trail / anti-candle / momentum decay)
                                                    │
                                                    └── Position closed, cooldown, re-enter
```

**Files**:
| File | Purpose |
|---|---|
| `.env` | All credentials & config |
| `config.py` | Env loader, all parameters |
| `main.py` | Entry point (uvicorn FastAPI) |
| `backtest.py` | Historical simulation (MT5/CAPITAL/YAHOO) |
| `optimize.py` | Parameter sweeper |
| `app/capital_client.py` | Capital.com REST API client |
| `app/mt5_client.py` | MT5 connection wrapper |
| `app/bot.py` | Main loop: bias → signal → entry → exit |
| `app/bias_engine.py` | H1 trend detection |
| `app/signal_engine.py` | Entry/exit signal generation |
| `app/risk_manager.py` | EquityScaler + RiskManager |
| `app/trade_executor.py` | Order execution (delegates to client) |
| `app/position_manager.py` | Position tracking & PnL |
| `app/logger.py` | Logging |
| `app/api.py` | FastAPI dashboard |

---

## 2. BiasEngine (`app/bias_engine.py`)

### Input
- 96 H1 candles (last 4 days)

### Logic: Swing Point Detection + Majority Vote

1. **Swing points**: HIGH where price ≥ both sides (`lookback=5`), LOW where price ≤ both sides
2. **Majority vote**:
   ```
   h_up = count of higher highs in last swings
   h_dn = count of lower highs
   l_up = count of higher lows
   l_dn = count of lower lows
   score = (h_up - h_dn) + (l_up - l_dn) / total
   BULLISH if score >= +0.30
   BEARISH if score <= -0.30
   ```
3. **Strength**: `0.3` if non-neutral, `0.0` otherwise
4. **Tradeable**: bias must be BULLISH/BEARISH with strength >= 0.3

---

## 3. SignalEngine (`app/signal_engine.py`)

### 3a. Entry — `evaluate(m1_data, bias, price, h1_high, h1_low)`

**Conditions** (ALL must pass):
1. Bias is BULLISH or BEARISH with strength >= 0.30
2. H1 high/low provided and valid (h1_high > h1_low)
3. **Breakout condition**:
   - BUY: price > h1_high (trading above H1 range)
   - SELL: price < h1_low (trading below H1 range)
4. **Recent range test**: at least 1 of the last 2-5 M1 candles was within the H1 range (confirms breakout just occurred)
5. **Minimum distance**: breakout >= 2% of H1 range

**Score** = min(breakout_distance / h1_range, 1.0), entry if >= 0.70

### 3b. Exit — `evaluate_exit()`

Exit mode depends on timeframe:
- **M5 entry (exit_mode=1)**: trailing SL + anti-candle + momentum decay
- **H1-only (exit_mode=4)**: ATR-based stop + double counter-candle + momentum decay

---

## 4. Configuration (`.env` / `config.py`)

| Parameter | Default | Description |
|---|---|---|
| `BROKER` | CAPITAL | "CAPITAL" or "MT5" |
| `CAPITAL_DEMO` | true | true=demo, false=live |
| `SIGNAL_ENTRY_THRESHOLD` | 0.70 | Min signal score for entry |
| `EXIT_THRESHOLD_TIGHT` | 0.50 | Momentum decay exit threshold |
| `LOT_MULTIPLIER` | 5 | Multiply scaler lot by this |
| `LOT_SIZE` | 0.01 | Base micro lot |
| `MAX_TRADES_PER_EVENT` | 10 | Max positions per event |
| `RE_ENTRY_COOLDOWN_SEC` | 60 | Cooldown after exit |
| `ALLOWED_SESSIONS` | ASIA,LONDON,NEW_YORK | Sessions to trade |
| `MAX_DAILY_LOSS_USD` | 10.00 | Hard daily stop |
| `MAX_EVENT_LOSS_USD` | 5.00 | Per-event stop |
| `BIAS_UPDATE_INTERVAL_SEC` | 60 | Bias refresh interval |

---

## 5. Bot Loop (`app/bot.py`)

```
IDLE → AWAITING_SIGNAL → (signal found?) → execute_entry → IN_TRADE
  ↑          ↑                                    │
  └──────────┘                                    ↓
                                             check_exit (every tick)
                                                  │
                                           exit condition met?
                                                  │
                                            yes → COOLDOWN → IDLE
```

**Tick cycle** (~2 sec):
1. Refresh positions & PnL via client
2. Check daily/event loss limits
3. If IN_TRADE: evaluate exit conditions
4. If IDLE/AWAITING_SIGNAL: update H1 bias every 60s, fetch M1 data + H1 range, check entry
5. On entry: `lot = scaler.get_lot(balance) * LOT_MULTIPLIER`, open trades
6. On exit: close all, set cooldown

**Capital.com-specific**: Session auto-refresh on 401, CST+X-SECURITY-TOKEN header auth

---

## 6. Position Sizing

`EquityScaler`:
```
lot = base_lot * (balance / starting_balance)
lot = min(lot, balance / 5000)  # equity cap
trades = base_trades * tier_mult * confidence_mult
total_lot = lot * trades * LOT_MULTIPLIER
```

Tiers: 1 (balance<50), 2 (50-100), 3 (100+)

Example with $20, 5x multiplier:
- Start: 0.01 * (20/20) = 0.01, capped at 20/5000 = 0.004 → 0.01 min_lot
- After scaling: trades × lot × 5

---

## 7. Backtest Results

### H1-only (Yahoo GC=F / MT5 XAUUSD)

| Year | Start | End | WR | PF | Max DD | Return |
|------|-------|-----|----|-----|--------|--------|
| 2023 | $20 | $64,249 | 78.1% | 10.92 | $1,258 | +321,147% |
| 2024 | $20 | $79,473 | 79.5% | 10.21 | $824 | +397,267% |
| 2025 | $20 | $30,657 | 68.0% | 7.10 | $1,046 | +166,131% |

### M5 entry OOS (Yahoo GC=F, Apr-Jun 2026)

| Start | End | WR | PF | Trades |
|-------|-----|----|-----|--------|
| $20 | $81.66 | 53.3% | 1.59 | 30 |

---

## 8. Key Properties

1. **Breakout-based**: Only trades when price breaks H1 range with bias alignment
2. **Fast exits**: H1-only uses ATR trail (exit_mode=4), M5 uses trailing SL + anti-candle
3. **No overnight holds**: Most trades exit within 1-5 candles
4. **Self-correcting**: Tight stops + fast re-entry after cooldown
5. **All-session**: Trades ASIA, LONDON, NEW_YORK (configurable)
6. **Broker-agnostic**: Same strategy via Capital.com REST API or MT5

---

## 9. Capital.com API Notes

- Auth: `POST /api/v1/session` → CST + X-SECURITY-TOKEN headers
- Session expires after ~10min inactivity (auto-refresh on 401)
- Prices: `GET /api/v1/prices/{epic}` (epic in path, not query)
- Range query broken (400), count-based fallback (max=1000, paginate with `to`)
- XAUUSD epic = "GOLD", decimalPlacesFactor=2
- Demo: `demo-api-capital.backend-capital.com`
- Live: `api-capital.backend-capital.com`
- Toggle with `CAPITAL_DEMO=true/false`

---

## 10. Broker Switching

```env
BROKER=CAPITAL
# vs
BROKER=MT5
```

- CAPITAL → `app/capital_client.py` (REST API, no MT5 needed)
- MT5 → `app/mt5_client.py` (requires MetaTrader5 terminal)
- `trade_executor.py` delegates to whichever client is active

---

## 11. Dependencies

```
requests          # Capital.com REST API
pandas            # OHLC data
numpy             # Numerical ops
python-dotenv     # .env loading
uvicorn           # FastAPI server
fastapi           # Dashboard
yfinance          # Backtest data (optional, Yahoo)
MetaTrader5       # Backtest data (optional, MT5)
```
