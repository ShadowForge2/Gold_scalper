# Gold Scalper — Anchored Summary

## Project
Gold scalping bot trading XAUUSD on Capital.com (MT5 removed). Built with Python, deployed on Render.

## Bugs Found & Fixed

### Initial Deep Audit (12 bugs + MT5 purge)
All documented in prior conversation. Key fixes:
- `capital_client.py:177`: `reconnect()` now passes new credentials instead of cached
- `bot.py:419,582`: momentum `hasattr` fixed, Gemini exit advice wired
- `signal_engine.py:410-419`: recovered positions get 2× ATR trailing SL fallback
- `risk_manager.py:128-130`: spread filter normalized using `point`
- `capital_client.py:465-483`: daily PnL uses balance-based calc
- `bot.py:397`: entry threshold override was never read — added `getattr`
- `bot.py:162-166`: `shutdown()` guarded against missing `trade_executor`/`client`
- `bot.py:325`: bid/ask price selection fixed when no bid/ask columns
- `bot_pool.py:222`: subscription check fails closed (returns `False`)
- `trade_executor.py:43`: `close_all_bot_positions()` handles `None`
- `signal_engine.py:559`: NaN guard in ATR computation
- MT5 purge: deleted `mt5_client.py`, cleaned all references

### Final Audit (5 more bugs, commit `99e4870`)
1. **CRITICAL** — `capital_client.py:399`: position filter used `and` instead of `or` (matched wrong magic numbers)
2. **MEDIUM** — `bot.py:326`: BULLISH entry must always buy at ask regardless of M1 column availability
3. **MEDIUM** — `backtest.py:523`: event-loss forced close used bar `open` instead of `exit_px`
4. **MEDIUM** — `eurusd_strategy.py:159`: tp2 copy-paste bug (REVERTED — out of scope)
5. **LOW** — `bot.py:937`: `allowed_sessions` parsing failed on list input

### Session 2026-06-30: Entry Threshold Bug
**Discovered**: Live log analysis of bot on Render (2hr observation).

**Symptoms**:
- Bias engine works: correctly detects BEARISH (strength=1.00)
- Signal engine works: correctly detects H1 low breakouts
- **No trades ever enter** — every entry blocked by `score_below_entry_threshold`
- Max observed score: 0.135 (price at $4009.08, $1.07 below H1 low of $4010.12)
- Threshold: 0.85 required 85% of H1 range ($6.72) past boundary

**Root cause**: `SIGNAL_ENTRY_THRESHOLD = 0.85` in `config.py:91`. Score formula = `breakout_dist / range_size`. With H1 range ~$7.90, a 0.85 threshold requires $6.72 breakout — impossible for normal scalping. Industry research confirms: gold scalping entries trigger on $0.50-1.50 breakouts (scores 0.06-0.19).

**Fix** (3 files):
- `config.py:91`: `SIGNAL_ENTRY_THRESHOLD = 0.10` (was 0.85) — 10% of H1 range past boundary
- `config.py:137-138`: `META_THRESHOLD_MIN = 0.03`, `MAX = 0.30` (were 0.65/0.95)
- `bot.py:397-401`: Added `atr_entry_threshold` floor from signal (matching backtest behavior at `backtest.py:572-574`)

**Evidence**: Web research on gold scalping breakouts:
- Pro-Scalper: "H1 ATR below 15 pips → breakout entries fail" — minimum meaningful breakout ~$1.50
- Goldbtc.ai: "M5 ATR usually 0.8-2.5 USD" for XAUUSD
- Industry standard: entry on candle close past level, not deep penetration

## Live Bot Status
- Deployed at `https://gold-scalper-qyhg.onrender.com`
- Connects to Capital.com successfully, API 200 OK
- Bias engine correctly detects BEARISH/NEUTRAL
- **Prior to fix**: zero trades due to impossible entry threshold
- **After fix**: should enter on breakouts of ~$0.80+ past H1 boundary
- Balance: $22.72 (GoldScalper) / $19.40 (Bot-aliyuzub)
- FutureWarning: `google.generativeai` deprecated — should migrate to `google.genai`

### Session 2026-07-01: Adaptive Confirmation (vreg_vtight)
**Backtest sweep** confirmed that volatility-regime adaptive gating improves PF vs baseline:

| Mode | Trades | PF | WR | Description |
|---|---|---|---|---|
| baseline | 1391 | 2.48 | 32.1% | No confirmation |
| vreg_tight | 1384 | 2.70 | 33.9% | Low vol: BR >= 50th pct |
| **vreg_vtight** | **1382** | **2.83** | **35.0%** | **Low: >=60th, Normal: >=40th** |

**Key bug found**: Backtest's `pre_compute` broke because M5 is resampled from same M1 as H1, so M5 high can never exceed the *current* H1 high. Fixed by shifting H1 index by -2 (use *previous completed* H1 bar) — matching live bot behavior at `signal_engine.py`.

**Implementation** (3 files):
- `app/adaptive_confirmation.py` — new `AdaptiveConfirmation` class with rolling ATR/BR windows
- `app/bot.py:292` — `update(m1_data)` called every tick after M1 load
- `app/bot.py:453` — `should_enter()` gates entry before `_execute_entry`
- `config.py` — 5 new env vars (`ADAPTIVE_CONFIRMATION_ENABLED`, `ADAPTIVE_CONF_WINDOW`, `ADAPTIVE_CONF_P_LOW/NORM/HIGH`)

## Key Config Values
- `SIGNAL_ENTRY_THRESHOLD = 0.10` (env override: `SIGNAL_ENTRY_THRESHOLD`)
- `MIN_BREAKOUT_SCORE = 0.02` (minimum breakout as fraction of H1 range)
- `ATR_MULTIPLIER = 1.0`, `ATR_PERIOD = 14`
- `SL_ATR_MULTIPLIER = 1.0`, `TP1/2/3_MULTIPLIER = 2.0/4.0/6.0`
- `META_ENABLED = False` (default), clamps: 0.03-0.30
- `ADAPTIVE_CONFIRMATION_ENABLED = True` — gates entry by body-ratio percentile per vol regime
- `ADAPTIVE_CONF_P_LOW = 60` — low vol: require body ratio >= 60th pct
- `ADAPTIVE_CONF_P_NORM = 40` — normal vol: require body ratio >= 40th pct
- `ADAPTIVE_CONF_P_HIGH = 0` — high vol: no filter (0 = min body ratio)

### Session 2026-07-02: ML Direction Prediction (XGBoost)
**Trained two XGBoost models on 2007-2021 M5 data (1.1M bars, 26 features).**

**Model 1: Direction Predictor** — generic M5 direction 3 bars ahead (~75% accuracy).

**Model 2: SL/TP Predictor** — predicts if trade will hit 2xATR TP before 1xATR SL.
- Unconditional win rate (2xATR TP before 1xATR SL): only ~33% (noisy market)
- Model correct: 73% of bars (79% at high confidence)
- **When model says WIN with high confidence: 64-67% actually win** (2x baseline)

**Backtest comparison (Meta-Aggressive + ML filter, with SL):**
| Year | Trades | WR | PF | Trades | WR | PF |
|---|---|---|---|---|---|---|
| | **Direction Model** | | | **SL/TP Model** | | |
| 2022 | 3,866 | 54.6% | 1.20 | **2,581** | **59.5%** | **1.47** |
| 2023 | 3,631 | 57.4% | 1.35 | **2,436** | **62.4%** | **1.66** |

**Without SL** (TP-only exit, no hard stop):
| Year | Trades | WR | PF |
|---|---|---|---|
| 2010 | 3,525 | 63.4% | 1.73 |
| 2011 | 4,138 | 68.9% | 2.22 |
| 2020 | 3,763 | 68.5% | 2.17 |
| 2022 | 3,874 | 62.1% | 1.64 |
| 2023 | 3,626 | 62.6% | 1.67 |

**Implementation (5 files):**
- `app/direction_predictor.py` — new: feature engineering + XGBoost wrapper class
- `_train_direction_model.py` — standaline training script (2007-2021 train, yearly test)
- `app/signal_engine.py:42-70` — `_get_ml_direction()` resamples M1→M5, predicts via model
- `app/signal_engine.py:144-159` — ML validation in evaluate(): rejects if ML disagrees with bias
- `app/bot.py:45-55` — loads `DirectionPredictor` from `models/direction_xgb_m5.joblib` on init
- `config.py:148-150` — 3 new env vars (`ML_CONFIDENCE_THRESHOLD`, `ML_MODEL_PATH`, `ML_M1_HISTORY_BARS`)

**How it works:**
1. Live bot fetches 500 M1 bars instead of 100 for enough M5 history
2. `_get_ml_direction` resamples M1→M5→features→predict
3. If ML prediction (BUY/SELL) doesn't match bias direction, signal is rejected
4. Only enters when ML agrees with bias at confidence >= threshold (default 0.55)

## Key Config Values
- `ML_CONFIDENCE_THRESHOLD = 0.60` — minimum confidence for ML prediction
- `ML_MODEL_PATH = models/direction_xgb_m5.joblib` — direction model
- `ML_BUY_MODEL_PATH = models/buy_sltp_xgb.joblib` — BUY SL/TP model
- `ML_SELL_MODEL_PATH = models/sell_sltp_xgb.joblib` — SELL SL/TP model
- `ML_M1_HISTORY_BARS = 500` — M1 bars fetched for M5 feature computation

## Relevant Files
- `config.py` — all thresholds and multipliers
- `app/bot.py:397-401` — entry threshold logic (uses signal.score + atr_entry_threshold floor)
- `app/bot.py:452-458` — adaptive confirmation gate before entry
- `app/signal_engine.py:60,92-93,121` — score = min(breakout_dist / range_size, 1.0)
- `app/adaptive_confirmation.py` — volatility-regime adaptive confirmation class
- `app/backtest.py:572-574` — backtest threshold logic (uses max(atr_thresh, effective_et))
- `sweep_adaptive.py` — confirmation-mode sweep script (backtest comparison tool)
- `app/direction_predictor.py` — ML feature engineering + DirectionPredictor + SLTPredictor classes
- `_train_direction_model.py` — direction model training (2007-2021, OOS test 2022-2025)
- `_train_sltp_model.py` — SL/TP model training (2xATR TP vs 1xATR SL target)
- `_bt_ml.py` — ML-integrated backtest script
- `models/direction_xgb_m5.joblib` — trained direction model
- `models/buy_sltp_xgb.joblib` — trained BUY SL/TP model
- `models/sell_sltp_xgb.joblib` — trained SELL SL/TP model

### Session 2026-07-02b: Codebase Audit Fixes
**Discovered**: Systematic code review during H1 data debugging identified 11 additional bugs across 6 files.

**Fixes applied:**

| # | File | Line | Bug | Fix |
|---|---|---|---|---|
| 1 | `risk_manager.py` | 51 | Tier indexing off-by-one (`t` used as 0-based) | `t - 1` |
| 2 | `risk_manager.py` | 172 | `event_trades` reset to 0 on single exit breaks event limit | `-= 1` |
| 3 | `signal_engine.py` | 58-82 | ML predict exception kills signal evaluation | `try/except` |
| 4 | `meta_strategy.py` | 124-128 | DRAWDOWN regime unreachable (CHOPPY checked first) | Swap order |
| 5 | `capital_client.py` | 107-119 | Zombie session on ping failure (non-200 silently ignored) | `self.connected = False` |
| 6 | `capital_client.py` | 178-181 | `reconnect()` passed stale `api_key` from init | Optional `api_key` param |
| 7 | `capital_client.py` | 142-147 | 429 rate-limit retry exhausts loop, returns `None` | Return response on last attempt |
| 8 | `trade_executor.py` | 34-41 | `close_position()` crashes on exception | `try/except` |
| 9 | `trade_executor.py` | 53-55 | `get_positions()` returns `None` → crash | `or []`, `.get("symbol")` |
| 10 | `trade_executor.py` | 28-30 | `last_order_error()` called as function but might be property | `callable` guard |
| 11 | `bot.py` | 553 | `_event_start_ts` access crashes if `position_manager` replaced | `getattr` guard |
