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

## Key Config Values
- `SIGNAL_ENTRY_THRESHOLD = 0.10` (env override: `SIGNAL_ENTRY_THRESHOLD`)
- `MIN_BREAKOUT_SCORE = 0.02` (minimum breakout as fraction of H1 range)
- `ATR_MULTIPLIER = 1.0`, `ATR_PERIOD = 14`
- `SL_ATR_MULTIPLIER = 1.0`, `TP1/2/3_MULTIPLIER = 2.0/4.0/6.0`
- `META_ENABLED = False` (default), clamps: 0.03-0.30

## Relevant Files
- `config.py` — all thresholds and multipliers
- `app/bot.py:397-401` — entry threshold logic (uses signal.score + atr_entry_threshold floor)
- `app/signal_engine.py:60,92-93,121` — score = min(breakout_dist / range_size, 1.0)
- `app/backtest.py:572-574` — backtest threshold logic (uses max(atr_thresh, effective_et))
