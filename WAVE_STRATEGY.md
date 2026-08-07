# Wave Scalper — Intra-Candle M1 Micro-Wave Strategy (LOCKED)

> **Strategy**: Scalp the micro-waves **inside** the forming H1 candle on M1 bars.
> The ML model is **only a chop gate** — it never vetoes direction and has no
> confidence floor. When the H1 candle jumps, switch to a hold-and-trail rider.
> **Status**: Parameters locked 2026-08-07. Lot scaling disabled (lot-exp = 0).

---

## 1. Core Philosophy

> "Doing everything the candle is doing, but in a very minimal and sensible way."

- We do **not** predict the whole H1 candle. We scalp its **micro-waves** as it forms.
- We oscillate with the candle: when it moves against us we stop immediately;
  when it commits we enter; when it goes against us mid-wave we **lock profit at
  the wave's peak**; we wait for it to jump back in the predicted direction to
  re-enter; if we are confused we wait for the next candle.
- The H1 candle flips rarely → the binary model stays consistent.
- The M1 layer acts fast: many tiny waves per candle.

---

## 2. Layered Structure

| Layer | Timeframe | Role |
|-------|-----------|------|
| Decision | **H1 (60m)** | Determines which candle to scalp; flips rarely |
| Chop gate | previous H1 | ML says NONE + not a jump → sit out this candle |
| Scalping | **M1** | Ride the micro-waves of the forming candle |
| Rider | M1 | Jump candle → hold + trail + close on reversal |

---

## 3. LOCKED Configuration (source of truth: `config.py`)

| Key | Value | Meaning |
|-----|-------|---------|
| `CANDLE_ENGINE_WAVE_ENTRY_R` | **0.50** | Enter when a wave travels 0.50×ATR past its base |
| `CANDLE_ENGINE_WAVE_CUT_R` | **0.03** | Stop loss ~zero (0.03×ATR) |
| `CANDLE_ENGINE_WAVE_PROFIT_R` | **0.05** | Lock profit at wave peak − 0.05×ATR pullback |
| `CANDLE_ENGINE_WAVE_TRAIL_R` | **0.5** | Rider trail distance (×ATR) |
| `CANDLE_ENGINE_WAVE_REVERSAL_R` | **0.5** | Rider close-on-reversal distance (×ATR) |
| `CANDLE_ENGINE_COST_R` | **0.05** | Round-trip cost in R |
| `CANDLE_ENGINE_JUMP_BREAK_R` | **1.5** | Jump candle: close ≥1.5×ATR past the open |
| `CANDLE_ENGINE_JUMP_BODY_R` | **0.70** | Jump candle: body ≥70% of the H1 range |
| `CANDLE_ENGINE_WAVE_LOT_EXP` | **0.0** | **LOCKED — cut/profit do NOT scale with lot** |

Best combo everywhere: **entry 0.50 / cut 0.03 / profit 0.05**.

---

## 4. Chop Gate (the model's ONLY job)

The XGBoost model (same pipeline as `_train_candle_h1.py`: labels with
`entry_min_r=0.90`, `edge_margin=1.75`) outputs BUY / SELL / NONE for each
completed H1 candle.

- **Gate rule**: skip the forming candle **only** when the **previous completed**
  H1 candle is NONE-leaning (`P(NONE) > max(P(BUY), P(SELL))`) AND it is not a
  jump candle.
- There is **no confidence floor** and **no direction veto** — the model never
  tells us which way to trade. It only tells us when the market is chop.

---

## 5. Wave Engine — Strict Mechanics

One pass over the forming candle's M1 bars (`[open, high, low, close]` per bar).

### Entry (flat → in)
- **Wave entry**: price touches `base + 0.50×ATR` (long) or `base − 0.50×ATR`
  (short), where `base` = candle open (then previous exit price).
- **Jump-rider trigger** (when enabled): if the forming candle's body reaches
  `1.5×ATR` past the open AND covers ≥70% of its range, enter immediately at the
  candle open in that direction and switch to rider mode.

### Exit — STRICT order (fixes the fill-honesty bug)
For each M1 bar, **exits are checked against the PREVIOUS bar's peak**, then the
peak is updated. A bar can never sell its own high/buy its own low.

1. **Cut**: `low ≤ entry − 0.03×ATR` (long) → close, R = −0.03 − cost.
2. **Lock**: `low ≤ peak − 0.05×ATR` → close, R = (peak − 0.05×ATR − entry) − cost.
3. **Rider trail**: `low ≤ peak − 0.5×ATR` → close, R = (peak − 0.5×ATR − entry) − cost.
4. **Debounce**: the entry bar is skipped for exit checks (`just_entered`), so we
   never buy the same bar's high/low.

(Short side mirrors: `high ≥` thresholds.)

---

## 6. Validation Results (H1 OOS 2023–2025, strict engine)

All **12 pairs positive**; PF ranges **3.16 (XAGUSD) to 5.59 (GAS)**.
XAUUSD reference: **PF 4.17**, +927R, dd 2.8R, 5202 trades, WR 29.6%.

| Pair | PF | Pair | PF |
|------|----|----|----|
| XAUUSD | 4.17 | COPPER | 3.36 |
| XAGUSD | 3.16 | XPT | 3.98 |
| BRENT | 3.76 | US100 | 4.78 |
| WTI | 3.90 | US500 | 4.23 |
| GAS | **5.59** | US30 | 4.93 |
| — | — | DE40 | 4.11 |
| — | — | JP225 | 3.84 |

Drawdowns 2.1–5.1R everywhere. Low win rate (~25–35%) + high PF is **expected**
and desired: the rare big wave covers many small cuts.

> **2026-08-07 correction — cut-R sign bug.** The reference sweep computed long
> cuts as `(entry − stop)/ATR − cost` = `+cut − cost` (−0.02R) instead of the
> true `−cut − cost` (−0.08R). Masked because cost (0.05) > cut (0.03). The
> sign was flipped in `_sweep_candle_wave.py` and `_wave_strict_compare.py`;
> the numbers above are the corrected, honest results (e.g. XAUUSD PF 6.61 →
> **4.17**, net +1035 → **+927R**, dd 1.4 → **2.8R**). All 12 pairs remain
> strongly positive. The live engine (`app/wave_scalper.py`) prices exits from
> actual fills, so its PnL is the honest −0.08R per cut either way.

### Live-equivalence verification
The live `WaveScalper` state machine was verified **trade-for-trade, R-for-R**
against `run_candle_wave`:
- Synthetic M1 (5 seeds × 40 candles, fixed ATR/gate): **ALL MATCH**.
- Real XAUUSD M1, three 3-month OOS windows (2023, 2024, 2025): **ALL MATCH** —
  7316 trades, zero differences (`_wave_engine_test.py`, `_wave_real_compare.py`).

### Honesty check
The strict engine (exits checked against the PREVIOUS bar's peak) was originally
verified against the old same-bar-exit variant. That comparison script
(`_wave_strict_compare.py`) now imports the strict `run_candle_wave`, so both
sides are identical — the old honesty numbers are historical. The strict
mechanics are the locked spec in §5, and the live engine replicates them exactly
(see live-equivalence verification above).

---

## 7. Cost Model — the ONE number to watch live

| Item | R |
|------|---|
| Losing wave (cut) | 0.03 |
| Round-trip cost | 0.05 |
| **Loss budget per losing wave** | **~0.08R** |

Live spread + slippage must stay under the **0.05R cost budget**, or the whole PF
profile degrades. This is the critical live-monitoring number.

---

## 8. Lot Scaling — WHY IT IS OFF (LOT_EXP = 0, LOCKED)

The equity scaler already grows `lot = base × (balance/20) × mult`, so dollar PnL
per wave = `R × ATR × lot` is already a **constant fraction of equity**. There is
nothing to fix.

Tested candidate: **"widen the cut with lot"** (cut/profit distance ∝ lot).

| Setting | Dollar risk growth | XAUUSD result |
|---------|-------------------|----------------|
| `lot-exp 0` (locked) | linear in equity (constant %) | all combos positive, PF 2.9–6.6 |
| `lot-exp 0.5` | equity^1.5 (superlinear) | **every combo ran to negative equity** |

Widening the cut with lot makes dollar risk superlinear → any big losing streak at
high equity wipes the account. **Do not change `CANDLE_ENGINE_WAVE_LOT_EXP`.**

If drawdowns ever need to be tamer at big equity, the correct lever is the **lot
growth exponent** in `EquityScaler.get_lot` (e.g. `lot ∝ equity^0.5`), not the cut.

---

## 9. Files

| File | Role |
|------|------|
| `_sweep_candle_wave.py` | Wave-scalper sweep/validation (`--sweep` restores the grid; default = locked combo; `--compound` = equity-accounted sim) |
| `_wave_strict_compare.py` | Proof of the strict-engine fill-honesty fix |
| `config.py` | `CANDLE_ENGINE_WAVE_*` — locked parameters |
| `app/risk_manager.py` | `EquityScaler` (lot sizing; do not touch `LOT_MULTIPLIER` > 2) |
| `app/candle_engine.py` | Shared features / ATR / labels / jump flags |
| `_train_candle_h1.py` | Model training (labels entry_min_r 0.90, edge_margin 1.75) |

---

*Locked 2026-08-07. Strategy spec — not financial advice. Validate live spread ≤
0.05R budget on demo before risking capital.*
