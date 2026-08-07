# Candle Strategy
## Higher-Timeframe Candle-Following Scalper (H1 / 30m)

A strategy that **follows the candle instead of predicting it**. Trade one H1 (or 30m)
candle at a time, enter at the candle's *decision point*, ride the full move, and let a
handful of small losses get covered by the rare full move.

---

## Core Philosophy

> **We do not predict. We do not fight. We follow what the candle is saying.**
> When the candle says up — we are up. When the candle says down — we are down.

The M5/M1 candle engines fail because they **flick constantly** — they flip on every
fakeout, reversal, close, and reopen. On a higher timeframe an H1 flip is a *real* flip.
That is the whole edge: **slow, honest signals instead of fast, fake ones.**

---

## 1. The Timeframe

| Setting | Value | Reason |
|---------|-------|--------|
| Signal timeframe | **H1 (60m)** primary, **30m** fallback | H1 flips are real flips |
| Entry decision | New candle formation | Every candle has a starting point |
| No lower-timeframe churn | M5/M1 entries disabled for this engine | That flickering is the bug we are removing |

The bot must **stare at the candle more than it enters**. Fewer entries, fewer exits,
less bleeding.

---

## 2. The Candle Is a Decision

Every candle, when it starts, asks the same question in both directions: *"Will buyers or
sellers take this candle?"* The candle's early life is the **decision window**:

- It may form a **doji** → undecided → stand aside, wait for the next candle.
- It may form a **hammer / pin** → one side rejected the other → watch which side closes.
- It may form an **evening star / shooting star** → sellers took control → we go down.
- It may form a **morning star / engulfing bull** → buyers took control → we go up.

The **starting point of the candle is the entry reference**. We do not enter mid-candle
blind — we wait until the candle commits.

---

## 3. Order-Flow Load (the "both sides" reading)

The ML watches **both sides of the load** — the pressure building on the buy side vs the
sell side of the candle:

- If the **sell load** is increasing → the candle is being pulled **down**.
- If the **buy load** is increasing → the candle is being pulled **up**.
- If neither side commits → the candle is **testing both sides** → no full move yet →
  do not force a trade.

The new candle tests both sides *or* commits straight into one direction. Only when one
side wins and the candle commits to a **full move** do we enter. We follow the winner.

---

## 4. Entry Rules

1. Wait for a new candle to form and **commit** (not a doji, not still testing).
2. Confirm the **load side**: increasing buy pressure → BUY; increasing sell pressure → SELL.
3. Enter at the commit point — when price moves past the candle's open in the committed
   direction (the "past the open point in a profitable move" condition).
4. No entry if the candle is still undecided (doji / testing both sides / low conviction).

> Rule of thumb: the first entry of the day is the best. Do not force re-entries into a
> candle that already failed once.

---

## 5. Trade Management — Ride the Full Move

| Phase | Behavior |
|-------|----------|
| **Commit** | Enter when the candle commits in one direction |
| **Ride** | While the candle is on a full move, **stay in** — no retracement tolerance |
| **Trail** | Once price has moved past the open point profitably, **trail the profit** |
| **Reversal** | If the candle comes back, we ride it back too — or close and wait for a fresh candle |
| **Uncertain** | If the candle becomes confusing, **close and wait for the next candle** |

**No drawdown tolerance, no retracement tolerance.** We are not holding a position hoping
for a bounce — we are following a committed candle. The moment the candle stops committing
and starts reversing, we are out and waiting.

---

## 6. The Loss Profile (why this is OK)

The expected sequence when the market is not sure which way the candle is going:

```
0.00 $
-0.02 $
-0.04 $
+0.02 $
-0.02 $
+3.00 $   ← the full move — covers all the bleeding
```

- **Many small losses** (tiny, fixed bleeding) while the market tests directions.
- **Tiny profits** on short reversals.
- **One full move** that covers everything — that is the actual profit.

The final profit comes from holding the full move for **hours** if needed. The small losses
are the cost of admission; they must stay **small** (fixed fraction), so they never add up
to more than one full move.

---

## 7. The "Full Move" Decision

1. When price has moved **past the open point** into profit → the bot may **wait and trail**.
2. If price **retraces** → close.
3. Re-enter **only if** the candle would test the area again before a new candle forms.
4. If it is **confusing** → close and wait for the **next candle**. Never chase.

This keeps us out of the account-bleeding loop: we do not enter/exit frequently; we wait
for committed candles and let the winners run.

---

## 7b. The "Jump Candle" Scan (add-on)

Beyond the commit/reversal logic above, there is a **second, higher-conviction signal**
worth scanning for every new candle:

> A candle that **leaves its starting point and jumps in its full direction** — a strong
> body that opens, immediately breaks past the open, and keeps running without pulling
> back to the open.

How to detect it on the forming candle (before it closes, or at commit):
1. **Break-past-open fast**: price crossed the candle's open by ≥ `JUMP_BREAK_R × ATR` and
   never returned to it.
2. **Full-body commitment**: body ratio high (wicks small vs body), and the body covers a
   large share of the H1 range (e.g. ≥ 60–70%).
3. **Follow-through**: once past the open, each new M5 (or tick) pushes the move further in
   the same direction — momentum keeps confirming, no rejection wick.

This is essentially a **marubozu / strong breakout** candle. When it appears, the entry is
higher conviction: enter at the jump (past the open), ride the full move, trail once in
profit (same management as Sections 5 & 7). The jump candle is the *rare full move* that
covers the small bleeds, so the scanner should fire aggressively on it and stand aside on
everything else.

Combined flow per new H1 candle:
- **Doji / testing / low conviction** → STAND-ASIDE.
- **Commit with one-sided load** → enter, ride, trail.
- **Jump candle (break past open + full-body run)** → enter immediately, ride the full move.

---

## 7c. Pair Selection — Only the Best Mover Fires

The bot is **not** forced to trade every enabled pair. At any moment it ranks the enabled
pairs and **only the ones currently moving well are allowed to fire**:

1. **Per-pair live momentum score** each candle: strength of the committed/jump move
   (z-score of the body, efficiency ratio over the window, load-side conviction).
2. **Dynamic per-pair threshold**: each pair only fires when its own recent (e.g. 30–60 day)
   score percentile is high enough — i.e. the pair is in one of its *good* states. Pairs
   stuck in a bad state (choppy, dead, no moves) are automatically muted.
3. **Only the top-K best pairs trade** at any time (e.g. top 2–3 across all enabled pairs),
   so capital always goes to the pair that is paying, not the pair that is idle.
4. **Rotation**: when a pair stops paying (scores collapse), the bot jumps away from it and
   lets the next-ranked pair fire instead. No loyalty to a dead symbol.

This is the "best one moving in a good way at the time should fire, and we jump away from
the symbol not paying up at that time" rule. The ML is per-pair; the selection layer decides
*which* pair's model output actually gets acted on.

---

## 8. Implementation Notes (codebase mapping)

| Concept | Where it lives |
|---------|----------------|
| Candle features (body ratio, momentum z, trend strength) | `app/momentum_engine.py` (`compute_features`) |
| Candle-brain training (the "both sides" model) | `_train_candle_brain.py` |
| Signal evaluation loop | `app/bot.py` (`_search_symbol`, `_momentum_entry_signal`) |
| Entry gates / per-symbol state | `app/bot.py` (`_tick_symbol`, `_execute_entry`) |
| Risk / sizing | `app/risk_manager.py` (fixed small risk per trade) |
| H1 data | resample from M1 via `resample("1h")` (parquet data in `data/dukascopy_*`) |

**New engine requirements (to build):**
- H1-only decision engine that reads the last closed H1 candle (or the live forming one).
- A load/bias measure on both sides: e.g. body vs wick ratio, close position in range,
  and a learned model (candle brain) that outputs BUY / SELL / STAND-ASIDE.
- A trailing-exit controller keyed to the full-move logic (Section 5 & 7).
- A throttle that **reduces the check rate** — evaluate on new-candle boundaries, not on
  every tick.

---

## 9. Acceptance Criteria (what "working" means)

1. **Rare entries** — not hundreds per day; a handful, on committed H1 candles.
2. **No churn** — the bot does not enter/exit repeatedly inside one candle.
3. **Small consistent losses** between winners (fixed bleed, bounded).
4. **Winners cover the bleed** — a small number of +3$ full moves beat the accumulated
   small losses.
5. **Backtest** the H1 rules against 12+ months of H1 data (from M1 parquet) and only
   promote to live when profit factor ≥ 1.5 with a reasonable trade count.

---

*This is a strategy spec. It is not financial advice. Validate on backtest and demo before
risking capital.*
