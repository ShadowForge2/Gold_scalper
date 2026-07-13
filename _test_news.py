"""Quick test for news state machine + calendar + hardcoded events."""
import sys; sys.path.insert(0, '.')
from app.economic_calendar import EconomicCalendar, _build_hardcoded_events
from app.news_state_machine import NewsStateMachine
from datetime import datetime, timezone

# Check hardcoded events
ev = _build_hardcoded_events()
by_type = {}
for e in ev:
    by_type.setdefault(e["title"], 0)
    by_type[e["title"]] += 1

print("Hardcoded events (next 30 days):")
for title, count in sorted(by_type.items(), key=lambda x: -x[1]):
    print(f"  {title}: {count}")
print(f"Total: {len(ev)} events")

# Check for same-time duplicates
for i, e in enumerate(ev):
    for j, e2 in enumerate(ev):
        if i < j:
            diff = abs((e["datetime"] - e2["datetime"]).total_seconds())
            if diff < 60:
                print(f"  OVERLAP: {e['title']} & {e2['title']} @ {e['datetime']}")

# Test State Machine
print("\n--- State Machine Test ---")
nsm = NewsStateMachine()
print(f"Default state: {nsm.state}")

for i in range(20):
    nsm.feed_m5_bar(100.0 + i*0.1, 99.8 + i*0.05, 100.0 + i*0.08)
print(f"ATR after 20 bars: {nsm._atr:.2f}")

# Test calendar
print("\n--- Economic Calendar ---")
cal = EconomicCalendar(cache_path="data/test_calendar_cache.pkl")
next_ev = cal.get_next_event()
if next_ev:
    mins = (next_ev["datetime"] - datetime.now(timezone.utc)).total_seconds() / 60.0
    print(f"Next event: {next_ev['title']} @ {next_ev['datetime']} ({mins:.0f} min away)")
    print(f"Source: {next_ev['source']}, Impact: {next_ev['impact']}")
else:
    print("No upcoming events")
print(f"Calendar available: {cal.is_available}")

# Test state machine with calendar
nsm2 = NewsStateMachine()
nsm2.set_calendar(cal)
state = nsm2.update()
print(f"\nState: {state}")
print(f"Block entry: {nsm2.should_block_entry()}")
print(f"Multipliers: {nsm2.get_entry_multipliers()}")

# Test state transitions
print("\n--- State Transition Test ---")
nsm3 = NewsStateMachine()
nsm3.set_calendar(cal)
# Feed bars to establish ATR
for i in range(20):
    nsm3.feed_m5_bar(100.0 + i*0.1, 99.8 + i*0.05, 100.0 + i*0.08)
print(f"ATR: {nsm3._atr:.2f}")
# Feed a 4x ATR spike bar
base = nsm3._m5_closes[-1]
high_vol = base + nsm3._atr * 4
nsm3.feed_m5_bar(high_vol + 1, base - 0.1, high_vol)
# Update should detect SPIKE
state = nsm3.update()
print(f"After vol spike: state={state}")
print(f"Block entry: {nsm3.should_block_entry()}")
print(f"Multipliers: {nsm3.get_entry_multipliers()}")
print(f"Info: {nsm3.get_state_info()['state']}")
