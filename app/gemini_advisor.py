import json
import re
from typing import Optional

import config as cfg

try:
    from google import genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False


_ENTRY_SYSTEM_PROMPT = """You are a trading advisor for an automated gold scalping bot.
Your job is to analyze the current market setup and return a structured suggestion.
You can reference external news or macro context if relevant, but base your advice primarily on the numbers provided.

Respond with valid JSON only, no markdown:
{
  "action": "proceed" | "skip" | "caution",
  "tp_modifier": <0.0 to 2.0>,
  "reason": "<1-sentence explanation>"
}

- "proceed": the setup looks good, trade as planned (tp_modifier=1.0)
- "skip": avoid this trade entirely (tp_modifier ignored)
- "caution": take the trade but adjust TP (tp_modifier < 1.0 for smaller TP, > 1.0 for larger)
"""

_EXIT_SYSTEM_PROMPT = """You are a trading advisor for an automated gold scalping bot that has an open position.
Analyze the current market context and advise whether to hold or exit early.

Respond with valid JSON only, no markdown:
{
  "action": "hold" | "exit_early",
  "reason": "<1-sentence explanation>"
}
"""


class GeminiAdvisor:
    def __init__(self):
        self.enabled = (
            _HAS_GEMINI
            and cfg.GEMINI_ENABLED
            and bool(cfg.GEMINI_API_KEY)
        )
        self.client = None
        self.model_name = cfg.GEMINI_MODEL
        if self.enabled:
            self.client = genai.Client(api_key=cfg.GEMINI_API_KEY)

    async def advise_entry(self, setup: dict) -> Optional[dict]:
        if not self.enabled:
            return None
        prompt = self._build_entry_prompt(setup)
        try:
            response = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt)
            return self._parse(response.text)
        except Exception:
            return None

    async def advise_exit(self, context: dict) -> Optional[dict]:
        if not self.enabled:
            return None
        prompt = self._build_exit_prompt(context)
        try:
            response = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt)
            return self._parse(response.text)
        except Exception:
            return None

    def _build_entry_prompt(self, s: dict) -> str:
        return (
            f"{_ENTRY_SYSTEM_PROMPT}\n\n"
            f"Current setup:\n"
            f"- Bias: {s.get('bias', 'N/A')}\n"
            f"- Direction: {s.get('direction', 'N/A')}\n"
            f"- Signal score: {s.get('score', 'N/A')}\n"
            f"- ATR: {s.get('atr', 'N/A')}\n"
            f"- Breakout distance: {s.get('breakout_dist', 'N/A')} pts\n"
            f"- H1 range: {s.get('range_size', 'N/A')} pts\n"
            f"- Spread: {s.get('spread', 'N/A')} pips\n"
            f"- Consecutive losses: {s.get('consecutive_losses', 0)}\n"
            f"- Session: {s.get('session', 'N/A')}\n"
            f"- Momentum: {s.get('momentum', 'N/A')}\n"
            f"- Volatility: {s.get('volatility', 'N/A')} pips\n"
        )

    def _build_exit_prompt(self, c: dict) -> str:
        return (
            f"{_EXIT_SYSTEM_PROMPT}\n\n"
            f"Current context:\n"
            f"- Direction: {c.get('direction', 'N/A')}\n"
            f"- PnL: ${c.get('pnl', 'N/A')}\n"
            f"- Run duration: {c.get('run_duration_min', 'N/A')} min\n"
            f"- Momentum: {c.get('momentum', 'N/A')}\n"
            f"- Distance from entry: {c.get('distance_from_entry', 'N/A')} pts\n"
            f"- TP1 target: {c.get('tp1', 'N/A')}\n"
            f"- TP2 target: {c.get('tp2', 'N/A')}\n"
            f"- Spread: {c.get('spread', 'N/A')} pips\n"
        )

    def _parse(self, text: str) -> Optional[dict]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
