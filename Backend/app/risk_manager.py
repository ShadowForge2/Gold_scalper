from typing import Dict, Tuple
from datetime import datetime
import config as cfg


class RiskManager:
    def __init__(self):
        self.max_spread = cfg.MAX_SPREAD_PIPS

    def can_enter_trade(self, symbol_info: Dict,
                        current_time: datetime, symbol: str = "XAUUSD") -> Tuple[bool, str]:
        point = symbol_info.get("point", 0.0001)
        spread_pips = float(symbol_info.get("spread", 0)) / point if point > 0 else 0
        max_spread = getattr(cfg, 'SYMBOL_MAX_SPREAD', {}).get(symbol, self.max_spread)
        if spread_pips > max_spread:
            return False, f"spread_too_high ({spread_pips:.1f} > {max_spread})"

        return True, "ok"
