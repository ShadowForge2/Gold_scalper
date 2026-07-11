"""
Economic calendar — fetches high-impact gold-relevant events.
Sources (priority order):
  1. Forex Factory (scrape) — currently 403 Forbidden
  2. JBlanked API — needs JBLANKED_API_KEY, 1 req/day free
  3. Finnhub API — needs FINNHUB_API_KEY, 60 req/min free (signup: finnhub.io)
  4. Hardcoded repeating events (always works, date approximations)
  5. User JSON override file
"""
import json
import os
import pickle
import re
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional
import logging

import httpx

logger = logging.getLogger(__name__)

GOLD_RELEVANT_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "CNY"}

HIGH_IMPACT_KEYWORDS = [
    "non-farm", "employment change", "unemployment", "nfp",
    "cpi", "consumer price", "inflation",
    "fomc", "fed", "interest rate", "federal funds",
    "gdp", "gross domestic",
    "retail sales",
    "ism", "manufacturing", "services pmi",
    "jobless claims",
    "ppi", "producer price",
    "michigan", "consumer sentiment",
    "industrial production",
    "treasury",
]

def _build_hardcoded_events() -> list[dict]:
    """Generate known repeating gold-impacting event dates for next 30 days."""
    events = []
    today = date.today()
    for delta in range(30):
        d = today + timedelta(days=delta)
        y, m, day = d.year, d.month, d.day
        wd = d.weekday()
        dom = day  # day of month
        last_day = 31  # approximated

        # ── Weekly ─────────────────────────────────
        # Initial Jobless Claims: every Thursday 12:30 UTC
        if wd == 3:
            events.append({
                "datetime": datetime(y, m, day, 12, 30, tzinfo=timezone.utc),
                "title": "Initial Jobless Claims", "currency": "USD", "impact": 2, "source": "hardcoded",
            })

        # ── Monthly (fixed day-of-month estimates) ──
        # CPI: ~13th-15th of month 12:30 UTC
        if dom in (13, 14, 15) and wd <= 4:
            events.append({
                "datetime": datetime(y, m, day, 12, 30, tzinfo=timezone.utc),
                "title": "CPI m/m", "currency": "USD", "impact": 2, "source": "hardcoded",
            })
        # PPI: ~12th-14th 12:30 UTC
        if dom in (12, 13, 14) and wd <= 4:
            events.append({
                "datetime": datetime(y, m, day, 12, 30, tzinfo=timezone.utc),
                "title": "PPI m/m", "currency": "USD", "impact": 2, "source": "hardcoded",
            })
        # Retail Sales: ~14th-16th 12:30 UTC
        if dom in (14, 15, 16) and wd <= 4:
            events.append({
                "datetime": datetime(y, m, day, 12, 30, tzinfo=timezone.utc),
                "title": "Retail Sales m/m", "currency": "USD", "impact": 2, "source": "hardcoded",
            })
        # Michigan Consumer Sentiment (prelim): 2nd Friday 14:00 UTC
        if wd == 4 and 8 <= dom <= 15:
            events.append({
                "datetime": datetime(y, m, day, 14, 0, tzinfo=timezone.utc),
                "title": "Michigan Consumer Sentiment", "currency": "USD", "impact": 1, "source": "hardcoded",
            })

        # ── NFP: first Friday at 12:30 UTC ──
        if wd == 4 and dom <= 7:
            events.append({
                "datetime": datetime(y, m, day, 12, 30, tzinfo=timezone.utc),
                "title": "Non-Farm Employment Change", "currency": "USD", "impact": 2, "source": "hardcoded",
            })
            events.append({
                "datetime": datetime(y, m, day, 12, 30, tzinfo=timezone.utc),
                "title": "Unemployment Rate", "currency": "USD", "impact": 2, "source": "hardcoded",
            })

        # ── ISM: First business day of month ──
        first_biz = _first_business_day(y, m)
        if day == first_biz.day:
            events.append({
                "datetime": datetime(y, m, day, 15, 0, tzinfo=timezone.utc),
                "title": "ISM Manufacturing PMI", "currency": "USD", "impact": 2, "source": "hardcoded",
            })
        # ISM Services: 3rd business day
        third_biz = _nth_business_day(y, m, 3)
        if third_biz and day == third_biz.day:
            events.append({
                "datetime": datetime(y, m, day, 15, 0, tzinfo=timezone.utc),
                "title": "ISM Services PMI", "currency": "USD", "impact": 2, "source": "hardcoded",
            })

        # ── FOMC: ~every 6 weeks. Estimate 3rd Wed of Jan/Mar/May/Jun/Aug/Sep/Nov/Dec ──
        fomc_months = {1, 3, 5, 6, 8, 9, 11, 12}
        if m in fomc_months:
            third_wed = _nth_weekday(y, m, 3, 2)
            if third_wed and day == third_wed.day:
                events.append({
                    "datetime": datetime(y, m, day, 18, 0, tzinfo=timezone.utc),
                    "title": "FOMC Interest Rate Decision", "currency": "USD", "impact": 2, "source": "hardcoded",
                })

        # ── GDP (quarterly advance/revised): last Wednesday of quarter ──
        quarter_map = {1: "Q4", 4: "Q1", 7: "Q2", 10: "Q3"}
        if m in quarter_map:
            last_wed = _last_weekday(y, m, 2)
            if last_wed and day == last_wed.day:
                events.append({
                    "datetime": datetime(y, m, day, 12, 30, tzinfo=timezone.utc),
                    "title": f"GDP {quarter_map[m]} Advance", "currency": "USD", "impact": 2, "source": "hardcoded",
                })

    return events


def _first_business_day(y: int, m: int) -> date:
    """Return the first business day (Mon-Fri) of the month."""
    d = date(y, m, 1)
    while d.weekday() >= 5:  # Sat/Sun
        d += timedelta(days=1)
    return d


def _nth_business_day(y: int, m: int, n: int) -> Optional[date]:
    """Return the nth business day of the month."""
    d = _first_business_day(y, m)
    count = 1
    while count < n:
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        count += 1
    return d


def _nth_weekday(y: int, m: int, n: int, weekday: int) -> Optional[date]:
    """Return the nth occurrence of a weekday (0=Mon) in a month."""
    d = date(y, m, 1)
    count = 0
    while d.month == m:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    return None


def _last_weekday(y: int, m: int, weekday: int) -> Optional[date]:
    """Return the last occurrence of a weekday (0=Mon) in a month."""
    d = date(y, m + 1, 1) - timedelta(days=1)  # last day of month
    while d.month == m:
        if d.weekday() == weekday:
            return d
        d -= timedelta(days=1)
    return None


def _is_relevant(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in HIGH_IMPACT_KEYWORDS)


class EconomicCalendar:
    def __init__(self, cache_path: str = "data/calendar_cache.pkl", cache_ttl_hours: int = 6, user_events_path: str = "", jblanked_api_key: str = "", finnhub_api_key: str = ""):
        self.cache_path = Path(cache_path)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.user_events_path = user_events_path
        self.jblanked_api_key = jblanked_api_key
        self.finnhub_api_key = finnhub_api_key
        self.events: list[dict] = []
        self._last_fetch_ok = False
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                if datetime.now() - data.get("fetched_at", datetime.min) < self.cache_ttl:
                    self.events = [e for e in data.get("events", [])
                                   if e.get("datetime", datetime.min) > datetime.now(timezone.utc) - timedelta(hours=2)]
                    if self.events:
                        logger.info(f"Loaded {len(self.events)} events from calendar cache")
                        self._last_fetch_ok = True
                        return
            except Exception as e:
                logger.warning(f"Calendar cache load failed: {e}")
        self._fetch(api_key=self.jblanked_api_key)

    def _save_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"events": self.events, "fetched_at": datetime.now()}, f)
        except Exception as e:
            logger.debug(f"Calendar cache save failed: {e}")

    def _fetch(self, api_key: str = ""):
        events = []
        # Try Forex Factory scrape
        try:
            fx_events = self._fetch_forexfactory()
            if fx_events:
                logger.info(f"Calendar: fetched {len(fx_events)} events from Forex Factory")
                events = fx_events
        except Exception as e:
            logger.debug(f"Forex Factory fetch failed: {e}")

        # Try JBlanked API if FF failed
        if not events and api_key:
            try:
                jb_events = self._fetch_jblanked(api_key)
                if jb_events:
                    logger.info(f"Calendar: fetched {len(jb_events)} events from JBlanked")
                    events = jb_events
            except Exception as e:
                logger.debug(f"JBlanked fetch failed: {e}")

        # Try Finnhub API (generous free tier, 60 req/min)
        if not events and self.finnhub_api_key:
            try:
                fh_events = self._fetch_finnhub()
                if fh_events:
                    logger.info(f"Calendar: fetched {len(fh_events)} events from Finnhub")
                    events = fh_events
            except Exception as e:
                logger.debug(f"Finnhub fetch failed: {e}")

        # Hardcoded fallback
        if not events:
            events = _build_hardcoded_events()
            if events:
                logger.info(f"Calendar: using {len(events)} hardcoded events")

        user_events = self._load_user_events()
        if user_events:
            events.extend(user_events)
            logger.info(f"Calendar: loaded {len(user_events)} user events")

        events.sort(key=lambda e: e["datetime"])
        now = datetime.now(timezone.utc)
        events = [e for e in events if e["datetime"] > now - timedelta(hours=2)]
        self.events = events
        self._last_fetch_ok = bool(events)
        self._save_cache()

    def _fetch_forexfactory(self) -> list[dict]:
        url = "https://www.forexfactory.com/calendar"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        events: list[dict] = []
        current_date: Optional[date] = None
        rows = soup.select("tr.calendar__row")

        for row in rows:
            date_el = row.select_one("td.calendar__date")
            if date_el:
                parsed = self._parse_ff_date(date_el.get_text(strip=True))
                if parsed:
                    current_date = parsed

            if not current_date:
                continue

            impact_el = row.select_one("td.calendar__impact span")
            if not impact_el:
                continue
            impact = self._parse_ff_impact(" ".join(impact_el.get("class", [])))
            if impact < 2:
                continue

            currency_el = row.select_one("td.calendar__currency")
            if not currency_el:
                continue
            cur = currency_el.get_text(strip=True).upper()
            if cur not in GOLD_RELEVANT_CURRENCIES:
                continue

            event_el = row.select_one("td.calendar__event span.calendar__event-name")
            if not event_el:
                event_el = row.select_one("td.calendar__event a")
            if not event_el:
                continue
            title = event_el.get_text(strip=True)
            if not _is_relevant(title):
                continue

            time_el = row.select_one("td.calendar__time")
            time_str = time_el.get_text(strip=True) if time_el else ""
            event_dt = self._parse_ff_time(current_date, time_str)
            if not event_dt:
                continue

            events.append({
                "datetime": event_dt,
                "title": title,
                "currency": cur,
                "impact": impact,
                "source": "forexfactory",
            })

        return events

    def _fetch_jblanked(self, api_key: str) -> list[dict]:
        """Fetch events from JBlanked News API (free, 1 req/day on free tier)."""
        url = "https://www.jblanked.com/news/api/forex-factory/calendar/week/"
        headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        events = []
        for item in data if isinstance(data, list) else data.get("events", data.get("results", [])):
            title = item.get("Name") or item.get("title") or item.get("event", "")
            if not _is_relevant(title):
                continue
            cur = (item.get("Currency") or "").upper()
            if cur not in GOLD_RELEVANT_CURRENCIES:
                continue
            impact_str = (item.get("Impact") or "").lower()
            impact = 2 if impact_str == "high" else (1 if impact_str == "medium" else 0)
            if impact < 2:
                continue
            date_str = item.get("Date") or item.get("datetime") or item.get("date", "")
            dt = self._parse_jb_date(date_str)
            if not dt:
                continue
            events.append({
                "datetime": dt, "title": title,
                "currency": cur, "impact": impact, "source": "jblanked",
            })
        return events

    def _fetch_finnhub(self) -> list[dict]:
        """Fetch events from Finnhub API (free, 60 req/min)."""
        today = date.today()
        from_str = today.isoformat()
        to_str = (today + timedelta(days=30)).isoformat()
        url = f"https://finnhub.io/api/v1/calendar-economic?token={self.finnhub_api_key}&from={from_str}&to={to_str}"
        resp = httpx.get(url, headers={"Accept": "application/json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        raw_events = data.get("economicCalendar", [])
        events = []
        for item in raw_events:
            title = item.get("event") or ""
            if not _is_relevant(title):
                continue
            country = (item.get("country") or "").upper()
            cur = {"US": "USD", "UK": "GBP", "EU": "EUR", "JP": "JPY", "CH": "CHF", "CN": "CNY", "GB": "GBP"}.get(country, country)
            if cur not in GOLD_RELEVANT_CURRENCIES:
                continue
            impact_str = (item.get("impact") or "").lower()
            impact = 2 if impact_str in ("high", "⭐high") else (1 if impact_str in ("medium", "⭐medium") else 0)
            if impact < 2:
                continue
            time_str = item.get("time") or ""
            dt = self._parse_finnhub_date(time_str)
            if not dt:
                continue
            events.append({
                "datetime": dt, "title": title,
                "currency": cur, "impact": impact, "source": "finnhub",
            })
        return events

    @staticmethod
    def _parse_finnhub_date(date_str: str) -> Optional[datetime]:
        """Parse Finnhub ISO date format."""
        if not date_str:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_jb_date(date_str: str) -> Optional[datetime]:
        """Parse JBlanked date format '2024.02.08 15:30:00'."""
        if not date_str:
            return None
        date_str = date_str.strip().replace(".", "-")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _parse_ff_date(self, text: str) -> Optional[date]:
        text = text.lower().strip()
        if "today" in text:
            return date.today()
        if "tomorrow" in text:
            return date.today() + timedelta(days=1)
        match = re.search(r"([a-z]{3})[a-z]*\.?\s+(\d{1,2})", text)
        if match:
            month_abbr, day = match.groups()
            day = int(day)
            months = "jan feb mar apr may jun jul aug sep oct nov dec".split()
            now = date.today()
            for i, m in enumerate(months, 1):
                if m.startswith(month_abbr[:3]):
                    year = now.year
                    if i < now.month or (i == now.month and day < now.day):
                        year += 1
                    return date(year, i, day)
        return None

    def _parse_ff_time(self, d: date, t: str) -> Optional[datetime]:
        t = t.strip()
        if not t or t.lower() == "allday":
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        match = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", t, re.IGNORECASE)
        if match:
            hour, minute, ampm = int(match.group(1)), int(match.group(2)), match.group(3)
            if ampm:
                ampm = ampm.lower()
                if ampm == "pm" and hour < 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0
            return datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc)
        return None

    def _parse_ff_impact(self, class_str: str) -> int:
        cs = class_str.lower()
        if "high" in cs or "red" in cs:
            return 2
        if "medium" in cs or "orange" in cs:
            return 1
        return 0

    def _load_user_events(self) -> list[dict]:
        if not self.user_events_path or not os.path.exists(self.user_events_path):
            return []
        try:
            with open(self.user_events_path) as f:
                data = json.load(f)
            results = []
            for item in data if isinstance(data, list) else data.get("events", []):
                dt = datetime.fromisoformat(item["datetime"]).replace(tzinfo=timezone.utc)
                results.append({
                    "datetime": dt,
                    "title": item.get("title", "User Event"),
                    "currency": item.get("currency", "USD"),
                    "impact": item.get("impact", 2),
                    "source": "user",
                })
            return results
        except Exception as e:
            logger.warning(f"User events load failed: {e}")
            return []

    def get_next_event(self) -> Optional[dict]:
        now = datetime.now(timezone.utc)
        for e in self.events:
            if e["datetime"] > now:
                return e
        return None

    def get_upcoming(self, n: int = 10) -> list[dict]:
        now = datetime.now(timezone.utc)
        return [e for e in self.events if e["datetime"] > now][:n]

    def time_until_next(self) -> Optional[float]:
        e = self.get_next_event()
        if e is None:
            return None
        return (e["datetime"] - datetime.now(timezone.utc)).total_seconds() / 60.0

    def refresh(self):
        self._fetch(api_key=self.jblanked_api_key)

    @property
    def is_available(self) -> bool:
        return self._last_fetch_ok and bool(self.events)
