"""Quick deploy safety check — run before pushing to Render.

Usage:
    python check_deploy.py              # Check primary backend
    python check_deploy.py --force      # Show status but don't block
    python check_deploy.py --url URL    # Check a specific backend
"""
import sys
import json
import urllib.request
import urllib.error

PRIMARY = "https://gold-scalper-qyhg.onrender.com"
BACKUP = "https://gold-scalper.onrender.com"
ENDPOINT = "/api/deploy/check"
TIMEOUT = 15


def check(url: str) -> dict:
    try:
        req = urllib.request.Request(f"{url}{ENDPOINT}")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def main():
    force = "--force" in sys.argv
    custom_url = None
    for i, a in enumerate(sys.argv):
        if a == "--url" and i + 1 < len(sys.argv):
            custom_url = sys.argv[i + 1]

    urls = [custom_url] if custom_url else [PRIMARY, BACKUP]

    result = None
    used_url = None
    for url in urls:
        result = check(url)
        if result is not None:
            used_url = url
            break

    if result is None:
        print("[WARN] Both backends unreachable — proceeding (server may be down)")
        sys.exit(0)

    print(f"Backend: {used_url}")
    print(json.dumps(result, indent=2))

    if result.get("safe"):
        print("\n[OK] Deploy safe — no open trades")
        sys.exit(0)
    else:
        total = result.get("total_open_positions", 0)
        bots = result.get("bots_with_positions", [])
        print(f"\n[BLOCKED] {total} open position(s) across {len(bots)} bot(s)")
        for b in bots:
            print(f"  - {b['identifier']}: {b['open_positions']} position(s) [{b['state']}] {b['symbol']}")
        if force:
            print("\n[FORCE] --force flag set — overriding block")
            sys.exit(0)
        print("\nWait for trades to close, then retry. Or use --force to override.")
        sys.exit(1)


if __name__ == "__main__":
    main()
