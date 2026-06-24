import re, os, glob

home = os.path.expanduser("~")
pattern = os.path.join(home, ".local", "share", "opencode", "tool-output", "tool_ef9d46a*")
files = glob.glob(pattern)
if not files:
    print("No output file found")
    exit(1)

text = open(files[0], encoding="utf-8").read()
lines = re.findall(r"^\s{2}(\d{4}-\d{2}-\d{2}):\s+(\d+)\s+tr", text, re.MULTILINE)
total_tr = sum(int(n) for _, n in lines)
days = len(lines)
print(f"Total trades: {total_tr}")
print(f"Trading days: {days}")
print(f"Avg trades/day: {total_tr/days:.2f}")
print(f"Min: {min(int(n) for _,n in lines)} / Max: {max(int(n) for _,n in lines)}")
