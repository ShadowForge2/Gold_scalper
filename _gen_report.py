from fpdf import FPDF
import os

data = [
    (2003, 1515, 88.6, 7.81, 341, 3),
    (2004, 2369, 90.1, 9.12, 2378, 16),
    (2005, 2197, 88.9, 8.00, 1179, 17),
    (2006, 5744, 87.4, 6.93, 582491, 3046),
    (2007, 7638, 91.6, 10.92, 801667, 5009),
    (2008, 14532, 92.2, 11.88, 3814786, 10167),
    (2009, 11859, 91.7, 11.08, 2429874, 10737),
    (2010, 11104, 91.0, 10.10, 2269147, 4644),
    (2011, 13480, 91.4, 10.69, 3938605, 16023),
    (2012, 11749, 90.5, 9.57, 2535402, 8972),
    (2013, 9848, 91.0, 10.09, 2804201, 16523),
    (2014, 10679, 89.0, 8.10, 1582476, 4038),
    (2015, 9898, 89.9, 8.87, 1273082, 5105),
    (2016, 11108, 90.5, 9.54, 1972080, 4962),
    (2017, 11395, 90.3, 9.26, 1085196, 4991),
    (2018, 11080, 89.9, 8.92, 1068756, 3336),
    (2019, 11979, 89.2, 8.27, 1628390, 11323),
    (2020, 10262, 89.1, 8.14, 3123133, 18175),
    (2021, 10656, 90.2, 9.25, 2813222, 9554),
    (2022, 11072, 87.6, 7.09, 2864737, 13661),
    (2023, 11663, 87.3, 6.86, 2344704, 30991),
    (2024, 10393, 88.0, 7.33, 3285536, 5700),
    (2025, 10062, 86.4, 6.34, 6067350, 155647),
    (2026, 3985, 90.3, 9.30, 6552404, 54670),
]

total_trades = sum(r[1] for r in data)
avg_wr = sum(r[2] for r in data) / len(data)
avg_pf = sum(r[3] for r in data) / len(data)
total_pnl = sum(r[4] for r in data)
min_wr = min(r[2] for r in data)
max_wr = max(r[2] for r in data)
min_wr_year = [r[0] for r in data if r[2] == min_wr][0]
max_wr_year = [r[0] for r in data if r[2] == max_wr][0]

pdf = FPDF(orientation="L", unit="mm", format="A4")
pdf.add_page()

pdf.set_font("Helvetica", "B", 20)
pdf.cell(0, 12, "XAUUSD Gold Scalper - Full Backtest Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "Capital.com Real Parameters: CS=1, margin_rate=0.05 | Multi-Trade (live bot logic) | ML + MetaStrategy", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 6, "Start: $20/year | Exit Mode: Peak Harvest (no SL) | Max Overrides: 20/day", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)

pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 5, f"Period: 2003-2026 (24 years, each independent from $20)")
pdf.cell(0, 5, "", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Total trades: {total_trades:,}")
pdf.cell(0, 5, "", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Average win rate: {avg_wr:.1f}% (range: {min_wr:.1f}% in {min_wr_year} - {max_wr:.1f}% in {max_wr_year})")
pdf.cell(0, 5, "", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Average profit factor: {avg_pf:.2f}")
pdf.cell(0, 5, "", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Total net PnL across all years: ${total_pnl:,}")
pdf.cell(0, 5, "", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Model: XGBoost Direction Predictor (trained 2007-2021, 26 features)")
pdf.cell(0, 5, "", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"All years OOS except 2007-2021 (in-sample)")
pdf.ln(3)

pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "Year-by-Year Results", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

col_w = [16, 24, 20, 20, 40, 28]
headers = ["Year", "Trades", "WR%", "PF", "Net PnL", "Max DD"]
aligns = ["C", "C", "C", "C", "R", "R"]

pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(30, 60, 120)
pdf.set_text_color(255, 255, 255)
for h, w in zip(headers, col_w):
    pdf.cell(w, 6, h, border=1, align="C", fill=True)
pdf.ln()

pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "", 9)
fill = False
for year, trades, wr, pf, pnl, dd in data:
    if fill:
        pdf.set_fill_color(235, 240, 250)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.cell(col_w[0], 5.5, str(year), border=1, align="C", fill=True)
    pdf.cell(col_w[1], 5.5, f"{trades:,}", border=1, align="C", fill=True)
    pdf.cell(col_w[2], 5.5, f"{wr:.1f}", border=1, align="C", fill=True)
    pdf.cell(col_w[3], 5.5, f"{pf:.2f}", border=1, align="C", fill=True)
    pnl_str = f"+${pnl:,}" if pnl >= 0 else f"-${abs(pnl):,}"
    pdf.cell(col_w[4], 5.5, pnl_str, border=1, align="R", fill=True)
    pdf.cell(col_w[5], 5.5, f"${dd:,}" if dd >= 1000 else f"${dd}", border=1, align="R", fill=True)
    pdf.ln()
    fill = not fill

pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(30, 60, 120)
pdf.set_text_color(255, 255, 255)
pdf.cell(col_w[0], 6, "Total", border=1, align="C", fill=True)
pdf.cell(col_w[1], 6, f"{total_trades:,}", border=1, align="C", fill=True)
pdf.cell(col_w[2], 6, f"{avg_wr:.1f}%", border=1, align="C", fill=True)
pdf.cell(col_w[3], 6, f"{avg_pf:.2f}", border=1, align="C", fill=True)
pdf.cell(col_w[4], 6, f"+${total_pnl:,}", border=1, align="R", fill=True)
pdf.cell(col_w[5], 6, "-", border=1, align="C", fill=True)
pdf.ln()

pdf.set_text_color(0, 0, 0)
pdf.ln(5)

pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "Notes", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
notes = [
    "Each year runs independently starting from $20 balance (no compounding across years).",
    "Multi-trade: up to 5 sub-trades per event (capped by MetaStrategy TRENDING_STRONG regime).",
    "Exit mode: Peak Harvest (no SL) - trail stop, direction loss, momentum decay, ML hold, ML reversal.",
    "ML override limit: 20/day. ML confidence threshold: 0.60. ML bias override threshold: 0.70.",
    "2003-2005 low PnL due to gold price ($350-500/oz) with $20 starting balance.",
    "2026 is partial year (Jan-Jul only).",
    "Lowest WR year: 2025 (86.4%) - furthest OOS from training data (2007-2021).",
    "Highest WR year: 2008 (92.2%) - financial crisis volatility favorable to system.",
]
for n in notes:
    pdf.cell(0, 5, f"  * {n}", new_x="LMARGIN", new_y="NEXT")

fname = "backtest_report_2003_2026.pdf"
pdf.output(fname)
print(f"PDF saved: {fname} ({os.path.getsize(fname):,} bytes)")
