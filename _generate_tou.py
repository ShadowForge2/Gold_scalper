from fpdf import FPDF
import os

class TermsPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(180, 140, 30)
        self.cell(0, 10, 'QuantoraFX - Terms of Use', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Last Updated: July 2026', align='C', new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def stitle(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(160, 120, 20)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.cell(5, 5, '-')
        self.multi_cell(0, 5, text)
        self.ln(1)

    def warning(self, text):
        self.set_fill_color(255, 235, 235)
        self.set_draw_color(200, 50, 50)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(180, 30, 30)
        self.cell(0, 7, '   RISK WARNING', new_x="LMARGIN", new_y="NEXT")
        self.set_x(12)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(120, 30, 30)
        self.set_fill_color(255, 245, 245)
        self.multi_cell(176, 5, text)
        self.ln(4)

pdf = TermsPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

pdf.body(
    'These Terms of Use ("Terms") govern your access to and use of the QuantoraFX automated trading '
    'bot ("the Bot"), its associated mobile application, and backend infrastructure (collectively, '
    '"the Service"). By installing, accessing, or using the Service, you agree to be bound by these '
    'Terms. If you do not agree, do not use the Service.'
)

pdf.warning(
    'Trading in foreign exchange (Forex) and Contracts for Difference (CFDs) involves substantial '
    'risk of loss and is not suitable for all investors. You could lose some or all of your invested '
    'capital. Past performance is not indicative of future results. Backtested results are hypothetical '
    'and do not represent actual trading. Never trade with money you cannot afford to lose.'
)

pdf.stitle('1. Service Description')
pdf.body(
    'QuantoraFX provides an automated algorithmic trading bot that connects to your Capital.com '
    'trading account via API. The Bot analyzes market data using proprietary signal engines, machine '
    'learning models (XGBoost), and adaptive confirmation logic to generate buy/sell signals for '
    'XAUUSD (Gold) on M1/M5 timeframes. The Bot executes trades, manages stop-losses and take-profits, '
    'and enforces risk limits on your behalf.'
)

pdf.stitle('2. Risk Disclosure')
pdf.warning(
    '1. CAPITAL AT RISK: You may lose all capital allocated to this Bot. No strategy guarantees '
    'profit or prevents loss. Historical backtest win rates of 55-70% do not guarantee future '
    'performance.\n\n'
    '2. BACKTEST LIMITATIONS: Backtests use historical data and cannot account for slippage, '
    'liquidity gaps, spread widening during news events, or broker latency. Real-world results '
    'will differ.\n\n'
    '3. DRAWDOWN: Maximum drawdown in backtests ranged from 5-20% depending on configuration. '
    'Live drawdown may exceed these figures.\n\n'
    '4. LEVERAGE: Trading on margin amplifies both gains and losses.\n\n'
    '5. TECHNOLOGY RISK: The Bot depends on internet connectivity, server uptime, and broker API '
    'availability. Downtime, API changes, or network failures may prevent trade execution or '
    'risk management.'
)

pdf.stitle('3. Backtest Performance Summary')

pdf.set_font('Helvetica', 'B', 9)
pdf.set_text_color(160, 120, 20)
col_w = [90, 28, 28, 28]
headers = ['Scenario', 'Win Rate', 'Profit Factor', 'Max DD']
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 6, h, align='C' if i > 0 else 'L')
pdf.ln()

data = [
    ('ML Model 2022 (LOTx2)', '87-90%', '9-13', '~15%'),
    ('ML Model 2023 (LOTx2)', '87-90%', '9-13', '~12%'),
    ('ML Model 2024 (LOTx2)', '87-90%', '9-13', '~10%'),
    ('Direction Model 2022', '54.6%', '1.20', '~18%'),
    ('Direction Model 2023', '57.4%', '1.35', '~15%'),
    ('SL/TP Model 2022', '59.5%', '1.47', '~14%'),
    ('SL/TP Model 2023', '62.4%', '1.66', '~12%'),
]
pdf.set_font('Helvetica', '', 9)
pdf.set_text_color(60, 60, 60)
for i, (label, wr, pf, dd) in enumerate(data):
    if i % 2 == 0:
        pdf.set_fill_color(245, 245, 250)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.cell(col_w[0], 6, label, fill=True)
    pdf.cell(col_w[1], 6, wr, align='C', fill=True)
    pdf.cell(col_w[2], 6, pf, align='C', fill=True)
    pdf.cell(col_w[3], 6, dd, align='C', fill=True)
    pdf.ln()

pdf.ln(3)
pdf.set_font('Helvetica', 'I', 8)
pdf.set_text_color(100, 100, 100)
pdf.multi_cell(0, 4,
    'Note: LOT_MULT=2 uses 0.02 lots per trade (3 concurrent = 0.06 max). '
    'Results with LOT_MULT=5 showed 14-20% event loss rate and blew up in 2022. '
    'Higher lot multipliers significantly increase risk of total account loss.'
)

pdf.stitle('4. User Responsibilities')
pdf.bullet('You must have a funded Capital.com trading account with sufficient margin.')
pdf.bullet('You are responsible for monitoring the Bot and manually intervening if necessary.')
pdf.bullet('You must maintain a stable internet connection for the Bot to function.')
pdf.bullet('You agree not to modify, reverse-engineer, or redistribute the Bot software.')
pdf.bullet('You will not use the Bot in violation of any applicable laws or regulations.')
pdf.bullet('You are solely responsible for any tax liabilities from your trading activity.')

pdf.stitle('5. Fees and Subscription')
pdf.body(
    'The Service operates on a trial-to-subscription model. New users receive a 30-day free trial '
    'with no fees on demo accounts. Live account users are subject to a 15% profit-sharing fee on '
    'monthly profits after the trial period ends. Fees are calculated and collected automatically '
    'via the integrated payment system. Non-payment of due fees may result in Service suspension.'
)

pdf.stitle('6. Limitation of Liability')
pdf.body(
    'To the maximum extent permitted by law, QuantoraFX and its developers shall not be liable for '
    'any direct, indirect, incidental, special, consequential, or punitive damages arising from your '
    'use of the Service, including but not limited to trading losses, lost profits, or data loss. '
    'The Service is provided "as is" without any warranty, express or implied.'
)

pdf.stitle('7. No Financial Advice')
pdf.body(
    'QuantoraFX is an automated trading tool and does not constitute financial advice, investment '
    'recommendation, or solicitation to trade. We are not registered financial advisors, brokers, '
    'or investment professionals. All trading decisions are based on algorithmic models and '
    'historical patterns, not individualized financial analysis.'
)

pdf.stitle('8. Termination')
pdf.body(
    'We reserve the right to suspend or terminate your access to the Service at any time. Upon '
    'termination, you must cease all use of the Service and uninstall the application. Provisions '
    'regarding limitation of liability, risk disclosure, and governing law shall survive termination.'
)

pdf.stitle('9. Governing Law')
pdf.body(
    'These Terms shall be governed by the laws of the Federal Republic of Nigeria. Any disputes '
    'shall be resolved through binding arbitration in accordance with the Arbitration and '
    'Conciliation Act, Cap A18, Laws of the Federation of Nigeria, 2004.'
)

pdf.stitle('10. Contact')
pdf.body(
    'Email: support@quantorafx.com\n'
    'Website: https://gold-scalper-qyhg.onrender.com'
)

out = os.path.join(os.path.dirname(__file__) or '.', 'QuantoraFX_Terms_of_Use.pdf')
pdf.output(out)
print(f"PDF saved to: {out}")
