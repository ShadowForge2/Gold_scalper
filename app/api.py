import asyncio
from typing import List, Dict
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from app.bot import Bot
import config as cfg

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold Scalper - Live Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #060814;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --gold: #d4af37;
            --gold-bright: #fbbf24;
            --gold-glow: rgba(212, 175, 55, 0.15);
            --green: #10b981;
            --green-glow: rgba(16, 185, 129, 0.15);
            --red: #f43f5e;
            --red-glow: rgba(244, 63, 94, 0.15);
            --indigo: #6366f1;
            --blue: #3b82f6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-main);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            background-image:
                radial-gradient(circle at 10% 20%, rgba(99,102,241,0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(212,175,55,0.05) 0%, transparent 40%);
        }
        header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 1.5rem 2rem; border-bottom: 1px solid var(--border-color);
            background: rgba(6,8,20,0.8); backdrop-filter: blur(12px);
            position: sticky; top: 0; z-index: 100;
        }
        .logo-container { display: flex; align-items: center; gap: 0.75rem; }
        .logo-icon {
            width: 2.5rem; height: 2.5rem;
            background: linear-gradient(135deg, var(--gold-bright), var(--gold));
            border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;
            box-shadow: 0 0 15px rgba(212,175,55,0.4); font-weight: 800; color: #060814; font-size: 1.25rem;
        }
        .logo-text h1 {
            font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; letter-spacing: 0.05em;
            background: linear-gradient(to right, #fff, #d4af37); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .logo-text p { font-size: 0.75rem; color: var(--text-secondary); }
        .connection-badge {
            display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem;
            background: rgba(255,255,255,0.03); padding: 0.5rem 1rem; border-radius: 9999px;
            border: 1px solid var(--border-color);
        }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--red); box-shadow: 0 0 10px var(--red); }
        .dot.connected { background: var(--green); box-shadow: 0 0 10px var(--green); }
        .dashboard-container { max-width: 1400px; margin: 2rem auto; padding: 0 1.5rem; display: grid; grid-template-columns: 1fr; gap: 1.5rem; }
        @media (min-width: 1024px) { .dashboard-container { grid-template-columns: 8fr 4fr; } }
        .card {
            background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 1rem;
            padding: 1.5rem; backdrop-filter: blur(20px);
        }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.75rem; }
        .card-title { font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; }
        .grid-2col { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
        .stat-box {
            background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.03);
            border-radius: 0.75rem; padding: 1rem; display: flex; flex-direction: column; gap: 0.25rem;
        }
        .stat-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
        .stat-value { font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; }
        .positive { color: var(--green); }
        .negative { color: var(--red); }
        .bias-display {
            padding: 0.75rem 1.25rem; border-radius: 0.5rem; text-align: center; font-family: 'Outfit', sans-serif;
            font-weight: 700; font-size: 1.1rem; letter-spacing: 0.05em; margin: 0.5rem 0 1rem;
        }
        .bias-BULLISH { background: rgba(16,185,129,0.15); color: var(--green); border: 1px solid rgba(16,185,129,0.2); }
        .bias-BEARISH { background: rgba(244,63,94,0.15); color: var(--red); border: 1px solid rgba(244,63,94,0.2); }
        .bias-NEUTRAL { background: rgba(75,85,99,0.15); color: var(--text-secondary); border: 1px solid rgba(255,255,255,0.05); }
        .bias-CONFLICT { background: rgba(251,191,36,0.15); color: var(--gold-bright); border: 1px solid rgba(251,191,36,0.2); }
        .status-hero {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 1.5rem; border-radius: 0.75rem; background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.04); margin-bottom: 1rem; text-align: center;
        }
        .status-badge {
            font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 800;
            padding: 0.5rem 1.5rem; border-radius: 0.5rem; letter-spacing: 0.05em;
        }
        .status-IDLE { background: #4b5563; color: white; }
        .status-AWAITING_SIGNAL { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; animation: pulse-indigo 2s infinite; }
        .status-IN_TRADE { background: linear-gradient(135deg, var(--green), #059669); color: white; animation: pulse-green 2s infinite; }
        .status-COOLDOWN { background: #92400e; color: #fbbf24; border: 1px solid #d97706; }
        .status-STOPPED { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
        .status-ENTERING { background: linear-gradient(135deg, var(--blue), #2563eb); color: white; animation: pulse-blue 1.5s infinite; }
        .status-EXITING { background: linear-gradient(135deg, var(--red), #dc2626); color: white; animation: pulse-red 1.5s infinite; }
        .status-BIAS_ANALYSIS { background: #1e40af; color: #bfdbfe; }
        @keyframes pulse-indigo { 0% { box-shadow: 0 0 10px var(--indigo-glow, rgba(99,102,241,0.3)); } 50% { box-shadow: 0 0 25px rgba(99,102,241,0.5); } 100% { box-shadow: 0 0 10px var(--indigo-glow, rgba(99,102,241,0.3)); } }
        @keyframes pulse-green { 0% { transform: scale(1); box-shadow: 0 0 20px var(--green-glow); } 50% { transform: scale(1.05); box-shadow: 0 0 35px rgba(16,185,129,0.4); } 100% { transform: scale(1); box-shadow: 0 0 20px var(--green-glow); } }
        @keyframes pulse-red { 0% { transform: scale(1); box-shadow: 0 0 20px var(--red-glow); } 50% { transform: scale(1.05); box-shadow: 0 0 35px rgba(244,63,94,0.4); } 100% { transform: scale(1); box-shadow: 0 0 20px var(--red-glow); } }
        @keyframes pulse-blue { 0% { box-shadow: 0 0 10px rgba(59,130,246,0.3); } 50% { box-shadow: 0 0 25px rgba(59,130,246,0.5); } 100% { box-shadow: 0 0 10px rgba(59,130,246,0.3); } }
        .controls-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.75rem; }
        .btn {
            font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600; font-size: 0.95rem;
            padding: 0.85rem 1.5rem; border-radius: 0.5rem; border: none; cursor: pointer;
            transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 0.5rem;
        }
        .btn-primary { background: linear-gradient(135deg, var(--gold-bright), var(--gold)); color: #060814; box-shadow: 0 4px 15px rgba(212,175,55,0.25); }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(212,175,55,0.4); }
        .btn-danger { background: linear-gradient(135deg, var(--red), #be123c); color: white; box-shadow: 0 4px 15px rgba(244,63,94,0.25); }
        .btn-danger:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(244,63,94,0.4); }
        .btn-secondary { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); color: var(--text-primary); }
        .btn-secondary:hover { background: rgba(255,255,255,0.1); transform: translateY(-1px); }
        .btn-full { grid-column: span 2; }
        .metric-container { display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem; }
        .metric-row { display: flex; flex-direction: column; gap: 0.4rem; }
        .metric-info { display: flex; justify-content: space-between; font-size: 0.875rem; }
        .progress-track { height: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 9999px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 9999px; transition: width 0.3s ease, background 0.3s ease; width: 0%; }
        .bidirectional-track { height: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 9999px; position: relative; }
        .bidirectional-center { position: absolute; left: 50%; top: -2px; width: 2px; height: 0.75rem; background: var(--gray, #6b7280); }
        .bidirectional-fill { position: absolute; height: 100%; transition: left 0.3s ease, width 0.3s ease, background 0.3s ease; }
        .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .input-group { display: flex; flex-direction: column; gap: 0.35rem; }
        .input-group label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; }
        .input-group input { background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.1); color: white; padding: 0.6rem; border-radius: 0.5rem; font-size: 0.9rem; }
        .input-group input:focus { outline: none; border-color: var(--gold); }
        .table-container { width: 100%; overflow-x: auto; margin-top: 0.5rem; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.875rem; }
        th { padding: 0.75rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); color: var(--text-secondary); font-weight: 500; text-transform: uppercase; font-size: 0.75rem; }
        td { padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.03); }
        tr:last-child td { border-bottom: none; }
        .type-BUY { background: rgba(16,185,129,0.15); color: var(--green); padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 700; }
        .type-SELL { background: rgba(244,63,94,0.15); color: var(--red); padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 700; }
        .log-box {
            background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05); border-radius: 0.5rem;
            height: 250px; padding: 0.75rem; font-family: monospace; font-size: 0.75rem; overflow-y: auto;
            display: flex; flex-direction: column; gap: 0.35rem;
        }
        .log-entry { display: flex; gap: 0.5rem; line-height: 1.4; }
        .log-time { color: var(--gold); flex-shrink: 0; }
        .log-INFO .log-message { color: #9ca3af; }
        .log-WARNING .log-message { color: var(--gold-bright); font-weight: 600; }
        .log-ERROR .log-message { color: var(--red); font-weight: 600; }
        .empty-state { padding: 2rem; text-align: center; color: var(--text-secondary); font-style: italic; }
        .chip { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; }
        .chip-bullish { background: rgba(16,185,129,0.15); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }
        .chip-bearish { background: rgba(244,63,94,0.15); color: var(--red); border: 1px solid rgba(244,63,94,0.3); }
        .risk-row { display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.03); font-size: 0.8rem; }
    </style>
</head>
<body>
<header>
    <div class="logo-container">
        <div class="logo-icon">G</div>
        <div class="logo-text">
            <h1>GOLD SCALPER</h1>
            <p>Bias-Driven | 24/7 Automated</p>
        </div>
    </div>
    <div class="connection-badge">
        <div id="connection-dot" class="dot"></div>
        <span id="connection-text">Connecting...</span>
    </div>
</header>
<div class="dashboard-container">
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <div class="card">
            <div class="card-header">
                <div class="card-title">Trading Account</div>
                <span id="acc-num" style="font-family: monospace; color: var(--gold);">-------</span>
            </div>
            <div class="grid-2col">
                <div class="stat-box"><span class="stat-label">Balance</span><span class="stat-value" id="acc-balance">$0.00</span></div>
                <div class="stat-box"><span class="stat-label">Equity</span><span class="stat-value" id="acc-equity">$0.00</span></div>
                <div class="stat-box"><span class="stat-label">Floating P&L</span><span class="stat-value" id="acc-profit">$0.00</span></div>
                <div class="stat-box"><span class="stat-label">Free Margin</span><span class="stat-value" id="acc-free-margin">$0.00</span></div>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Open Positions</div>
                <span id="pos-count" class="connection-badge">0 Active</span>
            </div>
            <div class="table-container">
                <table>
                    <thead><tr><th>Ticket</th><th>Symbol</th><th>Type</th><th>Vol</th><th>Entry</th><th>Current</th><th>Profit</th></tr></thead>
                    <tbody id="trades-tbody"><tr><td colspan="7" class="empty-state">No open positions.</td></tr></tbody>
                </table>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">System Logs</div>
            </div>
            <div class="log-box" id="system-logs"></div>
        </div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <div class="card">
            <div class="card-header">
                <div class="card-title">Bot Status</div>
                <span id="txt-symbol" class="connection-badge">XAUUSD</span>
            </div>
            <div class="status-hero">
                <span id="bot-status-badge" class="status-badge status-IDLE">IDLE</span>
                <span id="bot-status-desc" style="font-size: 0.8rem; color: var(--text-secondary);">Initializing...</span>
            </div>
            <div class="controls-row">
                <button class="btn btn-primary" id="btn-start" onclick="controlBot('start')">Start Bot</button>
                <button class="btn btn-danger" id="btn-stop" onclick="controlBot('stop')">Stop Bot</button>
                <button class="btn btn-secondary btn-full" id="btn-close-all" onclick="controlBot('close_all')">Emergency Close All</button>
            </div>
            <div class="metric-container">
                <div id="bias-section" class="bias-display bias-NEUTRAL">Computing bias...</div>
                <div class="stat-box" style="display: flex; flex-direction: row; justify-content: space-between; align-items: center;">
                    <div><span class="stat-label">Daily P&L</span><div class="stat-value" id="bot-daily-pnl">$0.00</div></div>
                    <div style="text-align: right;"><span class="stat-label">Limit</span><div style="font-family: 'Outfit', sans-serif; font-weight: 600; color: var(--red);" id="val-daily-loss">-$3.00</div></div>
                </div>
                <div class="stat-box" style="display: flex; flex-direction: row; justify-content: space-between; align-items: center;">
                    <div><span class="stat-label">Event P&L</span><div class="stat-value" id="bot-event-pnl">$0.00</div></div>
                    <div style="text-align: right;"><span class="stat-label">Limit</span><div style="font-family: 'Outfit', sans-serif; font-weight: 600; color: var(--red);" id="val-event-loss">-$1.00</div></div>
                </div>
                <div class="metric-row">
                    <div class="metric-info"><span>Signal Momentum</span><span id="txt-momentum" style="font-weight: 600; color: var(--gold);">0.00</span></div>
                    <div class="progress-track"><div id="bar-momentum" class="progress-fill" style="background: var(--gold); width: 0%;"></div></div>
                    <span style="font-size: 0.7rem; color: var(--text-secondary);">Threshold: >0.65 to trigger</span>
                </div>
                <div class="metric-row">
                    <div class="metric-info"><span>Candle Strength</span><span id="txt-candle" style="font-weight: 600;">0.00</span></div>
                    <div class="progress-track"><div id="bar-candle" class="progress-fill" style="background: var(--indigo, #6366f1); width: 0%;"></div></div>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header"><div class="card-title">Risk Status</div></div>
            <div id="risk-panel">
                <div class="risk-row"><span>Consecutive Losses</span><span id="txt-consec-losses" style="font-weight: 600;">0</span></div>
                <div class="risk-row"><span>Session Trades</span><span id="txt-session-trades" style="font-weight: 600;">0</span></div>
                <div class="risk-row"><span>Cooldown</span><span id="txt-cooldown" style="font-weight: 600;">--</span></div>
                <div class="risk-row"><span>State</span><span id="txt-state" style="font-weight: 600;">IDLE</span></div>
            </div>
        </div>
        <div class="card">
            <div class="card-header"><div class="card-title">Bot Parameters</div></div>
            <form id="settings-form" onsubmit="saveSettings(event)" style="display: flex; flex-direction: column; gap: 1rem;">
                <div class="settings-grid">
                    <div class="input-group"><label for="inp-event-loss">Max Event Loss ($)</label><input type="number" id="inp-event-loss" min="0.10" max="1000.0" step="0.05" required></div>
                    <div class="input-group"><label for="inp-daily-loss">Max Daily Loss ($)</label><input type="number" id="inp-daily-loss" min="0.10" max="10000.0" step="0.05" required></div>
                    <div class="input-group"><label for="inp-max-trades">Max Trades / Event</label><input type="number" id="inp-max-trades" min="1" max="50" step="1" required></div>
                    <div class="input-group"><label for="inp-cooldown">Cooldown (sec)</label><input type="number" id="inp-cooldown" min="0" max="600" step="5" required></div>
                </div>
                <button type="submit" class="btn btn-primary" style="width: 100%;">Apply Settings</button>
            </form>
        </div>
        <div class="card">
            <div class="card-header"><div class="card-title">Account Manager</div></div>
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                <div style="display: flex; gap: 0.5rem;">
                    <input type="text" id="inp-new-server" placeholder="Server" style="flex: 3; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.1); color: white; padding: 0.6rem; border-radius: 0.5rem; font-size: 0.85rem;">
                    <input type="text" id="inp-new-account" placeholder="Account #" style="flex: 2; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.1); color: white; padding: 0.6rem; border-radius: 0.5rem; font-size: 0.85rem;">
                    <input type="password" id="inp-new-password" placeholder="Password" style="flex: 2; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.1); color: white; padding: 0.6rem; border-radius: 0.5rem; font-size: 0.85rem;">
                    <button class="btn btn-primary" onclick="addAccount()" style="padding: 0.6rem 1rem; font-size: 0.85rem; white-space: nowrap;">+ Add</button>
                </div>
                <div id="acct-list" style="display: flex; flex-direction: column; gap: 0.4rem; max-height: 200px; overflow-y: auto;"></div>
                <div id="acct-login-status" style="font-size: 0.8rem; text-align: center; color: var(--text-secondary); display: none;"></div>
            </div>
        </div>
    </div>
</div>
<script>
    const logBox = document.getElementById('system-logs');
    function addLog(message, level = 'INFO') {
        const entry = document.createElement('div');
        entry.className = `log-entry log-${level}`;
        entry.innerHTML = '<span class="log-time">' + new Date().toLocaleTimeString() + '</span><span class="log-message">' + message + '</span>';
        logBox.appendChild(entry);
        while (logBox.childNodes.length > 150) logBox.removeChild(logBox.firstChild);
        logBox.scrollTop = logBox.scrollHeight;
    }
    const connectionDot = document.getElementById('connection-dot');
    const connectionText = document.getElementById('connection-text');
    let ws;
    function connectWS() {
        const p = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(p + '//' + window.location.host + '/ws');
        ws.onopen = () => { connectionDot.className = 'dot connected'; connectionText.textContent = 'Live'; addLog('WebSocket connected', 'INFO'); };
        ws.onclose = () => { connectionDot.className = 'dot'; connectionText.textContent = 'Reconnecting...'; addLog('WebSocket disconnected', 'WARNING'); setTimeout(connectWS, 2000); };
        ws.onmessage = (e) => updateUI(JSON.parse(e.data));
    }
    let isSettingsPopulated = false;
    function updateUI(data) {
        const { account, bot, logs } = data;
        if (account && !account.error) {
            document.getElementById('acc-num').textContent = account.account_number || '---';
            document.getElementById('acc-balance').textContent = '$' + (account.balance || 0).toFixed(2);
            document.getElementById('acc-equity').textContent = '$' + (account.equity || 0).toFixed(2);
            const pe = document.getElementById('acc-profit');
            const pv = account.profit || 0;
            pe.textContent = (pv >= 0 ? '+' : '') + '$' + pv.toFixed(2);
            pe.className = 'stat-value ' + (pv >= 0 ? 'positive' : 'negative');
            document.getElementById('acc-free-margin').textContent = '$' + (account.free_margin || 0).toFixed(2);
        }
        if (bot) {
            const sb = document.getElementById('bot-status-badge');
            const st = bot.state || 'IDLE';
            sb.textContent = st;
            sb.className = 'status-badge status-' + st;
            const desc = document.getElementById('bot-status-desc');
            const descMap = { IDLE: 'Analyzing market...', AWAITING_SIGNAL: 'Watching for entry signal', IN_TRADE: 'Positions active - monitoring', COOLDOWN: 'Cooldown active - waiting', STOPPED: 'Bot halted', ENTERING: 'Placing trades...', EXITING: 'Closing trades...', BIAS_ANALYSIS: 'Computing bias...' };
            desc.textContent = descMap[st] || st;

            const bias = bot.bias || {};
            const biasEl = document.getElementById('bias-section');
            const biasDir = bias.bias || 'NEUTRAL';
            const strength = bias.strength || 0;
            biasEl.textContent = (biasDir === 'BULLISH' ? 'BULLISH BIAS' : biasDir === 'BEARISH' ? 'BEARISH BIAS' : biasDir === 'CONFLICT' ? 'CONFLICTING TIMEFRAMES' : 'NEUTRAL / WAITING') + '  |  Strength: ' + strength.toFixed(2);
            biasEl.className = 'bias-display bias-' + biasDir;

            const pos = bot.positions || {};
            document.getElementById('bot-daily-pnl').textContent = ((pos.daily_pnl || 0) >= 0 ? '+' : '') + '$' + (pos.daily_pnl || 0).toFixed(2);
            document.getElementById('bot-daily-pnl').className = 'stat-value ' + ((pos.daily_pnl || 0) >= 0 ? 'positive' : 'negative');
            document.getElementById('bot-event-pnl').textContent = ((pos.event_pnl || 0) >= 0 ? '+' : '') + '$' + (pos.event_pnl || 0).toFixed(2);
            document.getElementById('bot-event-pnl').className = 'stat-value ' + ((pos.event_pnl || 0) >= 0 ? 'positive' : 'negative');
            document.getElementById('val-daily-loss').textContent = '-$' + (bot.risk?.max_daily_loss || 3).toFixed(2);
            document.getElementById('val-event-loss').textContent = '-$' + (bot.risk?.max_event_loss || 1).toFixed(2);

            const sig = bot.signal || {};
            const mom = sig.momentum || 0;
            document.getElementById('txt-momentum').textContent = mom.toFixed(3);
            document.getElementById('bar-momentum').style.width = Math.min(mom * 100, 100) + '%';
            const cs = sig.candle_strength || 0;
            document.getElementById('txt-candle').textContent = cs.toFixed(3);
            document.getElementById('bar-candle').style.width = Math.min(cs * 100, 100) + '%';

            const risk = bot.risk || {};
            document.getElementById('txt-consec-losses').textContent = risk.consecutive_losses || 0;
            document.getElementById('txt-session-trades').textContent = risk.session_trades || 0;
            document.getElementById('txt-cooldown').textContent = risk.cooldown_active ? 'Active' : '--';
            document.getElementById('txt-state').textContent = st;

            document.getElementById('txt-symbol').textContent = bot.symbol || 'XAUUSD';

            if (!isSettingsPopulated && pos) {
                const r = bot.risk || {};
                document.getElementById('inp-event-loss').value = r.max_event_loss || 1;
                document.getElementById('inp-daily-loss').value = r.max_daily_loss || 3;
                document.getElementById('inp-max-trades').value = r.max_trades_per_event || 5;
                document.getElementById('inp-cooldown').value = r.cooldown_seconds || 60;
                isSettingsPopulated = true;
            }

            const posList = pos.positions || [];
            const tbody = document.getElementById('trades-tbody');
            document.getElementById('pos-count').textContent = (pos.open_count || 0) + ' Active';
            if (posList.length > 0) {
                tbody.innerHTML = '';
                posList.forEach(t => {
                    const pnlColor = t.profit >= 0 ? 'var(--green)' : 'var(--red)';
                    const row = document.createElement('tr');
                    row.innerHTML = '<td style="font-family: monospace;">' + t.ticket + '</td><td style="font-weight: 600;">' + t.symbol + '</td><td><span class="type-' + t.type + '">' + t.type + '</span></td><td>' + t.volume.toFixed(2) + '</td><td>$' + t.price_open.toFixed(2) + '</td><td>$' + t.price_current.toFixed(2) + '</td><td style="font-weight: 700; color: ' + pnlColor + ';">' + (t.profit >= 0 ? '+' : '') + '$' + t.profit.toFixed(2) + '</td>';
                    tbody.appendChild(row);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No active positions.</td></tr>';
            }
        }
        if (logs && logs.length > 0) {
            logs.forEach(log => {
                const id = log.time + '-' + log.message;
                if (!window.renderedLogs) window.renderedLogs = new Set();
                if (!window.renderedLogs.has(id)) {
                    window.renderedLogs.add(id);
                    const entry = document.createElement('div');
                    entry.className = 'log-entry log-' + log.level;
                    entry.innerHTML = '<span class="log-time">' + log.time + '</span><span class="log-message">' + log.message + '</span>';
                    logBox.appendChild(entry);
                    logBox.scrollTop = logBox.scrollHeight;
                }
            });
        }
    }
    async function controlBot(action) {
        const e = { start: '/api/bot/start', stop: '/api/bot/stop', close_all: '/api/trades/close_all' }[action];
        if (!e) return;
        try {
            addLog('Request: ' + action.toUpperCase(), 'INFO');
            const r = await fetch(e, { method: 'POST' });
            const j = await r.json();
            addLog(j.message || 'Done', 'INFO');
        } catch (err) {
            addLog('Error: ' + err.message, 'ERROR');
        }
    }
    async function saveSettings(ev) {
        ev.preventDefault();
        try {
            await fetch('/api/bot/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    max_event_loss: parseFloat(document.getElementById('inp-event-loss').value),
                    max_daily_loss: parseFloat(document.getElementById('inp-daily-loss').value),
                    max_trades_per_event: parseInt(document.getElementById('inp-max-trades').value),
                    cooldown_seconds: parseInt(document.getElementById('inp-cooldown').value),
                })
            });
            addLog('Settings applied', 'INFO');
        } catch (err) {
            addLog('Settings error: ' + err.message, 'ERROR');
        }
    }
    async function botLogin(server, account, password, btnEl) {
        const statusEl = document.getElementById('acct-login-status');
        statusEl.style.display = 'block';
        statusEl.textContent = 'Connecting...';
        statusEl.style.color = 'var(--gold-bright)';
        if (btnEl) btnEl.disabled = true;
        try {
            const r = await fetch('/api/bot/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ server, account, password })
            });
            const j = await r.json();
            if (r.ok) {
                statusEl.textContent = 'Connected: ' + (j.account?.name || account) + ' | $' + (j.account?.balance || 0).toFixed(2);
                statusEl.style.color = 'var(--green)';
                addLog('Logged into ' + account, 'INFO');
            } else {
                statusEl.textContent = 'Failed: ' + (j.error || 'Unknown');
                statusEl.style.color = 'var(--red)';
                addLog('Login failed: ' + account + ' - ' + (j.error || ''), 'ERROR');
            }
        } catch (err) {
            statusEl.textContent = 'Error: ' + err.message;
            statusEl.style.color = 'var(--red)';
        }
        if (btnEl) btnEl.disabled = false;
    }
    async function addAccount() {
        const server = document.getElementById('inp-new-server').value.trim();
        const account = document.getElementById('inp-new-account').value.trim();
        const password = document.getElementById('inp-new-password').value;
        if (!server || !account || !password) { addLog('Fill all fields', 'WARNING'); return; }
        try {
            const r = await fetch('/api/accounts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: account, server, account, password })
            });
            const j = await r.json();
            if (r.ok) {
                document.getElementById('inp-new-server').value = '';
                document.getElementById('inp-new-account').value = '';
                document.getElementById('inp-new-password').value = '';
                addLog('Account saved: ' + account, 'INFO');
                renderAccounts(j.accounts);
            } else {
                addLog('Failed to save: ' + (j.message || ''), 'ERROR');
            }
        } catch (err) {
            addLog('Error: ' + err.message, 'ERROR');
        }
    }
    async function removeAccount(accountId) {
        if (!confirm('Remove account ' + accountId + '?')) return;
        try {
            const r = await fetch('/api/accounts/' + accountId, { method: 'DELETE' });
            const j = await r.json();
            if (r.ok) {
                addLog('Removed account ' + accountId, 'INFO');
                renderAccounts(j.accounts);
            }
        } catch (err) {
            addLog('Error: ' + err.message, 'ERROR');
        }
    }
    function renderAccounts(accounts) {
        const el = document.getElementById('acct-list');
        if (!accounts || accounts.length === 0) {
            el.innerHTML = '<div style="text-align: center; padding: 1rem; color: var(--text-secondary); font-size: 0.85rem;">No saved accounts. Add one above.</div>';
            return;
        }
        el.innerHTML = accounts.map(a =>
            '<div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 0.5rem; padding: 0.5rem 0.75rem;">' +
            '<div style="flex: 1; min-width: 0;">' +
            '<div style="font-weight: 600; font-size: 0.85rem;">' + (a.label || a.account) + '</div>' +
            '<div style="font-size: 0.7rem; color: var(--text-secondary);">' + a.server + ' · ' + a.account + ' · ' + (a.password || '****') + '</div>' +
            '</div>' +
            '<div style="display: flex; gap: 0.35rem; flex-shrink: 0;">' +
            '<button class="btn btn-primary" style="padding: 0.35rem 0.75rem; font-size: 0.75rem;" onclick=\'botLogin("' + a.server.replace(/'/g, "\\'") + '","' + a.account + '","' + a.password.replace(/'/g, "\\'") + '",this)\'>Connect</button>' +
            '<button class="btn btn-danger" style="padding: 0.35rem 0.65rem; font-size: 0.75rem;" onclick="removeAccount(\'' + a.account + '\')">✕</button>' +
            '</div></div>'
        ).join('');
    }
    async function loadAccounts() {
        try {
            const r = await fetch('/api/accounts');
            const j = await r.json();
            renderAccounts(j.accounts);
        } catch (e) { /* ignore */ }
    }
    window.addEventListener('load', () => { connectWS(); loadAccounts(); addLog('Dashboard loaded', 'INFO'); });
</script>
</body>
</html>"""


def create_app(bot: Bot) -> FastAPI:
    app = FastAPI(title="Gold Scalper", version="2.0.0")

    websockets: List[WebSocket] = []

    @app.get("/")
    async def root():
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/health")
    async def health():
        connected = bot.client is not None and bot.client.is_connected()
        return {
            "status": "healthy" if connected else "degraded",
            "state": bot.state,
            "connected": connected,
            "broker": cfg.BROKER,
            "symbol": bot.symbol,
        }

    @app.get("/api/account")
    async def get_account():
        info = bot.client.get_account_info()
        if info is None:
            return JSONResponse(status_code=503, content={"error": "MT5 not connected"})
        return info

    @app.get("/api/state")
    async def get_state():
        return bot.get_state_summary()

    @app.get("/api/positions")
    async def get_positions():
        bot.position_manager.refresh()
        return bot.position_manager.summary()

    @app.post("/api/bot/start")
    async def start_bot():
        bot.start()
        return {"message": "Bot started", "state": bot.state}

    @app.post("/api/bot/stop")
    async def stop_bot():
        bot.stop()
        return {"message": "Bot stopped", "state": bot.state}

    @app.post("/api/trades/close_all")
    async def close_all():
        count = await bot.emergency_close()
        return {"message": f"Closed {count} position(s)", "closed_count": count}

    @app.post("/api/bot/settings")
    async def update_settings(settings: dict):
        bot.update_settings(settings)
        return {"message": "Settings updated", "settings": settings}

    @app.post("/api/bot/login")
    async def bot_login(data: dict):
        server = data.get("server", "")
        account = data.get("account", "")
        password = data.get("password", "")
        result = bot.login(server, account, password)
        if result["success"]:
            return {"message": "Login successful", "account": result["account"]}
        return JSONResponse(status_code=401, content={"error": result.get("error", "Login failed")})

    @app.get("/api/accounts")
    async def list_accounts():
        return {"accounts": bot.list_accounts()}

    @app.post("/api/accounts")
    async def add_account(data: dict):
        label = data.get("label", "")
        server = data.get("server", "")
        account = data.get("account", "")
        password = data.get("password", "")
        result = bot.add_account(label, server, account, password)
        if result["success"]:
            return {"message": result["message"], "accounts": bot.list_accounts()}
        return JSONResponse(status_code=400, content=result)

    @app.delete("/api/accounts/{account_id}")
    async def remove_account(account_id: str):
        result = bot.remove_account(account_id)
        if result["success"]:
            return {"message": result["message"], "accounts": bot.list_accounts()}
        return JSONResponse(status_code=404, content=result)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        websockets.append(websocket)
        try:
            while True:
                account = bot.client.get_account_info()
                state = bot.get_state_summary()
                data = {
                    "account": account or {"error": "No connection"},
                    "bot": state,
                    "logs": bot.logger.logs[-50:],
                    "timestamp": datetime.now().isoformat(),
                }
                await websocket.send_json(data)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            if websocket in websockets:
                websockets.remove(websocket)

    return app
