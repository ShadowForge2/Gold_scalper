# API Contract — Gold Scalper Pro

## Base URL
- Local: `http://localhost:8000`
- Production: `https://gold-scalper.onrender.com`

## Headers
| Header | Required For | Description |
|--------|-------------|-------------|
| `X-Device-Id` | All `/api/device/*` endpoints | Device UUID (generated on first launch) |
| `Content-Type` | All POST/PUT requests | `application/json` |

---

## Backend Endpoints (Backend `app/api.py`)

### Health & Status
| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| `GET` | `/health` | Server health check | — | `{status, state, connected, broker, symbol}` |

### Admin Bot (mono, from env vars)
| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| `GET` | `/api/account` | Account info | — | Account details or 503 |
| `GET` | `/api/state` | Bot state summary | — | Full bot state |
| `GET` | `/api/positions` | Open positions | — | Position list |
| `POST` | `/api/bot/start` | Start admin bot | — | `{message, state}` |
| `POST` | `/api/bot/stop` | Stop admin bot | — | `{message, state}` |
| `POST` | `/api/trades/close_all` | Emergency close all | — | `{message, closed_count}` |
| `POST` | `/api/bot/settings` | Update bot settings | `{max_event_loss, max_daily_loss, ...}` | `{message, settings}` |
| `POST` | `/api/bot/login` | Login to MT5 account | `{server, account, password}` | `{message, account}` or 401 |
| `GET` | `/api/accounts` | List saved MT5 accounts | — | `{accounts: [...]}` |
| `POST` | `/api/accounts` | Save MT5 account | `{label, server, account, password}` | `{message, accounts}` |
| `DELETE` | `/api/accounts/{account_id}` | Remove MT5 account | — | `{message, accounts}` or 404 |

### Device-Based Account Management
| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| `GET` | `/api/device/accounts` | List device's Capital.com accounts | — | `{accounts: [...]}` |
| `POST` | `/api/device/accounts` | Add Capital.com account | `{api_key, identifier, password, demo}` | `{success, accounts}` |
| `DELETE` | `/api/device/accounts/{identifier}` | Remove account by identifier | — | `{success, accounts}` or 404 |

### Device Bot Control
| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| `POST` | `/api/device/bot/start` | Start bot for device's first account | — | `{message}` or 400/402/503 |
| `POST` | `/api/device/bot/stop` | Stop bot for device | — | Result from bot pool |
| `GET` | `/api/device/bot/state` | Get bot state | — | `{running, bot, ...}` or `{running: false}` |
| `GET` | `/api/device/bot/logs` | Get bot logs | — | `{logs: [...]}` |

### Subscription & Payments
| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| `GET` | `/api/device/subscription` | Get subscription status | — | Subscription object |
| `POST` | `/api/device/subscription/check` | Check & log trial status | — | Subscription object |
| `POST` | `/api/payment/initialize` | Initiate Paystack payment | `{email}` | `{authorization_url, reference}` or 500 |
| `POST` | `/api/payment/verify` | Verify Paystack payment | `{reference}` | `{message, data}` or 400 |

---

## Flutter App `bot_provider.dart` — Current API Calls

> **⚠️ CRITICAL MISMATCH**: The paths below don't match the backend. When `_useMockData = false`, calls will fail.

| Method | Path in Flutter | Actual Backend Path | Match? |
|--------|----------------|---------------------|--------|
| `GET` | `/bot/state` | `/api/device/bot/state` | ❌ |
| `GET` | `/bot/logs` | `/api/device/bot/logs` | ❌ |
| `GET` | `/accounts` | `/api/device/accounts` | ❌ |
| `GET` | `/subscription` | `/api/device/subscription` | ❌ |
| `POST` | `/bot/start` | `/api/device/bot/start` | ❌ |
| `POST` | `/bot/stop` | `/api/device/bot/stop` | ❌ |
| `POST` | `/payment/initialize` | `/api/payment/initialize` | ✅ |
| `POST` | `/payment/verify` | `/api/payment/verify` | ✅ |

**`bot_provider.dart:31`** — `_baseUrl = 'http://localhost:8000'`

---

## Flutter App — OLD/Unused Endpoints

These exist in `api_service.dart` and `auth_provider.dart` but are **no longer used** (replaced by `bot_provider.dart` + device-based flow):

| File | Method | Path | Status |
|------|--------|------|--------|
| `services/api_service.dart` | `GET` | `/api/user/bot/state` | 🗑️ Dead code |
| `services/api_service.dart` | `GET` | `/api/user/bot/logs` | 🗑️ Dead code |
| `services/api_service.dart` | `GET` | `/api/user/accounts` | 🗑️ Dead code |
| `services/api_service.dart` | `POST` | `/api/user/bot/start` | 🗑️ Dead code |
| `services/api_service.dart` | `POST` | `/api/user/bot/stop` | 🗑️ Dead code |
| `services/api_service.dart` | `POST` | `/api/user/accounts` | 🗑️ Dead code |
| `services/api_service.dart` | `DELETE` | `/api/user/accounts/{identifier}` | 🗑️ Dead code |
| `providers/auth_provider.dart` | `GET` | `/auth/me` | 🗑️ Dead code |
| `providers/auth_provider.dart` | `POST` | `/auth/register` | 🗑️ Dead code |
| `providers/auth_provider.dart` | `POST` | `/auth/login` | 🗑️ Dead code |

---

## Required Fixes

1. **Fix 6 paths in `bot_provider.dart`** to match backend:
   - `/bot/state` → `/api/device/bot/state`
   - `/bot/logs` → `/api/device/bot/logs`
   - `/accounts` → `/api/device/accounts`
   - `/subscription` → `/api/device/subscription`
   - `/bot/start` → `/api/device/bot/start`
   - `/bot/stop` → `/api/device/bot/stop`

2. **Remove dead code**: `services/api_service.dart` and `providers/auth_provider.dart` (or archive them)
