import MetaTrader5 as mt5
import config as cfg
acct = cfg.MT5_ACCOUNT
if not mt5.initialize(login=int(acct), password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER):
    print(f"MT5 init failed: {mt5.last_error()}")
    mt5.shutdown()
    exit(1)
print("MT5 initialized OK")
info = mt5.account_info()
if info:
    print(f"Account: {info.login} Balance: {info.balance}")
else:
    print(f"account_info failed: {mt5.last_error()}")
mt5.shutdown()
