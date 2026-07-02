import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_score
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import compute_features, FEATURE_COLS
from _train_sltp_model import create_sltp_target, load_year_data

buy_model = joblib.load('models/buy_sltp_xgb.joblib')
sell_model = joblib.load('models/sell_sltp_xgb.joblib')

for ty in [2022, 2023, 2024, 2025]:
    tm5, th1 = load_year_data(ty)
    tft = compute_features(tm5, th1)
    buy_tgt, sell_tgt = create_sltp_target(tm5)

    for name, model, tgt in [('BUY', buy_model, buy_tgt), ('SELL', sell_model, sell_tgt)]:
        data = tft.copy(); data['target'] = tgt
        data = data.dropna(subset=['target']); X = data[FEATURE_COLS]; y = data['target']
        y_pred = model.predict(X.values); y_prob = model.predict_proba(X.values)
        y_conf = np.max(y_prob, axis=1)

        win_idx = y_pred == 1; lose_idx = y_pred == 0
        win_prec = y[win_idx].mean() * 100 if win_idx.sum() > 0 else 0
        lose_prec = (1 - y[lose_idx]).mean() * 100 if lose_idx.sum() > 0 else 0

        hc = y_conf >= 0.60
        hc_win_prec = y[hc & win_idx].mean() * 100 if (hc & win_idx).sum() > 0 else 0
        hc_lose_prec = (1 - y[hc & lose_idx]).mean() * 100 if (hc & lose_idx).sum() > 0 else 0
        hc_acc = ((y_pred[hc] == y[hc]).mean() * 100) if hc.sum() > 0 else 0

        pred_wr = win_idx.sum() / len(y) * 100
        actual_wr = y.mean() * 100

        print(f"{ty} {name}: PredWin={pred_wr:.0f}% TrueWin={actual_wr:.0f}% | "
              f"WinPrec={win_prec:.1f}% LosePrec={lose_prec:.1f}% | "
              f"HC(>=60%): {hc.mean()*100:.0f}% bars, HCWinPrec={hc_win_prec:.1f}% HClosePrec={hc_lose_prec:.1f}% HCAcc={hc_acc:.1f}%")
