# Brain Health Check (injected by _render_optimize.py)
# Add this endpoint to app/api.py or import as a router.

import os
import time
import logging
from typing import Dict, Any

_brain_health_logger = logging.getLogger('GoldScalper.Health')


def _rss_mb_fallback() -> float:
    """Best-effort RSS in MB without psutil (Linux /proc, else 0)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def get_brain_health() -> Dict[str, Any]:
    """Comprehensive Brain health status for /health endpoint.

    Reports: memory usage, model load status, feature cache health.
    Safe to call even if Brain is not initialized.
    """
    try:
        import psutil
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        mem_mb = _rss_mb_fallback()

    brain_dir = os.environ.get("BRAIN_MODEL_DIR", "models/brain")
    brain_enabled = os.environ.get("BRAIN_ENABLED", "true").lower() in ("true", "1", "yes")

    models_status = {}
    models_loaded = 0
    models_total = 0
    expected = [
        "thesis_validator", "meta_fusion_xgb", "meta_fusion_lgb",
        "meta_fusion_cat", "meta_fusion_nn", "meta_fusion_rf",
        "twin_calibrator",
    ]

    for name in expected:
        models_total += 1
        onnx_path = os.path.join(brain_dir, f"{name}.onnx")
        jb_path = os.path.join(brain_dir, f"{name}.joblib")

        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path)
                models_status[name] = {"format": "onnx", "size_mb": round(size_mb, 2), "loadable": True}
                models_loaded += 1
            except Exception as e:
                models_status[name] = {"format": "onnx", "size_mb": round(size_mb, 2), "loadable": False, "error": str(e)}
        elif os.path.exists(jb_path):
            size_mb = os.path.getsize(jb_path) / (1024 * 1024)
            models_status[name] = {"format": "joblib", "size_mb": round(size_mb, 2), "loadable": True}
            models_loaded += 1
        else:
            models_status[name] = {"format": "missing", "size_mb": 0, "loadable": False}

    feature_cache_ok = True
    try:
        from app.brain.feature_cache import FeatureCache
        cache = FeatureCache()
        cache.init_symbol("HEALTH_CHECK")
        feature_cache_ok = True
    except Exception:
        feature_cache_ok = False

    reflection_stats = {}
    try:
        from app.brain.reflection import ReflectionEngine
        re = ReflectionEngine()
        reflection_stats = re.get_stats()
    except Exception:
        pass

    status = "healthy" if (models_loaded >= 4 and mem_mb < 450) else "degraded"
    if mem_mb > 480:
        status = "critical"

    return {
        "brain": {
            "enabled": brain_enabled,
            "status": status,
            "models_loaded": models_loaded,
            "models_total": models_total,
            "models": models_status,
            "feature_cache": "ok" if feature_cache_ok else "error",
            "reflection": reflection_stats,
        },
        "memory": {
            "rss_mb": round(mem_mb, 1),
            "limit_mb": 512,
            "usage_pct": round(mem_mb / 512 * 100, 1),
        },
    }
