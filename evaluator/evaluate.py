from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any
import requests

API_URL = os.getenv("API_URL", "http://api:8000")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/results"))
TIMEOUT = 5

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def wait_for_health(timeout_seconds: int = 120) -> dict:
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            last = response.json()
            if response.status_code == 200 and last.get("model_loaded") is True:
                return last
        except requests.RequestException as exc:
            last = {"error": str(exc)}
        time.sleep(2)
    raise RuntimeError(f"API did not become healthy: {last}")

def test_case(name: str, fn) -> dict:
    started = time.perf_counter()
    try:
        passed = bool(fn())
        return {"name": name, "passed": passed, "latency_ms": round((time.perf_counter()-started)*1000, 3)}
    except Exception as exc:
        return {"name": name, "passed": False, "latency_ms": round((time.perf_counter()-started)*1000, 3), "error": str(exc)}

def post(payload: Any) -> requests.Response:
    return requests.post(f"{API_URL}/recommend", json=payload, timeout=TIMEOUT)

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    health = wait_for_health()
    metadata = load_json(MODEL_DIR / "metadata.json")
    selected = metadata["selected_k"]
    final_eval = next(x for x in metadata["k_selection"] if x["k"] == selected)
    valid = {"user_id": "USR-EVAL-001", "watch_time_hours": 32.5, "top_genres": ["Action", "Thriller"], "avg_session_mins": 85.0}
    results, latencies = [], []

    def valid_request():
        r = post(valid); latencies.append(r.elapsed.total_seconds()*1000)
        return r.status_code == 200 and "segment_id" in r.json()
    def unknown_genre(): return post({**valid, "top_genres": ["Unknown-Genre-XYZ"]}).status_code == 200
    def empty_genres(): return post({**valid, "top_genres": []}).status_code == 200
    def zero_watch(): return post({**valid, "watch_time_hours": 0}).status_code == 200
    def extreme(): return post({**valid, "watch_time_hours": 999999, "avg_session_mins": 999999}).status_code == 200
    def missing_field(): return post({k: v for k, v in valid.items() if k != "watch_time_hours"}).status_code == 422
    def invalid_type(): return post({**valid, "watch_time_hours": "not-a-number"}).status_code == 422
    def negative_watch(): return post({**valid, "watch_time_hours": -1}).status_code == 422
    def negative_session(): return post({**valid, "avg_session_mins": -1}).status_code == 422
    def repeated():
        a, b = post(valid), post(valid)
        return a.status_code == 200 and b.status_code == 200 and a.json() == b.json()

    tests = [
        ("health_check", lambda: health["model_loaded"] is True),
        ("valid_recommendation", valid_request),
        ("unknown_genre", unknown_genre),
        ("empty_genres", empty_genres),
        ("zero_watch_time", zero_watch),
        ("extreme_values", extreme),
        ("missing_required_field", missing_field),
        ("invalid_numeric_type", invalid_type),
        ("negative_watch_time", negative_watch),
        ("negative_session_duration", negative_session),
        ("repeated_request_determinism", repeated),
    ]
    for name, fn in tests:
        results.append(test_case(name, fn))
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    metrics = {
        "model": {"algorithm": metadata["algorithm"], "selected_k": selected, "random_state": metadata["random_state"], "silhouette_score": metadata["final_silhouette_score"], "inertia": metadata["final_inertia"], "selected_k_evidence": final_eval},
        "cluster_balance": {"counts": final_eval["cluster_counts_full_data"], "percentages": final_eval["cluster_percentages_full_data"], "min_cluster_percentage": final_eval["min_cluster_percentage"], "max_cluster_percentage": final_eval["max_cluster_percentage"], "imbalance_ratio": round(final_eval["max_cluster_percentage"] / max(final_eval["min_cluster_percentage"], 1e-9), 4)},
        "api": {"health_check": health, "valid_request_samples": len(latencies), "average_latency_ms": round(sum(latencies)/len(latencies), 3) if latencies else None, "min_latency_ms": round(min(latencies), 3) if latencies else None, "max_latency_ms": round(max(latencies), 3) if latencies else None},
        "robustness": {"total_tests": total, "passed": passed, "failed": total-passed, "pass_rate": round(passed/total, 4), "tests": results},
        "reproducibility": {"random_state": metadata["random_state"], "deterministic_api_repeated_request": next(r["passed"] for r in results if r["name"] == "repeated_request_determinism")},
    }
    out = RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if passed != total:
        raise SystemExit(f"Evaluator failed: {total-passed}/{total} tests failed")
    print(f"Evaluation passed: {passed}/{total}")

if __name__ == "__main__":
    main()
