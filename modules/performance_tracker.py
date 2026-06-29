"""
modules/performance_tracker.py
────────────────────────────────
Tracks quiz scores per topic in a lightweight JSON file.
No database needed — perfect for getting started!
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

TRACKER_PATH = "data/performance.json"


def _load() -> dict:
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, "r") as f:
            return json.load(f)
    return {}


def _save(data: dict):
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
    with open(TRACKER_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API ─────────────────────────────────────────────────────────────

def record_quiz_result(topic: str, score: int, total: int):
    """
    Save a quiz attempt for a topic.

    Parameters
    ----------
    topic : str   – e.g. "neural networks"
    score : int   – number of correct answers
    total : int   – total questions in the quiz
    """
    data = _load()
    entry = {
        "score": score,
        "total": total,
        "pct": round(score / total * 100, 1),
        "timestamp": datetime.now().isoformat(),
    }
    data.setdefault(topic, []).append(entry)
    _save(data)


def get_performance(topic: Optional[str] = None) -> dict:
    """
    Return performance history.
    If topic is None, returns all topics.
    """
    data = _load()
    if topic:
        return {topic: data.get(topic, [])}
    return data


def get_weak_topics(threshold: float = 60.0) -> List[str]:
    """
    Return topics where the average score is below `threshold` percent.
    These are used to personalise resource recommendations.
    """
    data  = _load()
    weak  = []
    for topic, attempts in data.items():
        if not attempts:
            continue
        avg = sum(a["pct"] for a in attempts) / len(attempts)
        if avg < threshold:
            weak.append(topic)
    return weak


def get_summary_stats() -> Dict[str, dict]:
    """
    Return a summary dict per topic with: attempts, avg_score, best, latest.
    """
    data    = _load()
    summary = {}
    for topic, attempts in data.items():
        if not attempts:
            continue
        pcts = [a["pct"] for a in attempts]
        summary[topic] = {
            "attempts":  len(attempts),
            "avg_score": round(sum(pcts) / len(pcts), 1),
            "best":      max(pcts),
            "latest":    attempts[-1]["pct"],
        }
    return summary


