"""
Semantic cache threshold sweep analysis.

Methodology:
- Constructs paraphrase pairs (semantically equivalent, different surface form)
  and negative pairs (semantically unrelated queries).
- For each threshold value in THRESHOLD_RANGE, measures:
    hit_rate          = fraction of paraphrase pairs that exceed the threshold
    false_pos_rate    = fraction of negative pairs that exceed the threshold
    precision         = true_hits / (true_hits + false_hits)
- Saves results as CSV and a line chart.

The goal is not to find the "best" threshold but to characterise system
behaviour at each value — exactly what the assignment requires.

Run from project root:
    python -m analysis.threshold_analysis
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.embedder import Embedder
from logger.logging_config import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_RANGE = [0.70, 0.75, 0.78, 0.80, 0.83, 0.85, 0.88, 0.90, 0.93, 0.95, 0.99]

# Paraphrase pairs: semantically equivalent, surface form differs
PARAPHRASE_PAIRS = [
    ("How does gun control affect crime?", "What is the impact of firearm regulations on crime rates?"),
    ("What are the best graphics cards available?", "Which GPU should I buy for 3D rendering?"),
    ("How does the immune system fight infections?", "What is the body's defense mechanism against viruses?"),
    ("What causes a car engine to overheat?", "Why does an automobile engine run too hot?"),
    ("Is capital punishment morally justified?", "Should the death penalty be permitted by law?"),
    ("How do I recover from a hard disk failure?", "What steps should I take when my hard drive crashes?"),
    ("What are the health effects of anabolic steroids?", "How do performance-enhancing drugs affect the human body?"),
    ("Can space travel become a commercial industry?", "Is it viable to build a private space tourism business?"),
    ("What is the recommended treatment for type 2 diabetes?", "How should blood sugar be managed in diabetic patients?"),
    ("How does public-key encryption work?", "What is the mechanism behind asymmetric cryptography?"),
    ("What are the arguments for stricter gun laws?", "Why do some people support increased firearm regulation?"),
    ("How do solar panels generate electricity?", "What is the process by which photovoltaic cells produce power?"),
]

# Negative pairs: topically unrelated — should NOT exceed threshold
NEGATIVE_PAIRS = [
    ("How does gun control affect crime?", "What are the best pasta recipes for dinner?"),
    ("What GPU should I buy for gaming?", "How do I treat a knee injury after exercise?"),
    ("Is capital punishment ethical?", "How do I configure a home network router?"),
    ("What causes engine overheating?", "What are the latest discoveries in deep space astronomy?"),
    ("How does encryption work?", "What are common baseball pitching statistics?"),
    ("How do solar panels work?", "What is the history of the Roman Empire?"),
    ("What is the treatment for diabetes?", "How do I write a Python web scraper?"),
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.flatten(), b.flatten()))


def run_threshold_sweep() -> pd.DataFrame:
    logger.info("Threshold sweep started")

    embedder = Embedder()

    logger.info("Encoding %d paraphrase pairs", len(PARAPHRASE_PAIRS))
    para_scores = [
        _cosine_similarity(embedder.encode_query(q1), embedder.encode_query(q2))
        for q1, q2 in PARAPHRASE_PAIRS
    ]

    logger.info("Encoding %d negative pairs", len(NEGATIVE_PAIRS))
    neg_scores = [
        _cosine_similarity(embedder.encode_query(q1), embedder.encode_query(q2))
        for q1, q2 in NEGATIVE_PAIRS
    ]

    logger.info(
        "Paraphrase scores | min=%.3f | max=%.3f | mean=%.3f",
        min(para_scores), max(para_scores),
        sum(para_scores) / len(para_scores),
    )
    logger.info(
        "Negative pair scores | min=%.3f | max=%.3f | mean=%.3f",
        min(neg_scores), max(neg_scores),
        sum(neg_scores) / len(neg_scores),
    )

    records = []
    for theta in THRESHOLD_RANGE:
        true_hits = sum(1 for s in para_scores if s >= theta)
        false_hits = sum(1 for s in neg_scores if s >= theta)

        hit_rate = true_hits / len(para_scores)
        false_pos_rate = false_hits / len(neg_scores)
        precision = (
            true_hits / (true_hits + false_hits)
            if (true_hits + false_hits) > 0
            else 1.0
        )

        records.append(
            {
                "threshold": theta,
                "hit_rate": round(hit_rate, 4),
                "precision": round(precision, 4),
                "false_positive_rate": round(false_pos_rate, 4),
                "true_hits": true_hits,
                "false_hits": false_hits,
            }
        )

        logger.info(
            "theta=%.2f | hit_rate=%.3f | precision=%.3f | false_pos=%.3f",
            theta, hit_rate, precision, false_pos_rate,
        )

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "threshold_analysis.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Threshold analysis CSV saved | path=%s", csv_path)

    _plot_results(df)
    return df


def _plot_results(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(df["threshold"], df["hit_rate"], marker="o", label="Hit Rate (paraphrase pairs)", color="steelblue")
    ax.plot(df["threshold"], df["precision"], marker="s", label="Precision", color="seagreen")
    ax.plot(df["threshold"], df["false_positive_rate"], marker="^", label="False Positive Rate", color="crimson")
    ax.axvline(x=0.85, color="orange", linestyle="--", alpha=0.7, label="Default threshold (0.85)")

    ax.set_xlabel("Similarity Threshold (theta)")
    ax.set_ylabel("Rate")
    ax.set_title("Semantic Cache: System Behaviour Across Threshold Values")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.68, 1.01)
    ax.set_ylim(-0.05, 1.10)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "threshold_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Threshold analysis chart saved | path=%s", plot_path)


if __name__ == "__main__":
    run_threshold_sweep()
