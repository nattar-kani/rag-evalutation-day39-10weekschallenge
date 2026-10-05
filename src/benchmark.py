import csv
from pathlib import Path

RESULTS_PATH = Path("data/processed/retrieval_results.csv")


def save_result(stage, candidate_k, top_k, recall):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_PATH.exists()

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                ["stage", "candidate_k", "top_k", "recall"]
            )

        writer.writerow(
            [stage, candidate_k, top_k, f"{recall:.3f}"]
        )