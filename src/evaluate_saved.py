from datasets import load_from_disk
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from .euri_llm import ragas_llm
from .ragas_embeddings import ragas_embeddings


dataset = load_from_disk(
    "data/processed/ragas_dataset"
)

print(f"Loaded {len(dataset)} saved evaluation examples.")

run_config = RunConfig(
    timeout=300,
    max_retries=2,
    max_wait=10,
    max_workers=1,
)

print("\nRunning RAGAS evaluation...\n")

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config,
)

print("\n========== RAGAS RAW RESULT ==========")
print(result)
print("======================================")