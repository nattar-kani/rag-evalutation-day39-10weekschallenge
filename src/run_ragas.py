import csv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from .hybrid_retriever import build_hybrid_retriever, hybrid_search
from .euri_rag import client
from .euri_llm import ragas_llm
from .ragas_embeddings import ragas_embeddings


def load_questions():
    with open(
        "data/evaluation/golden_dataset.csv",
        "r",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def generate_rag_output(
    question,
    vector_store,
    embedding_model,
    bm25,
    chunks,
):
    results = hybrid_search(
        question,
        vector_store,
        embedding_model,
        bm25,
        chunks,
        candidate_k=10,
        top_k=3,
    )

    contexts = [
        item.content
        for item, _ in results
    ]

    context_text = "\n\n".join(contexts)

    prompt = f"""
Answer the question using only the provided context.

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = client.chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "system",
            "content": "Answer only using the provided context.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    options={
        "temperature": 0,
    },
)

    answer = response["message"]["content"]
    
    

    return answer, contexts


if __name__ == "__main__":
    questions = load_questions()

    print(f"Loaded {len(questions)} evaluation questions.")

    print("\nBuilding retrieval system...")
    vector_store, embedding_model, bm25, chunks = (
        build_hybrid_retriever()
    )

    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for index, row in enumerate(questions, start=1):
        question = row["question"]

        print(f"\nGenerating answer {index}/{len(questions)}")
        print(question)

        answer, contexts = generate_rag_output(
            question,
            vector_store,
            embedding_model,
            bm25,
            chunks,
        )

        evaluation_data["question"].append(question)
        evaluation_data["answer"].append(answer)
        evaluation_data["contexts"].append(contexts)
        evaluation_data["ground_truth"].append(
            row["ground_truth"]
        )

    dataset = Dataset.from_dict(evaluation_data)

    dataset.save_to_disk("data/processed/ragas_dataset")

    print("\nAll answers generated.")
    print("Saved evaluation dataset to:")
    print("data/processed/ragas_dataset")

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
)

    print("\n========== RAGAS RAW RESULT ==========")
    print(result)
    print("======================================")