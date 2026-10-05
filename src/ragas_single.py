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


def get_first_question():
    with open(
        "data/evaluation/golden_dataset.csv",
        "r",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        return next(reader)


def generate_rag_output(question):
    vector_store, embedding_model, bm25, chunks = (
        build_hybrid_retriever()
    )

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
    row = get_first_question()

    question = row["question"]
    ground_truth = row["ground_truth"]

    answer, contexts = generate_rag_output(question)

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print("\nGround Truth:")
    print(ground_truth)

    print("\nRunning RAGAS evaluation...")

    faithfulness.llm = ragas_llm
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            context_precision,
            context_recall,
        ],
    )

    print("\nRAGAS Results:")
    print(result)