import csv

from rank_bm25 import BM25Okapi

from .chunkers import sentence_chunks
from .loaders import load_document


def build_bm25_retriever():
    document = load_document("data/raw/sample.txt")
    chunks = sentence_chunks(document)

    texts = [chunk.content for chunk in chunks]
    tokenized_texts = [text.lower().split() for text in texts]

    bm25 = BM25Okapi(tokenized_texts)

    return bm25, chunks


def evaluate_bm25(bm25, chunks, evaluation_path, top_k=3):
    correct = 0
    total = 0

    with open(evaluation_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query = row["question"]
            expected_text = row["expected_text"]

            tokenized_query = query.lower().split()

            scores = bm25.get_scores(tokenized_query)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            retrieved_text = " ".join(
                chunks[index].content
                for index in ranked_indices
            )

            if expected_text.lower() in retrieved_text.lower():
                correct += 1

            total += 1
            
    return correct / total


if __name__ == "__main__":
    bm25, chunks = build_bm25_retriever()

    recall = evaluate_bm25(
        bm25,
        chunks,
        "data/evaluation/retrieval_questions.csv",
        top_k=3,
    )

    print(f"BM25 Retrieval Recall@3: {recall:.3f}")
    