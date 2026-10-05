from ollama import Client

from .hybrid_retriever import build_hybrid_retriever, hybrid_search


client = Client(host="http://localhost:11434")


def generate_answer(question: str) -> str:
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

    context = "\n\n".join(
        item.content
        for item, _ in results
    )

    prompt = f"""
Answer the question using only the provided context.

If the context does not contain enough information, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers "
                    "questions using only the provided context."
                ),
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

    return response["message"]["content"]


if __name__ == "__main__":
    question = "What does CO2e mean?"
    answer = generate_answer(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)