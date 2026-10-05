from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, results, top_k=3):
        pairs = [
            (query, item.content)
            for item, _ in results
        ]

        scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (item, float(score))
            for (item, _), score in ranked_results[:top_k]
        ]