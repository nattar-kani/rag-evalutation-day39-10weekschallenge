## RAG Evaluation

A 50-question golden evaluation dataset was created covering:

- GHG fundamentals
- Scope 1, 2 and 3 emissions
- Emission factors
- GHG calculations
- Units and conversions
- Reporting methodology
- Data quality and auditability
- Source and geography traceability

The same golden dataset was used to evaluate the RAG pipeline using RAGAS.

### Baseline Results

| Metric | Score |
|---|---:|
| Faithfulness | 0.9533 |
| Context Precision | 0.9900 |
| Context Recall | 0.9599 |
| Answer Relevancy | Not evaluated |

The evaluation was run locally using **Ollama with Qwen 2.5 1.5B** as the evaluation LLM and local `all-MiniLM-L6-v2` embeddings.

RAGAS evaluation was configured with a single worker because parallel evaluation caused timeouts with the local 1.5B model.

### Interpretation

The baseline results indicate that the retrieval pipeline is already performing strongly on this evaluation set:

- **Faithfulness (0.9533):** Most generated answers were supported by the retrieved context.
- **Context Precision (0.9900):** Retrieved chunks were highly relevant to the questions.
- **Context Recall (0.9599):** The retrieval system generally returned the information required to answer the questions.

Because the baseline scores are already close to 1.0, a 15% relative improvement is not realistically achievable for these metrics without exceeding their maximum score of 1.0.

Rather than artificially changing the benchmark or claiming an improvement that cannot be measured fairly, the current results are retained as the baseline.

## Potential Improvements

Several improvements could be explored in a future iteration:

### 1. Improve Faithfulness

Strengthen the generation prompt to require:

- answers based strictly on retrieved context
- no unsupported assumptions
- explicit acknowledgement when information is missing
- citations or source references for factual claims

### 2. Improve Context Recall

Experiment with:

- larger retrieval `top_k`
- query rewriting
- multi-query retrieval
- parent-document retrieval
- improved chunking strategies

The goal would be to reduce cases where the information required to answer a question is not retrieved.

### 3. Improve Context Precision

Although the baseline is already very high, precision could potentially be improved through:

- cross-encoder re-ranking
- stronger metadata filtering
- better hybrid-search weighting
- removing redundant chunks

### 4. Evaluate Answer Relevancy

Answer relevancy was not included in the final scorecard because the local Qwen 2.5 1.5B evaluator was unable to reliably produce the structured output required by the RAGAS metric.

A future experiment could evaluate this metric using a stronger local judge model.


