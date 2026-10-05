from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper


ollama_llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
)

ragas_llm = LangchainLLMWrapper(ollama_llm)