"""
prompts.py — All LLM prompt templates.

Written to work well with open-weight instruction-tuned models (Llama 3,
Mistral, Phi-3, Gemma) which are less instruction-following than Claude.
Rules are stated explicitly and repeated in the user turn for reliability.
"""

RAG_SYSTEM_PROMPT = """You are a precise question-answering assistant.
You answer questions ONLY using the document excerpts provided to you.

STRICT RULES:
- Use ONLY the information in the context excerpts below. Do not use outside knowledge.
- Always cite your source inline using this exact format: [Source: filename, Page N]
- If the answer is not in the excerpts, respond with exactly:
  "I could not find relevant information in the uploaded documents."
- Do not make up facts. Do not guess. Do not add information not present in the context.
- Be concise and direct."""

RAG_USER_TEMPLATE = """Here are the relevant excerpts from the uploaded documents:

{context}

---
Using ONLY the excerpts above, answer this question and cite your sources inline:

Question: {question}

Answer:"""


GUARDRAIL_SYSTEM_PROMPT = ""  # unused — guardrails are now rule-based


QUERY_EXPANSION_PROMPT = """Generate 2 alternative phrasings of the query below.
Respond with ONLY a JSON array of 2 strings, no other text.

Query: {query}

["rephrasing 1", "rephrasing 2"]"""
