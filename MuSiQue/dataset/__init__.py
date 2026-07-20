"""CogniFold trajectory dataset generation.

Produces JSONL trajectory datasets by running the CogniFold agent
on benchmark data and capturing every LLM call (input prompts +
output UpdatePlan operations).
"""
