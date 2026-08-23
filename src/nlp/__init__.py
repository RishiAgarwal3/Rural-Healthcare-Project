"""
NLP post-processing pipeline package.

Stages:
  1. clean_bengali    — Heuristic cleaning + quality scoring
  2. translate_english — Bengali → English (NLLB / OpenAI / Anthropic)
  3. format_enriched  — Reshape into analytics-ready schema
"""
