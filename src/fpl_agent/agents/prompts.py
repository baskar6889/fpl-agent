"""Prompt text for the provider-neutral ranking agent boundary."""

REQUEST_INTERPRETATION_PROMPT = """
Interpret the user's request only as a structured FPL player-ranking request.
Supported action: rank_players. Supported positions: DEF, MID, FWD.
Extract any stated maximum price and requested count exactly; do not change,
relax, or invent constraints. If the request is outside this scope, return an
action other than rank_players so deterministic guardrails reject it. If the
request is ambiguous, do not guess. Omit max_price when no price cap was given.
Use top_n=5 when no count was given. Do not rank players or calculate points.
""".strip()

RESPONSE_GENERATION_PROMPT = """
Describe only the supplied validated ranking result or guardrail error.
Do not calculate, reorder, add, remove, or alter players, prices, projections,
ranks, or constraints. For errors, explain the supplied error clearly and do
not suggest relaxing the user's constraints.
""".strip()