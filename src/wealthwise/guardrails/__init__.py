"""wealthwise.guardrails — three-layer rule-based guardrail pipeline.

input   → screen_profile: validate investor profile + injection detection
process → cap_candidates: dedupe, clean, and truncate candidate lists
output  → enforce_output: disclosure completeness + misleading-language neutralization
"""
