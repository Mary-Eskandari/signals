import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# A small, user-facing set spanning quality/cost tiers (fast/cheap -> balanced -> highest
# quality), not the full model catalog. Sonnet 5 is the default: strong quality at
# reasonable cost for report generation. OpenAI support may be added later — kept out of
# scope for now, but the report-generation call is isolated to one function (below) so
# adding a second provider later won't require touching the pipeline/grounding logic.
ALLOWED_REPORT_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8"]
REPORT_MODEL = os.environ.get("ANTHROPIC_REPORT_MODEL", "claude-sonnet-5")
