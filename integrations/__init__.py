"""
integrations/
Post-pipeline output integrations for the NVIDIA Strategic Intelligence Agent.

After every pipeline run, results are pushed to:
  - Slack:         CEO Briefing + top opportunities/risks → #nvidia-intelligence channel
  - Google Sheets: One row appended per run → historical trend tracker

Modules:
    slack_notifier.py  — post_intelligence_brief(state)
    sheets_logger.py   — log_pipeline_run(state)

Both are optional — they skip gracefully if credentials are missing.
"""

from integrations.slack_notifier import post_intelligence_brief
from integrations.sheets_logger  import log_pipeline_run

__all__ = ["post_intelligence_brief", "log_pipeline_run"]