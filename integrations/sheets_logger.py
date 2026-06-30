"""
integrations/sheets_logger.py
Appends one row to a Google Sheet after every pipeline run.
Over time this becomes a historical trend tracker showing how NVIDIA's
opportunity/risk profile evolves — something the dashboard cannot do
because ChromaDB only stores the latest snapshot.

Each row contains:
  Timestamp | Docs Indexed | Top Opportunity | Opp Impact |
  Top Risk | Risk Severity | Top Trend | Top Recommendation |
  Priority | CEO Briefing (truncated)

SETUP:
  1. Go to https://console.cloud.google.com
     → New Project → Enable "Google Sheets API" + "Google Drive API"
  2. IAM & Admin → Service Accounts → Create → Download JSON key
  3. Save the JSON key as: credentials/google_credentials.json
  4. Create a new Google Sheet, share it with the service account email
     (the 'client_email' field inside the JSON key) as Editor
  5. Copy the Sheet ID from the URL:
       https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit
  6. Add to .env:
       GOOGLE_SHEET_ID=your_sheet_id_here
       GOOGLE_CREDENTIALS_PATH=credentials/google_credentials.json

USAGE:
  from integrations.sheets_logger import log_pipeline_run
  log_pipeline_run(pipeline_state)
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# See main.py for why this is needed on Windows consoles.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# Row headers — written once when the sheet is empty
HEADERS = [
    "Timestamp",
    "Docs Indexed",
    "Top Opportunity",
    "Opp Impact",
    "Opp Confidence",
    "Top Risk",
    "Risk Severity",
    "Risk Confidence",
    "Top Trend",
    "Trend Momentum",
    "Top Recommendation",
    "Rec Priority",
    "Rec Risk Level",
    "CEO Briefing (excerpt)",
]


def _get_config() -> tuple[str, str]:
    """Return (sheet_id, credentials_path) or raise EnvironmentError."""
    sheet_id = os.environ.get("GOOGLE_SHEET_ID", "").strip()
    creds    = os.environ.get(
        "GOOGLE_CREDENTIALS_PATH",
        "credentials/google_credentials.json"
    ).strip()

    if not sheet_id:
        raise EnvironmentError(
            "GOOGLE_SHEET_ID is not set.\n"
            "Add to .env:  GOOGLE_SHEET_ID=your_sheet_id_here\n"
            "Get it from the Google Sheets URL."
        )
    if not os.path.exists(creds):
        raise EnvironmentError(
            f"Google credentials file not found: {creds}\n"
            "Download a service account JSON key from Google Cloud Console\n"
            "and save it to: credentials/google_credentials.json"
        )
    return sheet_id, creds


def _get_client(creds_path: str):
    """Return an authorised gspread client."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install gspread google-auth"
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(creds_path, scopes=scopes)
    return gspread.authorize(creds)


def build_row(state: dict) -> list:
    """
    Extract one flat row from the pipeline state.
    All values are strings — Sheets stores everything as text.
    """
    now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    idx_result = state.get("index_result") or {}
    doc_count  = str(idx_result.get("documents_indexed", ""))

    opps  = state.get("opportunities") or []
    risks = state.get("risks")         or []
    trends = state.get("trends")       or []
    recs  = state.get("recommendations") or []

    # Opportunities
    top_opp     = opps[0].get("title", "")[:120]        if opps   else ""
    opp_impact  = opps[0].get("impact_level",
                  opps[0].get("impact", ""))             if opps   else ""
    opp_conf    = str(round(opps[0].get("confidence_score", 0) * 100)) + "%" if opps else ""

    # Risks
    top_risk    = risks[0].get("title", "")[:120]        if risks  else ""
    risk_sev    = risks[0].get("severity_level",
                  risks[0].get("risk_level", ""))         if risks  else ""
    risk_conf   = str(round(risks[0].get("confidence_score", 0) * 100)) + "%" if risks else ""

    # Trends
    top_trend   = trends[0].get("trend_name", "")[:80]  if trends else ""
    trend_mom   = str(round(trends[0].get("relevance_score", 0) * 100)) + "%" if trends else ""

    # Recommendations
    top_rec     = recs[0].get("recommendation", "")[:150] if recs else ""
    rec_priority = recs[0].get("priority", "")             if recs else ""
    rec_risk    = recs[0].get("risk_level", "")            if recs else ""

    # CEO Briefing — first 300 chars as a quick preview in the sheet
    briefing    = (state.get("ceo_briefing") or "")[:300].replace("\n", " ").strip()

    return [
        now, doc_count,
        top_opp, opp_impact, opp_conf,
        top_risk, risk_sev, risk_conf,
        top_trend, trend_mom,
        top_rec, rec_priority, rec_risk,
        briefing,
    ]


def log_pipeline_run(state: dict) -> bool:
    """
    Append one row to the Google Sheet.

    Args:
        state: The final PipelineState dict returned by run_pipeline().

    Returns:
        True if logged successfully, False otherwise.
    """
    try:
        sheet_id, creds_path = _get_config()
    except EnvironmentError as e:
        print(f"[SheetsLogger] Skipping — {e}")
        return False

    try:
        client = _get_client(creds_path)
    except ImportError as e:
        print(f"[SheetsLogger] Skipping — {e}")
        return False

    try:
        sheet = client.open_by_key(sheet_id).sheet1

        # Write headers if the sheet is empty
        if sheet.row_count == 0 or not sheet.row_values(1):
            sheet.append_row(HEADERS, value_input_option="RAW")
            print("[SheetsLogger] Header row written")

        row = build_row(state)
        sheet.append_row(row, value_input_option="USER_ENTERED")

        now = row[0]
        print(f"[SheetsLogger] ✅ Row appended — {now}")
        return True

    except Exception as e:
        print(f"[SheetsLogger] ❌ Failed to write to sheet: {e}")
        return False


if __name__ == "__main__":
    # Quick test with dummy data — run: python integrations/sheets_logger.py
    dummy_state = {
        "index_result": {"documents_indexed": 528},
        "opportunities": [
            {"title": "AI Data Center Demand Surge", "impact_level": "High", "confidence_score": 0.85},
        ],
        "risks": [
            {"title": "US Export Controls on China", "severity_level": "High", "confidence_score": 0.78},
        ],
        "trends": [
            {"trend_name": "Data Center & Cloud", "relevance_score": 0.92},
        ],
        "recommendations": [
            {
                "recommendation": "Accelerate Blackwell production partnerships",
                "priority": "High",
                "risk_level": "Medium",
            }
        ],
        "ceo_briefing": (
            "NVIDIA is operating in a high-velocity AI market with surging data center demand. "
            "Blackwell GPU shipments are accelerating and hyperscaler capex commitments signal "
            "sustained infrastructure investment through 2026."
        ),
    }
    success = log_pipeline_run(dummy_state)
    print("Test result:", "PASSED" if success else "FAILED — check .env and credentials/")