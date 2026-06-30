"""
integrations/slack_notifier.py
Posts the CEO Briefing, top opportunities, and top risks to a Slack channel
after every pipeline run.

SETUP:
  1. Go to https://api.slack.com/apps → Create New App → From Scratch
  2. Under "Incoming Webhooks" → Activate → Add New Webhook to Workspace
  3. Choose your channel (e.g. #nvidia-intelligence) → Copy the webhook URL
  4. Add to .env:
       SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ

USAGE (called automatically by main.py after pipeline completes):
  from integrations.slack_notifier import post_intelligence_brief
  post_intelligence_brief(pipeline_state)
"""

import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# See main.py for why this is needed on Windows consoles.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


def _get_webhook() -> str:
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        raise EnvironmentError(
            "SLACK_WEBHOOK_URL is not set.\n"
            "Get a webhook at: https://api.slack.com/apps\n"
            "Then add to .env:  SLACK_WEBHOOK_URL=https://hooks.slack.com/..."
        )
    return url


def _severity_emoji(level: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")


def _impact_emoji(level: str) -> str:
    return {"High": "🚀", "Medium": "📈", "Low": "➡️"}.get(level, "➡️")


def build_slack_payload(state: dict) -> dict:
    now        = datetime.now().strftime("%d %b %Y · %H:%M")
    briefing   = (state.get("ceo_briefing") or "").strip()
    opps       = state.get("opportunities") or []
    risks      = state.get("risks")         or []
    recs       = state.get("recommendations") or []
    idx_result = state.get("index_result")  or {}
    doc_count  = idx_result.get("documents_indexed", "?")

    # Deduplicate by title
    seen = set()
    opps_deduped, risks_deduped = [], []
    for o in opps:
        t = o.get("title", "")
        if t not in seen:
            seen.add(t)
            opps_deduped.append(o)
    seen.clear()
    for r in risks:
        t = r.get("title", "")
        if t not in seen:
            seen.add(t)
            risks_deduped.append(r)

    # Briefing — first 2 sentences only for Slack
    sentences = briefing.replace("\n", " ").split(". ")
    brief_short = ". ".join(sentences[:2]).strip()
    if brief_short and not brief_short.endswith("."):
        brief_short += "."

    # Opportunities
    opp_lines = []
    for o in opps_deduped[:3]:
        impact = o.get("impact_level", o.get("impact", "—"))
        emoji  = {"High": "🟢", "Medium": "🟡", "Low": "🔵"}.get(impact, "⚪")
        title  = o.get("title", "")[:70]
        opp_lines.append(f"{emoji}  {title}  ·  _{impact} impact_")
    opp_text = "\n".join(opp_lines) or "_None detected_"

    # Risks
    risk_lines = []
    for r in risks_deduped[:3]:
        sev   = r.get("severity_level") or r.get("severity") or r.get("risk_level") or "—"
        emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(sev, "⚪")
        title = r.get("title", "")[:70]
        risk_lines.append(f"{emoji}  {title}  ·  _{sev} severity_")
    risk_text = "\n".join(risk_lines) or "_None detected_"

    # Top recommendation
    rec_text = "_No recommendations generated_"
    if recs:
        r0 = recs[0]
        rec_text = (
            f"*{r0.get('recommendation', '')[:100]}*\n"
            f"Priority: *{r0.get('priority','')}*  ·  "
            f"Risk: *{r0.get('risk_level','')}*  ·  "
            f"Timeline: *{r0.get('expected_impact',{}).get('timeline','')}*"
        )

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "NVIDIA · Strategic Intelligence Brief", "emoji": True},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn",
                "text": f"*{now}*  ·  {doc_count} documents analysed  ·  Mistral-7B + ChromaDB + LangGraph"}],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*SITUATION*\n{brief_short}"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*OPPORTUNITIES*\n{opp_text}"},
                {"type": "mrkdwn", "text": f"*RISKS*\n{risk_text}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*TOP RECOMMENDATION*\n{rec_text}"},
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "NVIDIA Strategic Intelligence Agent"}],
        },
    ]

    return {"blocks": blocks}


def post_intelligence_brief(state: dict) -> bool:
    """
    Post the pipeline results to Slack.

    Args:
        state: The final PipelineState dict returned by run_pipeline().

    Returns:
        True if posted successfully, False otherwise.
    """
    try:
        webhook = _get_webhook()
    except EnvironmentError as e:
        print(f"[SlackNotifier] Skipping — {e}")
        return False

    payload = build_slack_payload(state)

    try:
        response = requests.post(
            webhook,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if response.status_code == 200:
            print("[SlackNotifier] ✅ Brief posted to Slack successfully")
            return True
        else:
            print(f"[SlackNotifier] ❌ Slack returned {response.status_code}: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"[SlackNotifier] ❌ Request failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test with dummy data — run: python integrations/slack_notifier.py
    dummy_state = {
        "index_result": {"documents_indexed": 528},
        "ceo_briefing": (
            "NVIDIA is operating in a high-velocity AI market with surging data center demand. "
            "Blackwell GPU shipments are accelerating, and hyperscaler capex commitments signal "
            "sustained infrastructure investment through 2026. Geopolitical headwinds from US export "
            "controls remain the primary risk to revenue guidance.\n\n"
            "Management should immediately prioritise three actions: accelerate Blackwell supply chain "
            "partnerships, deepen sovereign AI relationships in non-restricted markets, and establish "
            "a dedicated regulatory scenario planning function."
        ),
        "opportunities": [
            {"title": "AI Data Center Demand Surge", "impact_level": "High"},
            {"title": "Sovereign AI Partnerships", "impact_level": "High"},
            {"title": "Healthcare AI Expansion", "impact_level": "Medium"},
        ],
        "risks": [
            {"title": "US Export Controls on China", "severity": "High"},
            {"title": "AMD Competitive Pressure", "severity": "Medium"},
            {"title": "Supply Chain Concentration", "severity": "Medium"},
        ],
        "recommendations": [
            {
                "recommendation": "Accelerate Blackwell production partnerships",
                "priority": "High",
                "risk_level": "Medium",
                "expected_impact": {"timeline": "short-term (0-6mo)"},
            }
        ],
    }
    success = post_intelligence_brief(dummy_state)
    print("Test result:", "PASSED" if success else "FAILED — check SLACK_WEBHOOK_URL in .env")
