"""
FundFlow AI — Complete Backend
================================
Project  : FundFlow AI
Purpose  : Agentic AI Grant and Funding Intelligence Assistant for Startups
Problem  : IBM AICTE Internship — Problem Statement #18
Stack    : IBM Granite (watsonx.ai REST API) · Flask · Python
Security : All credentials loaded exclusively from .env / environment variables.
           No secret values are hardcoded, logged, or exposed to the frontend.
"""

import os
import json
import logging
import time
import re
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════
# 1. ENVIRONMENT & LOGGING
# ═══════════════════════════════════════════════════════════════

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fundflow")

# ═══════════════════════════════════════════════════════════════
# 2. CREDENTIAL ACCESSORS  (never stored as module constants)
# ═══════════════════════════════════════════════════════════════

def _get_env(key: str, warn: bool = True) -> str:
    val = os.getenv(key, "").strip()
    if not val and warn:
        logger.warning("Environment variable %s is not set.", key)
    return val


# ═══════════════════════════════════════════════════════════════
# 3. IBM IAM TOKEN CACHE
#    watsonx.ai uses IBM Cloud IAM bearer tokens.
#    We cache the token to avoid re-authenticating on every call.
# ═══════════════════════════════════════════════════════════════

_iam_token_cache: dict = {"token": None, "expires_at": 0.0}

IBM_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
IAM_TOKEN_BUFFER_SECONDS = 120   # refresh this many seconds before expiry


def _get_iam_token() -> str:
    """
    Obtain (or return cached) IBM IAM bearer token using the API key from .env.
    Raises RuntimeError if the API key is missing or IBM IAM returns an error.
    Never logs or returns the raw API key.
    """
    now = time.time()
    if _iam_token_cache["token"] and now < _iam_token_cache["expires_at"] - IAM_TOKEN_BUFFER_SECONDS:
        return _iam_token_cache["token"]

    api_key = _get_env("IBM_API_KEY")
    if not api_key:
        raise RuntimeError("IBM_API_KEY is not set. Cannot authenticate with IBM Cloud.")

    try:
        resp = requests.post(
            IBM_IAM_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        _iam_token_cache["token"] = payload["access_token"]
        _iam_token_cache["expires_at"] = now + int(payload.get("expires_in", 3600))
        logger.info("IBM IAM token obtained/refreshed successfully.")
        return _iam_token_cache["token"]
    except requests.exceptions.Timeout:
        raise RuntimeError("Timeout while contacting IBM IAM token service.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to IBM IAM service. Check internet connection.")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        raise RuntimeError(f"IBM IAM authentication failed (HTTP {status}). Check IBM_API_KEY.")
    except Exception as e:
        raise RuntimeError(f"IBM IAM token error: {type(e).__name__}")


# ═══════════════════════════════════════════════════════════════
# 4. IBM GRANITE CLIENT
#    Sends a chat-completions style request to watsonx.ai
#    Endpoint: /ml/v1/text/chat?version=...
# ═══════════════════════════════════════════════════════════════

def _validate_uuid(value: str) -> bool:
    """Return True if value looks like a UUID v4 (basic check)."""
    import re as _re
    return bool(_re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        value.lower()
    ))


def call_granite(
    messages: list,
    max_new_tokens: int = 800,
    temperature: float = 0.3,
) -> dict:
    """
    Call IBM Granite via the watsonx.ai /ml/v1/text/chat REST endpoint.

    Parameters
    ----------
    messages        : list of {"role": "system"|"user"|"assistant", "content": str}
    max_new_tokens  : maximum tokens to generate
    temperature     : sampling temperature (lower = more deterministic)

    Returns
    -------
    dict with keys:
        "text"          : generated text (str)
        "input_tokens"  : int or None
        "output_tokens" : int or None
        "total_tokens"  : int or None
        "model_id"      : str
        "error"         : None or error message string
    """
    url      = _get_env("IBM_GRANITE_URL")
    model_id = _get_env("IBM_GRANITE_MODEL_ID")

    if not url or not model_id:
        return {
            "text": "",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "model_id": model_id or "unknown",
            "error": "IBM Granite credentials are not fully configured (check .env).",
        }

    # Clean model_id — strip surrounding quotes if user left them
    model_id = model_id.strip('"').strip("'")
    # Clean URL — strip surrounding quotes if present
    url = url.strip('"').strip("'")

    try:
        token = _get_iam_token()
    except RuntimeError as e:
        return {"text": "", "input_tokens": None, "output_tokens": None,
                "total_tokens": None, "model_id": model_id, "error": str(e)}

    # watsonx.ai requires project_id (UUID v4) or space_id
    project_id = _get_env("IBM_PROJECT_ID", warn=False)
    space_id    = _get_env("IBM_SPACE_ID", warn=False)

    # Validate: reject obvious placeholders
    if project_id and not _validate_uuid(project_id):
        logger.warning(
            "IBM_PROJECT_ID does not look like a valid UUID v4 (%s). "
            "Go to IBM Cloud → watsonx.ai → your project → Manage → copy the Project ID.",
            project_id[:8] + "…" if len(project_id) > 8 else project_id,
        )
        return {
            "text": "", "input_tokens": None, "output_tokens": None,
            "total_tokens": None, "model_id": model_id,
            "error": (
                "IBM_PROJECT_ID is not a valid UUID. "
                "Find your Project ID in IBM Cloud → watsonx.ai → your project → Manage tab, "
                "then set IBM_PROJECT_ID=<uuid> in your .env file."
            ),
        }

    payload = {
        "model_id": model_id,
        "messages": messages,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }
    if project_id:
        payload["project_id"] = project_id
    elif space_id:
        payload["space_id"] = space_id
    else:
        return {
            "text": "", "input_tokens": None, "output_tokens": None,
            "total_tokens": None, "model_id": model_id,
            "error": (
                "IBM_PROJECT_ID is not set in .env. "
                "Find your Project ID in IBM Cloud → watsonx.ai → your project → Manage tab."
            ),
        }

    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract generated text (chat completions format)
        text = ""
        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")

        # Extract token usage if available
        usage = data.get("usage", {})
        input_tokens  = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens  = usage.get("total_tokens")

        return {
            "text": text.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "model_id": model_id,
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {"text": "", "input_tokens": None, "output_tokens": None,
                "total_tokens": None, "model_id": model_id,
                "error": "Request to IBM Granite timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"text": "", "input_tokens": None, "output_tokens": None,
                "total_tokens": None, "model_id": model_id,
                "error": "Cannot connect to IBM Granite endpoint. Check network and IBM_GRANITE_URL."}
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        body_preview = ""
        if e.response is not None:
            try:
                body_preview = e.response.json().get("errors", [{}])[0].get("message", "")
            except Exception:
                pass
        return {"text": "", "input_tokens": None, "output_tokens": None,
                "total_tokens": None, "model_id": model_id,
                "error": f"IBM Granite HTTP {status}: {body_preview or 'request failed'}"}
    except Exception as e:
        return {"text": "", "input_tokens": None, "output_tokens": None,
                "total_tokens": None, "model_id": model_id,
                "error": f"Unexpected error calling IBM Granite: {type(e).__name__}"}


def test_granite_connection() -> dict:
    """Lightweight connectivity test — returns status dict without logging secrets."""
    result = call_granite(
        messages=[{"role": "user", "content": "Reply with exactly: FundFlow AI connected."}],
        max_new_tokens=20,
        temperature=0.0,
    )
    return {
        "connected": result["error"] is None,
        "model_id": result["model_id"],
        "error": result["error"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


# ═══════════════════════════════════════════════════════════════
# 5. FUNDING KNOWLEDGE BASE (RAG DATA LAYER)
# ═══════════════════════════════════════════════════════════════

_funding_kb: list = []


def load_funding_kb() -> list:
    """Load the curated funding knowledge base from data/funding_kb.json."""
    global _funding_kb
    if _funding_kb:
        return _funding_kb

    kb_path = Path(__file__).parent / "data" / "funding_kb.json"
    if not kb_path.exists():
        logger.error("Funding knowledge base not found at %s", kb_path)
        return []

    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            _funding_kb = json.load(f)
        logger.info("Loaded %d funding records from knowledge base.", len(_funding_kb))
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to load funding knowledge base: %s", e)
        _funding_kb = []

    return _funding_kb


def retrieve_relevant_opportunities(profile: dict, top_k: int = 8) -> list:
    """
    Retrieve funding opportunities from the knowledge base relevant to the startup profile.
    Uses keyword and attribute matching — no vector embeddings needed for this scale.

    Returns a list of opportunity dicts ranked by basic relevance score.
    """
    kb = load_funding_kb()
    if not kb:
        return []

    domain   = (profile.get("domain") or "").lower()
    stage    = (profile.get("stage") or "").lower()
    location = (profile.get("location") or "").lower()
    revenue  = (profile.get("revenue_status") or "pre-revenue").lower()
    desc     = (profile.get("description") or "").lower()

    # Build a location normalization map
    location_map = {
        "india": "India", "indian": "India",
        "us": "United States", "usa": "United States", "united states": "United States", "america": "United States",
        "eu": "European Union", "europe": "European Union", "european union": "European Union",
        "uk": "United Kingdom", "united kingdom": "United Kingdom",
    }
    normalized_location = location_map.get(location, location.title())

    scored = []
    for opp in kb:
        score = 0

        # Stage match (up to 30 pts)
        eligible_stages = [s.lower() for s in opp.get("eligible_stages", [])]
        if stage in eligible_stages:
            score += 30
        elif eligible_stages:
            # Adjacent stage tolerance
            stage_order = ["idea", "pre-seed", "seed", "series-a", "series-b"]
            try:
                idx_opp = min(stage_order.index(s) for s in eligible_stages if s in stage_order)
                idx_profile = stage_order.index(stage) if stage in stage_order else -1
                if idx_profile >= 0 and abs(idx_opp - idx_profile) <= 1:
                    score += 15
            except ValueError:
                pass

        # Domain match (up to 30 pts)
        eligible_domains = [d.lower() for d in opp.get("eligible_domains", [])]
        if "all" in eligible_domains:
            score += 20
        elif domain and domain in eligible_domains:
            score += 30
        elif domain:
            # Partial domain match check
            for ed in eligible_domains:
                if domain in ed or ed in domain:
                    score += 15
                    break
            # Check description keyword overlap
            opp_desc = opp.get("description", "").lower()
            domain_keywords = domain.replace("tech", "").replace("-", " ").split()
            for kw in domain_keywords:
                if kw and len(kw) > 3 and kw in opp_desc:
                    score += 5
                    break

        # Location match (up to 25 pts)
        eligible_locations = [loc.lower() for loc in opp.get("eligible_locations", [])]
        if "global" in eligible_locations:
            score += 20
        elif normalized_location.lower() in eligible_locations:
            score += 25
        elif location:
            for el in eligible_locations:
                if location in el or el in location:
                    score += 15
                    break

        # Revenue status match (up to 10 pts)
        eligible_revenue = [r.lower() for r in opp.get("eligible_revenue_status", [])]
        if revenue in eligible_revenue or "all" in eligible_revenue:
            score += 10

        # Description keyword overlap (up to 5 pts)
        if desc:
            opp_combined = (opp.get("description", "") + " " + opp.get("eligibility_summary", "")).lower()
            desc_words = set(w for w in re.split(r'\W+', desc) if len(w) > 4)
            opp_words  = set(w for w in re.split(r'\W+', opp_combined) if len(w) > 4)
            overlap = len(desc_words & opp_words)
            score += min(overlap, 5)

        if score > 0:
            scored.append((score, opp))

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [opp for _, opp in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════
# 6. PROMPT BUILDERS  (concise, no repetition, structured output)
# ═══════════════════════════════════════════════════════════════

def _profile_summary(profile: dict) -> str:
    """Produce a compact plain-text summary of the startup profile for prompts."""
    parts = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("domain"):
        parts.append(f"Domain: {profile['domain']}")
    if profile.get("stage"):
        parts.append(f"Stage: {profile['stage']}")
    if profile.get("location"):
        parts.append(f"Location: {profile['location']}")
    if profile.get("funding_need"):
        parts.append(f"Funding need: USD {profile['funding_need']:,}")
    if profile.get("revenue_status"):
        parts.append(f"Revenue status: {profile['revenue_status']}")
    if profile.get("description"):
        parts.append(f"Description: {profile['description'][:300]}")
    if profile.get("target_users"):
        parts.append(f"Target users: {profile['target_users'][:150]}")
    if profile.get("use_of_funds"):
        parts.append(f"Use of funds: {profile['use_of_funds'][:150]}")
    return "\n".join(parts)


def _opportunity_summary(opp: dict) -> str:
    """Produce a compact plain-text summary of a single opportunity for prompts."""
    return (
        f"Name: {opp.get('name', 'Unknown')}\n"
        f"Organization: {opp.get('organization', '')}\n"
        f"Type: {opp.get('type', '')}\n"
        f"Amount: {opp.get('amount_display', 'Not specified')}\n"
        f"Stages: {', '.join(opp.get('eligible_stages', []))}\n"
        f"Domains: {', '.join(opp.get('eligible_domains', []))}\n"
        f"Locations: {', '.join(opp.get('eligible_locations', []))}\n"
        f"Eligibility summary: {opp.get('eligibility_summary', '')[:400]}\n"
        f"Description: {opp.get('description', '')[:400]}\n"
        f"Deadline notes: {opp.get('deadline_notes', '')}"
    )


# ═══════════════════════════════════════════════════════════════
# 7. FUNDING DISCOVERY AGENT
# ═══════════════════════════════════════════════════════════════

class FundingDiscoveryAgent:
    """
    Retrieves relevant funding opportunities from the knowledge base using
    profile-attribute matching, then uses IBM Granite to enrich each result
    with a concise AI-generated relevance note.
    """

    def run(self, profile: dict) -> tuple[list, dict]:
        """
        Returns (opportunities, token_usage_dict)
        Each opportunity gets an ai_note field added.
        """
        logger.info("[DiscoveryAgent] Starting for: %s", profile.get("name", "unknown"))
        retrieved = retrieve_relevant_opportunities(profile, top_k=8)

        if not retrieved:
            logger.info("[DiscoveryAgent] No relevant opportunities found in KB.")
            return [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        logger.info("[DiscoveryAgent] Retrieved %d candidates from KB.", len(retrieved))

        # Enrich with AI summary (single batched call to reduce tokens)
        opp_list_text = "\n\n".join(
            f"[{i+1}] {o.get('name')} | {o.get('type')} | {o.get('amount_display')}"
            for i, o in enumerate(retrieved)
        )
        profile_text = _profile_summary(profile)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a funding analyst assistant. "
                    "Given a startup profile and a list of funding opportunities, "
                    "write one short sentence (15-25 words) for each opportunity explaining "
                    "why it could be relevant to this startup. "
                    "Respond ONLY with numbered lines: '1. <sentence>' through to the last number. "
                    "No extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"STARTUP PROFILE:\n{profile_text}\n\n"
                    f"OPPORTUNITIES:\n{opp_list_text}\n\n"
                    "Write one-sentence relevance notes for each."
                ),
            },
        ]

        result = call_granite(messages, max_new_tokens=300, temperature=0.3)
        ai_notes = _parse_numbered_list(result["text"], len(retrieved))

        for i, opp in enumerate(retrieved):
            opp["ai_note"] = ai_notes[i] if i < len(ai_notes) else ""

        usage = {
            "input_tokens":  result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens":  result["total_tokens"],
        }
        return retrieved, usage


# ═══════════════════════════════════════════════════════════════
# 8. ELIGIBILITY ANALYSIS AGENT
# ═══════════════════════════════════════════════════════════════

class EligibilityAnalysisAgent:
    """
    Evaluates each opportunity against the startup profile using IBM Granite.
    Produces: verdict (Eligible / Possibly Eligible / Not Eligible) + explanation.
    """

    def run(self, profile: dict, opportunities: list) -> tuple[list, dict]:
        """Returns (opportunities_with_eligibility, token_usage_dict)."""
        if not opportunities:
            return [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        logger.info("[EligibilityAgent] Analysing %d opportunities.", len(opportunities))
        profile_text = _profile_summary(profile)
        total_usage  = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Batch eligibility in a single call (more token-efficient than one per opp)
        opp_summaries = "\n\n---\n".join(
            f"[{i+1}] NAME: {o.get('name')}\n"
            f"Eligibility criteria: {o.get('eligibility_summary', '')[:300]}\n"
            f"Eligible stages: {', '.join(o.get('eligible_stages', []))}\n"
            f"Eligible locations: {', '.join(o.get('eligible_locations', []))}\n"
            f"Eligible domains: {', '.join(o.get('eligible_domains', []))}"
            for i, o in enumerate(opportunities)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an eligibility analyst for startup funding. "
                    "For each numbered opportunity, assess eligibility based on the startup profile. "
                    "Respond in this EXACT format for each:\n"
                    "[N] VERDICT: <Eligible|Possibly Eligible|Not Eligible>\n"
                    "REASON: <one sentence explanation, max 30 words>\n"
                    "GAP: <one key missing requirement or 'None'>\n"
                    "Do not add any other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"STARTUP PROFILE:\n{profile_text}\n\n"
                    f"OPPORTUNITIES TO EVALUATE:\n{opp_summaries}"
                ),
            },
        ]

        result = call_granite(messages, max_new_tokens=500, temperature=0.1)
        _accumulate_usage(total_usage, result)

        eligibility_data = _parse_eligibility_response(result["text"], len(opportunities))

        for i, opp in enumerate(opportunities):
            ed = eligibility_data[i] if i < len(eligibility_data) else {}
            opp["eligibility"]        = ed.get("verdict", "possibly-eligible").lower().replace(" ", "-")
            opp["eligibility_label"]  = ed.get("verdict", "Possibly Eligible")
            opp["eligibility_reason"] = ed.get("reason", "")
            opp["eligibility_gap"]    = ed.get("gap", "")

        return opportunities, total_usage


# ═══════════════════════════════════════════════════════════════
# 9. FUNDING MATCH & RANKING AGENT
# ═══════════════════════════════════════════════════════════════

class FundingMatchRankingAgent:
    """
    Ranks opportunities by a composite AI relevance score using IBM Granite.
    Scores are AI-estimated relevance indicators — not mathematical guarantees.
    """

    def run(self, profile: dict, opportunities: list) -> tuple[list, dict]:
        """Returns (ranked_opportunities, token_usage_dict)."""
        if not opportunities:
            return [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        logger.info("[RankingAgent] Ranking %d opportunities.", len(opportunities))
        profile_text = _profile_summary(profile)

        opp_list = "\n".join(
            f"[{i+1}] {o.get('name')} | Stage: {','.join(o.get('eligible_stages',[]))} | "
            f"Domain: {','.join(o.get('eligible_domains',[]))} | "
            f"Location: {','.join(o.get('eligible_locations',[]))} | "
            f"Eligibility: {o.get('eligibility_label', 'Unknown')}"
            for i, o in enumerate(opportunities)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a startup funding advisor. "
                    "Score each opportunity 0-100 for overall fit with the startup profile. "
                    "Consider: stage fit, domain fit, location match, funding amount fit, eligibility. "
                    "Respond ONLY with lines: '[N] SCORE: <0-100> | REASON: <10-15 word explanation>'\n"
                    "Do not add any other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"STARTUP PROFILE:\n{profile_text}\n\n"
                    f"OPPORTUNITIES:\n{opp_list}"
                ),
            },
        ]

        result = call_granite(messages, max_new_tokens=400, temperature=0.1)
        scores = _parse_score_response(result["text"], len(opportunities))

        for i, opp in enumerate(opportunities):
            s = scores[i] if i < len(scores) else {}
            opp["match_score"]  = s.get("score", 50)
            opp["match_reason"] = s.get("reason", "")

        # Sort by: eligible first, then match_score descending
        eligibility_order = {"eligible": 0, "possibly-eligible": 1, "not-eligible": 2}
        opportunities.sort(
            key=lambda o: (
                eligibility_order.get(o.get("eligibility", "possibly-eligible"), 1),
                -(o.get("match_score") or 0),
            )
        )

        usage = {
            "input_tokens":  result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens":  result["total_tokens"],
        }
        return opportunities, usage


# ═══════════════════════════════════════════════════════════════
# 10. VERIFICATION AGENT
# ═══════════════════════════════════════════════════════════════

class VerificationAgent:
    """
    Grounds each opportunity in its source data.
    Flags fields that should be verified before application.
    Does NOT fabricate or validate real-time external data.
    """

    # Fields in KB that are considered source-grounded
    GROUNDED_FIELDS = {
        "organization", "type", "amount_display", "application_url",
        "source_url", "last_verified",
    }

    def run(self, opportunities: list) -> list:
        """Attach verification metadata to each opportunity."""
        for opp in opportunities:
            opp["source_grounded"]      = bool(opp.get("source_url") or opp.get("application_url"))
            opp["verification_warning"] = (
                "Verify current deadlines, eligibility requirements, and application procedures "
                "directly on the official source before applying. Information may have changed."
            )
            opp["last_verified"]        = opp.get("last_verified", "Unknown")
            opp["application_url"]      = opp.get("application_url", "")
            opp["source_url"]           = opp.get("source_url", "")

        return opportunities


# ═══════════════════════════════════════════════════════════════
# 11. PROPOSAL ASSISTANT AGENT
# ═══════════════════════════════════════════════════════════════

class ProposalAssistantAgent:
    """
    Uses IBM Granite to generate a tailored draft proposal component.
    Output is clearly labelled as an AI DRAFT requiring human review.
    """

    SECTIONS = [
        "Executive Summary",
        "Problem Statement",
        "Proposed Solution",
        "Innovation & Differentiation",
        "Target Users & Market",
        "Funding Requirement & Use of Funds",
        "Expected Outcomes",
    ]

    def run(self, profile: dict, opportunity: dict) -> tuple[str, dict]:
        """Returns (draft_text, token_usage_dict)."""
        logger.info("[ProposalAgent] Generating draft for: %s", opportunity.get("name", "?"))

        profile_text = _profile_summary(profile)
        opp_text     = _opportunity_summary(opportunity)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert grant writing assistant. "
                    "Write a structured draft proposal for a startup applying to a funding opportunity. "
                    "Use these sections: " + ", ".join(self.SECTIONS) + ". "
                    "Each section should be 2-4 sentences. "
                    "Be specific to the startup and opportunity. "
                    "Begin the response with '⚠️ DRAFT — For review only. Verify all facts before submission.'\n"
                    "Use markdown headings (##) for each section."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"STARTUP PROFILE:\n{profile_text}\n\n"
                    f"FUNDING OPPORTUNITY:\n{opp_text}\n\n"
                    "Write the tailored draft proposal."
                ),
            },
        ]

        result = call_granite(messages, max_new_tokens=900, temperature=0.5)
        draft = result["text"]

        if not draft:
            draft = (
                "⚠️ DRAFT — For review only.\n\n"
                "The proposal could not be generated at this time. "
                "Please check the IBM Granite connection and try again.\n\n"
                f"Error: {result.get('error', 'No response from model.')}"
            )

        usage = {
            "input_tokens":  result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens":  result["total_tokens"],
        }
        return draft, usage


# ═══════════════════════════════════════════════════════════════
# 12. AGENTIC ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class FundFlowOrchestrator:
    """
    Coordinates the complete agentic pipeline:
    Discovery → Eligibility → Ranking → Verification → (Proposal on-demand)
    Aggregates token usage across all agents.
    """

    def __init__(self):
        self.discovery_agent    = FundingDiscoveryAgent()
        self.eligibility_agent  = EligibilityAnalysisAgent()
        self.ranking_agent      = FundingMatchRankingAgent()
        self.verification_agent = VerificationAgent()
        self.proposal_agent     = ProposalAssistantAgent()

    def find_funding(self, profile: dict) -> dict:
        """
        Run the full pipeline for a startup profile.

        Returns
        -------
        {
            "opportunities": [...],
            "meta": {...},
            "token_usage": {...},
            "error": null | "string"
        }
        """
        startup_name = profile.get("name") or "unnamed startup"
        logger.info("=== Pipeline start: %s ===", startup_name)

        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        error_msg = None

        try:
            # Stage 1: Discovery
            opportunities, usage = self.discovery_agent.run(profile)
            _accumulate_usage(total_usage, usage)

            if not opportunities:
                return {
                    "opportunities": [],
                    "meta": {
                        "profile_name": startup_name,
                        "total_found": 0,
                        "pipeline": "discovery → (no results)",
                        "status": "No matching opportunities found in knowledge base for this profile.",
                    },
                    "token_usage": total_usage,
                    "error": None,
                }

            # Stage 2: Eligibility analysis
            opportunities, usage = self.eligibility_agent.run(profile, opportunities)
            _accumulate_usage(total_usage, usage)

            # Stage 3: Ranking
            opportunities, usage = self.ranking_agent.run(profile, opportunities)
            _accumulate_usage(total_usage, usage)

            # Stage 4: Verification (no LLM call — source grounding only)
            opportunities = self.verification_agent.run(opportunities)

            # Prepare clean output (remove internal KB fields not needed by frontend)
            clean_opps = [_clean_opportunity(o) for o in opportunities]

            logger.info("=== Pipeline complete: %d opportunities ===", len(clean_opps))

            return {
                "opportunities": clean_opps,
                "meta": {
                    "profile_name": startup_name,
                    "total_found":  len(clean_opps),
                    "pipeline":     "discovery → eligibility → ranking → verification",
                    "status":       "complete",
                },
                "token_usage": total_usage,
                "error": None,
            }

        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
            return {
                "opportunities": [],
                "meta": {"profile_name": startup_name, "status": "error"},
                "token_usage": total_usage,
                "error": f"Pipeline error: {type(e).__name__}: {str(e)[:200]}",
            }

    def generate_proposal(self, profile: dict, opportunity: dict) -> dict:
        """Generate a draft proposal. Returns {draft, token_usage, error}."""
        try:
            draft, usage = self.proposal_agent.run(profile, opportunity)
            return {"draft": draft, "token_usage": usage, "error": None}
        except Exception as e:
            logger.error("Proposal generation error: %s", e)
            return {
                "draft": f"⚠️ Proposal generation failed: {type(e).__name__}",
                "token_usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
                "error": str(e)[:200],
            }


# ═══════════════════════════════════════════════════════════════
# 13. RESPONSE PARSERS
# ═══════════════════════════════════════════════════════════════

def _parse_numbered_list(text: str, expected: int) -> list:
    """Parse a numbered list like '1. text\\n2. text' into a Python list."""
    lines = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[\.\)]\s*(.+)", line)
        if m:
            lines.append(m.group(1).strip())
    # Pad if fewer than expected
    while len(lines) < expected:
        lines.append("")
    return lines


def _parse_eligibility_response(text: str, expected: int) -> list:
    """Parse structured eligibility response into list of dicts."""
    results = []
    blocks = re.split(r"\[\d+\]", text)
    blocks = [b.strip() for b in blocks if b.strip()]

    for block in blocks:
        verdict = ""
        reason  = ""
        gap     = ""

        v_match = re.search(r"VERDICT:\s*(.+)", block, re.IGNORECASE)
        r_match = re.search(r"REASON:\s*(.+)", block, re.IGNORECASE)
        g_match = re.search(r"GAP:\s*(.+)", block, re.IGNORECASE)

        if v_match:
            raw_verdict = v_match.group(1).strip()
            if "not eligible" in raw_verdict.lower():
                verdict = "Not Eligible"
            elif "possibly" in raw_verdict.lower():
                verdict = "Possibly Eligible"
            else:
                verdict = "Eligible"
        if r_match:
            reason = r_match.group(1).strip()
        if g_match:
            gap = g_match.group(1).strip()
            if gap.lower() in ("none", "n/a", "-"):
                gap = ""

        results.append({"verdict": verdict or "Possibly Eligible", "reason": reason, "gap": gap})

    while len(results) < expected:
        results.append({"verdict": "Possibly Eligible", "reason": "", "gap": ""})

    return results


def _parse_score_response(text: str, expected: int) -> list:
    """Parse score response '[N] SCORE: 85 | REASON: ...' into list of dicts."""
    results = []
    for line in text.splitlines():
        m = re.match(r"\[\d+\]\s*SCORE:\s*(\d+)\s*\|?\s*REASON:\s*(.+)", line, re.IGNORECASE)
        if m:
            score = min(100, max(0, int(m.group(1))))
            reason = m.group(2).strip()
            results.append({"score": score, "reason": reason})

    while len(results) < expected:
        results.append({"score": 50, "reason": ""})

    return results


def _accumulate_usage(total: dict, source: dict):
    """Add token counts from source into total, handling None values."""
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        v = source.get(key)
        if v is not None:
            total[key] = (total.get(key) or 0) + v


def _clean_opportunity(opp: dict) -> dict:
    """Return a clean dict safe to send to the frontend."""
    return {
        "id":                   opp.get("id", ""),
        "name":                 opp.get("name", ""),
        "organization":         opp.get("organization", ""),
        "type":                 opp.get("type", ""),
        "amount_display":       opp.get("amount_display", "Not specified"),
        "eligible_stages":      opp.get("eligible_stages", []),
        "eligible_domains":     opp.get("eligible_domains", []),
        "eligible_locations":   opp.get("eligible_locations", []),
        "description":          opp.get("description", ""),
        "eligibility_summary":  opp.get("eligibility_summary", ""),
        "deadline_notes":       opp.get("deadline_notes", ""),
        "application_url":      opp.get("application_url", ""),
        "source_url":           opp.get("source_url", ""),
        "last_verified":        opp.get("last_verified", ""),
        "tags":                 opp.get("tags", []),
        # AI-generated fields
        "ai_note":              opp.get("ai_note", ""),
        "eligibility":          opp.get("eligibility", "possibly-eligible"),
        "eligibility_label":    opp.get("eligibility_label", "Possibly Eligible"),
        "eligibility_reason":   opp.get("eligibility_reason", ""),
        "eligibility_gap":      opp.get("eligibility_gap", ""),
        "match_score":          opp.get("match_score", 50),
        "match_reason":         opp.get("match_reason", ""),
        "source_grounded":      opp.get("source_grounded", False),
        "verification_warning": opp.get("verification_warning", ""),
    }


# ═══════════════════════════════════════════════════════════════
# 14. FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════

def create_app() -> Flask:
    app = Flask(__name__, static_folder=".", template_folder=".")
    CORS(app)

    orchestrator = FundFlowOrchestrator()

    # Preload knowledge base at startup
    load_funding_kb()

    # ── Serve frontend ─────────────────────────────────────────
    @app.route("/")
    def index():
        return send_from_directory(".", "Index.html")

    @app.route("/Style.css")
    def stylesheet():
        return send_from_directory(".", "Style.css")

    # ── Health & status ────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status":  "ok",
            "service": "FundFlow AI",
            "kb_records": len(load_funding_kb()),
            "granite": {
                "api_key_set":    bool(_get_env("IBM_API_KEY", warn=False)),
                "url_set":        bool(_get_env("IBM_GRANITE_URL", warn=False)),
                "model_id_set":   bool(_get_env("IBM_GRANITE_MODEL_ID", warn=False)),
                "project_id_set": bool(_get_env("IBM_PROJECT_ID", warn=False)),
                "space_id_set":   bool(_get_env("IBM_SPACE_ID", warn=False)),
            },
        })

    @app.route("/api/test-granite", methods=["GET"])
    def test_granite():
        """Test IBM Granite connectivity — never exposes credentials."""
        result = test_granite_connection()
        return jsonify(result)

    # ── Funding discovery ──────────────────────────────────────
    @app.route("/api/find-funding", methods=["POST"])
    def find_funding():
        profile = request.get_json(silent=True) or {}
        if not profile:
            return jsonify({"error": "Request body must be a JSON startup profile."}), 400

        # Basic validation
        if not any(profile.get(k) for k in ("name", "domain", "stage", "description")):
            return jsonify({
                "error": "Please provide at least a startup name, domain, stage, or description."
            }), 400

        result = orchestrator.find_funding(profile)
        return jsonify(result)

    # ── Proposal generation ────────────────────────────────────
    @app.route("/api/generate-proposal", methods=["POST"])
    def generate_proposal():
        body = request.get_json(silent=True) or {}
        profile     = body.get("startup_profile", {})
        opportunity = body.get("opportunity", {})

        if not opportunity.get("name"):
            return jsonify({"error": "No opportunity selected."}), 400

        result = orchestrator.generate_proposal(profile, opportunity)
        return jsonify(result)

    # ── Error handlers ─────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(500)
    def server_error(_):
        return jsonify({"error": "Internal server error. Check application logs."}), 500

    return app


# ═══════════════════════════════════════════════════════════════
# 15. ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    host  = _get_env("FLASK_HOST", warn=False) or "127.0.0.1"
    port  = int(_get_env("FLASK_PORT", warn=False) or 5000)
    debug = (_get_env("FLASK_ENV", warn=False) or "development").lower() == "development"

    logger.info("Starting FundFlow AI on http://%s:%s (debug=%s)", host, port, debug)
    app = create_app()
    app.run(host=host, port=port, debug=debug)
