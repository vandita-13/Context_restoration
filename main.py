import os
import json
import re
import asyncio
import requests
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import google.generativeai as genai

# 1. SETUP & CONFIGURATION
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "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")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# 2. CORS SETTINGS (Critical for React to talk to FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your React app (localhost:5173 or 3000) to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Model
model = genai.GenerativeModel("gemini-2.5-flash") if GEMINI_API_KEY else None


class AnalyzeRequest(BaseModel):
    claim: str


def normalize_claim_text(claim: str) -> str:
    cleaned = claim.replace("\u00a0", " ")
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:1200]


def parse_json_from_model_text(raw_text: str):
    if not raw_text:
        return None

    text = raw_text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def is_rate_limit_error(error_text: str) -> bool:
    lowered = (error_text or "").lower()
    return "429" in lowered or "quota" in lowered or "rate limit" in lowered


def extract_retry_seconds(error_text: str, default_seconds: float = 8.0) -> float:
    if not error_text:
        return default_seconds

    # Example formats: "Please retry in 38.046051264s" or "retry_delay { seconds: 38 }"
    retry_match = re.search(r"retry\s+in\s+([\d.]+)s", error_text, flags=re.IGNORECASE)
    if retry_match:
        try:
            return max(1.0, float(retry_match.group(1)))
        except Exception:
            pass

    seconds_match = re.search(r"retry_delay\s*\{[^}]*seconds:\s*(\d+)", error_text, flags=re.IGNORECASE | re.DOTALL)
    if seconds_match:
        try:
            return max(1.0, float(seconds_match.group(1)))
        except Exception:
            pass

    return default_seconds


def clean_text_for_ui(text: str, max_len: int = 260) -> str:
    if not text:
        return ""

    cleaned = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace("â€¢", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if "â" in cleaned:
        cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned[:max_len]


def extract_best_evidence(context: str) -> str:
    if not context:
        return "No direct live evidence available."

    content_matches = re.findall(r"Content:\s*(.*)", context)
    for raw in content_matches:
        candidate = clean_text_for_ui(raw, max_len=260)
        if candidate and candidate.lower() != "n/a":
            return candidate

    return "No direct live evidence available."


def build_heuristic_analysis(claim: str, context: str, fallback_reason: str = ""):
    lower_claim = claim.lower()
    lower_context = context.lower()

    high_risk_terms = ["always", "never", "secret", "miracle", "guaranteed", "shocking", "viral"]
    risky_hits = sum(1 for term in high_risk_terms if term in lower_claim)

    truth = max(18, 58 - risky_hits * 8)
    bias = min(86, 24 + risky_hits * 11)

    if "no live web context available" in lower_context:
        truth = max(20, truth - 10)
        bias = min(90, bias + 8)

    sentiment = "Negative" if risky_hits > 1 else "Neutral"

    evidence = extract_best_evidence(context)

    summary = "Automated fallback analysis was used because the AI model was temporarily unavailable."
    if is_rate_limit_error(fallback_reason):
        retry_hint = extract_retry_seconds(fallback_reason, default_seconds=30.0)
        summary = f"Gemini is rate-limited right now. Showing a provisional result; retry in about {int(round(retry_hint))} seconds."

    detailed_lines = [
        "• The claim text was normalized before analysis.",
        f"• Detected {risky_hits} high-risk language indicator(s) in the text.",
        "• This is a provisional fallback result based on available web context and heuristics.",
        "• Retry shortly for a full model-backed analysis.",
    ]

    return {
        "truth": int(truth),
        "bias": int(bias),
        "sentiment": sentiment,
        "summary": clean_text_for_ui(summary, max_len=320),
        "evidence": evidence,
        "detailed": "\n".join(detailed_lines),
    }

def get_web_context(query: str):
    """
    RESEARCH PHASE: This function 'Googles' the claim using Tavily 
    to get real-time context before the AI evaluates it.
    """
    try:
        if not TAVILY_KEY:
            return "No live web context available (missing TAVILY_API_KEY)."

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": 3
        }
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        results = response.json().get('results', [])

        if not results:
            return "No live web context available."

        # Combine snippets from top 3 search results
        return "\n".join([f"Source: {r.get('url', 'N/A')}\nContent: {r.get('content', 'N/A')}" for r in results])
    except Exception as e:
        print(f"Search Error: {e}")
        return "No live web context available."

def build_error_response(message: str, details: str = "N/A"):
    return {
        "truth": 0,
        "bias": 0,
        "sentiment": "Error",
        "summary": message,
        "evidence": "N/A",
        "detailed": details,
    }


async def analyze_claim_internal(claim: str):
    """
    ANALYSIS PHASE: Fetches live data and uses Gemini to 
    return a structured JSON report.
    """
    try:
        claim = normalize_claim_text(claim)
        if not claim:
            return build_error_response("Please enter a valid claim.", "Claim text is empty.")

        # Step A: Get Live Data
        context = get_web_context(claim)

        if not GEMINI_API_KEY or model is None:
            return build_heuristic_analysis(claim, context, "Missing GEMINI_API_KEY")

        # Step B: Create a Structured Prompt
        prompt = f"""
        ACT AS A PROFESSIONAL FACT-CHECKER.
        
        USER CLAIM: "{claim}"
        LIVE WEB CONTEXT: 
        {context}

        INSTRUCTIONS:
        1. Evaluate the claim against the provided context.
        2. Assign a Truth Score (0-100).
        3. Assign a Media Bias Score (0-100, where 100 is highly biased/partisan).
        4. Detect Sentiment (Positive, Negative, or Neutral).
        5. Provide a 2-sentence Executive Summary.
        6. Extract a DIRECT QUOTE from the context as 'Evidence'.
        7. Provide 3 bullet points of 'Detailed' analysis.

        RESPONSE FORMAT:
        You MUST return ONLY a valid JSON object. Do not include markdown or backticks.
        JSON Structure:
        {{
            "truth": 85,
            "bias": 20,
            "sentiment": "Neutral",
            "summary": "Example summary here.",
            "evidence": "Example quote here.",
            "detailed": "• Point 1\\n• Point 2\\n• Point 3"
        }}
        """

        # Step C: Generate and Parse Response
        max_attempts = 2
        response = None
        for attempt in range(max_attempts):
            try:
                response = model.generate_content(prompt)
                break
            except Exception as gen_err:
                err_text = str(gen_err)
                if is_rate_limit_error(err_text) and attempt < (max_attempts - 1):
                    wait_for = min(extract_retry_seconds(err_text), 45.0)
                    await asyncio.sleep(wait_for)
                    continue
                raise

        raw_text = (response.text or "").strip()
        data = parse_json_from_model_text(raw_text)

        if not data:
            return build_heuristic_analysis(
                claim,
                context,
                "Model returned invalid JSON format",
            )

        if not isinstance(data, dict):
            return build_heuristic_analysis(claim, context, "Model returned non-object response")

        data.setdefault("truth", 0)
        data.setdefault("bias", 0)
        data.setdefault("sentiment", "Neutral")
        data.setdefault("summary", "No summary provided.")
        data.setdefault("evidence", "No evidence provided.")
        data.setdefault("detailed", "No detailed analysis provided.")

        data["truth"] = max(0, min(100, int(float(data.get("truth", 0)))))
        data["bias"] = max(0, min(100, int(float(data.get("bias", 0)))))
        return data

    except Exception as e:
        print(f"Server Error: {e}")
        fallback_reason = str(e)
        if "API_KEY_INVALID" in fallback_reason:
            fallback_reason = "Gemini API key is invalid"
        elif is_rate_limit_error(fallback_reason):
            retry_hint = extract_retry_seconds(fallback_reason, default_seconds=30.0)
            fallback_reason = f"Gemini rate limit reached. Retry in about {int(round(retry_hint))} seconds."
        context = get_web_context(claim)
        return build_heuristic_analysis(claim, context, fallback_reason)


@app.get("/analyze")
async def analyze_claim_get(claim: str | None = None):
    if not claim:
        return build_error_response("Please enter a valid claim.", "Missing 'claim' query parameter.")
    return await analyze_claim_internal(claim)


@app.post("/analyze")
async def analyze_claim_post(payload: AnalyzeRequest | None = Body(default=None), claim: str | None = None):
    resolved_claim = payload.claim if payload and payload.claim else claim
    if not resolved_claim:
        return build_error_response(
            "Please enter a valid claim.",
            "Provide claim in JSON body {\"claim\": \"...\"} or as query parameter.",
        )
    return await analyze_claim_internal(resolved_claim)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)