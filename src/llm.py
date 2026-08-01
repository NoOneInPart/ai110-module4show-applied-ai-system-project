"""
The AI layer: a constrained RAG pipeline over the deterministic recommender.

Flow (see DESIGN.md):
  1. extract_profile()        -- LLM call 1: free text -> UserProfile (+ a
                                 friendly intro), or a friendly error message.
  2. recommend_songs(k=WIDTH) -- the deterministic scorer acts as the RETRIEVER,
                                 narrowing the catalog to a wide candidate set.
  3. select_recommendations() -- LLM call 2: pick the final FINAL_K FROM those
                                 candidates, grounded in their real scores.

The deterministic scorer is always the ground truth: the LLM chooses among real
scores and never invents them, and every step degrades to a deterministic result
if the model is unavailable or returns something unusable. This module knows
nothing about how to reach Gemini -- it talks to any object exposing
`complete(system_prompt, user_prompt, json_output=bool) -> str` (see
llm_client.GeminiClient), which makes it testable with a fake client.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from .recommender import UserProfile, recommend_songs

# --- config knobs (top of file) ---
MODEL = "gemini-flash-lite-latest"  # flash-lite tier: cheap + fast
CANDIDATE_WIDTH = 15                 # how many the retriever hands the LLM
FINAL_K = 5                          # how many the LLM returns
TEMPERATURE = 0.3                    # low: selection should be stable
API_KEY_ENV = "GEMINI_API_KEY"

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt(name: str) -> str:
    """Read src/prompts/<name>.txt."""
    with open(os.path.join(_PROMPTS_DIR, f"{name}.txt"), encoding="utf-8") as f:
        return f.read()


@dataclass
class ProfileResult:
    """Outcome of the profile-extraction call: a profile, an error, or neither.

    - profile set            -> use it (intro is the friendly lead-in, if any)
    - error set              -> Gemini's message for a no-signal prompt; the
                                caller pipes it to the user, then falls back
    - all None (empty)       -> silent failure; the caller falls back quietly
    """
    profile: Optional[UserProfile] = None
    intro: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RecommendationRun:
    """What the orchestrator hands back to the UI.

    - picks non-empty        -> render them (intro printed first, if any)
    - error set              -> pipe it to the user, then fall back to manual
    - picks empty, no error  -> silent extraction failure; fall back to manual
    """
    picks: List[Tuple[Dict, float, str]] = field(default_factory=list)
    intro: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------
def _parse_json(raw: str) -> Any:
    """Parse the model's text as JSON. JSON mode makes this a plain json.loads,
    but we tolerate stray code fences / prose just in case."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    # Tolerant fallback: pull the first {...} or [...] block out of the text.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (ValueError, TypeError):
                continue
    return None


def _clamp01(value: Any) -> Optional[float]:
    """Coerce to a float in [0, 1], or None if it isn't a number."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def validate_profile(raw: Dict) -> UserProfile:
    """Turn a loose dict (from the LLM or from manual entry) into a validated
    UserProfile. Ranges are clamped, types coerced, junk dropped to None. Shared
    by both profile paths so there is a single guardrail to test."""
    def as_text(key: str) -> Optional[str]:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return None

    def as_bool(key: str) -> Optional[bool]:
        value = raw.get(key)
        return value if isinstance(value, bool) else None

    try:
        tempo: Optional[float] = max(20.0, min(300.0, float(raw.get("target_tempo_bpm"))))
    except (TypeError, ValueError):
        tempo = None

    return UserProfile(
        favorite_genre=as_text("favorite_genre"),
        favorite_mood=as_text("favorite_mood"),
        target_energy=_clamp01(raw.get("target_energy")),
        likes_acoustic=as_bool("likes_acoustic"),
        target_tempo_bpm=tempo,
        target_valence=_clamp01(raw.get("target_valence")),
        target_danceability=_clamp01(raw.get("target_danceability")),
    )


def _profile_is_empty(profile: UserProfile) -> bool:
    """True when no preference was supplied at all."""
    return all(value is None for value in asdict(profile).values())


# --------------------------------------------------------------------------
# LLM call 1: profile extraction
# --------------------------------------------------------------------------
def extract_profile(text: str, client: Any) -> ProfileResult:
    """Turn free text into a validated UserProfile (+ friendly intro), or an
    error message, or a silent failure. Never raises."""
    system = _load_prompt("profile_system")
    user = _load_prompt("profile_user").replace("{{REQUEST}}", text)

    data = _parse_json(client.complete(system, user, json_output=True))
    if not isinstance(data, dict):
        return ProfileResult()  # unparseable / model failure -> quiet fallback

    profile = validate_profile(data)
    if not _profile_is_empty(profile):
        note = data.get("note")
        intro = note.strip() if isinstance(note, str) and note.strip() else None
        return ProfileResult(profile=profile, intro=intro)

    # No usable preferences: honor an explicit error message, else quiet fallback.
    error = data.get("error")
    if isinstance(error, str) and error.strip():
        return ProfileResult(error=error.strip())
    return ProfileResult()


# --------------------------------------------------------------------------
# LLM call 2: grounded selection
# --------------------------------------------------------------------------
def _candidate_for_prompt(candidate: Tuple[Dict, float, str]) -> Dict:
    """Compact view of one candidate for the model: features + real score +
    the deterministic reasons that fired."""
    song, score, reasons = candidate
    return {
        "id": song["id"], "title": song["title"], "artist": song["artist"],
        "genre": song["genre"], "mood": song["mood"],
        "energy": song["energy"], "tempo_bpm": song["tempo_bpm"],
        "valence": song["valence"], "danceability": song["danceability"],
        "acousticness": song["acousticness"],
        "score": round(float(score), 2),
        "match_reasons": reasons.split("\n") if reasons else [],
    }


def _picks_from(data: Any) -> List[Dict]:
    if isinstance(data, dict) and isinstance(data.get("picks"), list):
        return data["picks"]
    if isinstance(data, list):
        return data
    return []


def select_recommendations(
    profile: UserProfile,
    candidates: List[Tuple[Dict, float, str]],
    client: Any,
) -> List[Tuple[Dict, float, str]]:
    """LLM picks the final FINAL_K from the retrieved candidates.

    Guardrails: picks are constrained to the candidate ids (invented ids
    dropped); the score is always the deterministic one; a pick's reasons are
    trusted only when `known` is true (the model recognizes the real song),
    otherwise that song's deterministic reasons are used. Falls back to the
    deterministic top-k on any failure, and backfills to FINAL_K if the model
    returns fewer.
    """
    by_id = {song["id"]: (song, score, reasons) for song, score, reasons in candidates}
    fallback = candidates[:FINAL_K]

    system = _load_prompt("select_system").replace("{{FINAL_K}}", str(FINAL_K))
    prof_json = json.dumps(
        {k: v for k, v in asdict(profile).items() if v is not None}, ensure_ascii=False
    )
    cand_json = json.dumps(
        [_candidate_for_prompt(c) for c in candidates], ensure_ascii=False
    )
    user = (
        _load_prompt("select_user")
        .replace("{{PROFILE}}", prof_json)
        .replace("{{CANDIDATES}}", cand_json)
        .replace("{{FINAL_K}}", str(FINAL_K))
    )

    picks = _picks_from(_parse_json(client.complete(system, user, json_output=True)))
    if not picks:
        return fallback

    result: List[Tuple[Dict, float, str]] = []
    seen = set()
    for pick in picks:
        if not isinstance(pick, dict):
            continue
        cid = pick.get("id")
        if cid not in by_id or cid in seen:
            continue  # guardrail: no invented / duplicate ids
        seen.add(cid)
        song, score, det_reasons = by_id[cid]
        reasons = pick.get("reasons")
        good_reasons = [
            r.strip() for r in reasons if isinstance(r, str) and r.strip()
        ] if isinstance(reasons, list) else []
        # Trust the model's prose only when it says it knows the real song.
        if pick.get("known") is True and good_reasons:
            explanation = "\n".join(good_reasons)
        else:
            explanation = det_reasons
        result.append((song, score, explanation))
        if len(result) >= FINAL_K:
            break

    if not result:
        return fallback

    # Backfill from the deterministic order if the model returned too few.
    if len(result) < FINAL_K:
        for candidate in candidates:
            if candidate[0]["id"] not in seen:
                result.append(candidate)
                seen.add(candidate[0]["id"])
                if len(result) >= FINAL_K:
                    break
    return result


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def recommend_from_text(text: str, songs: List[Dict], client: Any) -> RecommendationRun:
    """Full pipeline: extract -> retrieve (wide) -> select. Any failure degrades
    to a deterministic result; a no-signal prompt returns the model's error."""
    result = extract_profile(text, client)
    if result.error:
        return RecommendationRun(error=result.error)
    if result.profile is None:
        return RecommendationRun()  # quiet failure -> caller falls back to manual

    prefs = asdict(result.profile)
    candidates = recommend_songs(prefs, songs, k=CANDIDATE_WIDTH)
    picks = select_recommendations(result.profile, candidates, client)
    return RecommendationRun(picks=picks, intro=result.intro)
