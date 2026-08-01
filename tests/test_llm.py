"""
Offline tests for the LLM layer (src/llm.py). No API key or network needed: a
FakeClient stands in for GeminiClient and returns canned JSON, so these exercise
the guardrails and fallbacks, not the model itself.
"""

import json

from src.llm import (
    validate_profile,
    extract_profile,
    select_recommendations,
    recommend_from_text,
    FINAL_K,
)
from src.recommender import recommend_songs, score_song


def make_songs():
    base = dict(tempo_bpm=120.0, valence=0.5, danceability=0.5)
    return [
        {"id": 1, "title": "A", "artist": "X", "genre": "pop", "mood": "happy",
         "energy": 0.9, "acousticness": 0.1, **base},
        {"id": 2, "title": "B", "artist": "Y", "genre": "lofi", "mood": "chill",
         "energy": 0.3, "acousticness": 0.8, **base},
        {"id": 3, "title": "C", "artist": "Z", "genre": "rock", "mood": "intense",
         "energy": 0.8, "acousticness": 0.05, **base},
        {"id": 4, "title": "D", "artist": "W", "genre": "jazz", "mood": "relaxed",
         "energy": 0.4, "acousticness": 0.7, **base},
        {"id": 5, "title": "E", "artist": "V", "genre": "pop", "mood": "happy",
         "energy": 0.6, "acousticness": 0.2, **base},
    ]


class FakeClient:
    """Returns queued responses in order; "" once exhausted (the failure signal)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system_prompt, user_prompt, json_output=False):
        self.calls.append((system_prompt, user_prompt, json_output))
        return self.responses.pop(0) if self.responses else ""


# --------------------------------------------------------------------------
# validate_profile (shared guardrail)
# --------------------------------------------------------------------------
def test_validate_profile_clamps_and_coerces():
    p = validate_profile({
        "target_energy": 1.5,            # clamp -> 1.0
        "target_valence": -0.2,          # clamp -> 0.0
        "target_danceability": "0.4",    # string -> 0.4
        "target_tempo_bpm": 9999,        # clamp -> 300.0
        "likes_acoustic": True,
        "favorite_genre": "  POP ",      # trimmed + lowercased
        "favorite_mood": "",             # empty -> None
        "note": "ignored",               # non-profile keys ignored
    })
    assert p.target_energy == 1.0
    assert p.target_valence == 0.0
    assert p.target_danceability == 0.4
    assert p.target_tempo_bpm == 300.0
    assert p.likes_acoustic is True
    assert p.favorite_genre == "pop"
    assert p.favorite_mood is None


def test_validate_profile_drops_bad_values():
    p = validate_profile({"target_energy": "not a number", "likes_acoustic": "yes"})
    assert p.target_energy is None       # unparseable -> None
    assert p.likes_acoustic is None      # only real bools accepted


# --------------------------------------------------------------------------
# extract_profile (LLM call 1)
# --------------------------------------------------------------------------
def test_extract_profile_success_carries_intro():
    resp = json.dumps({"favorite_genre": "pop", "target_energy": 0.9,
                       "note": "Bangers incoming:"})
    result = extract_profile("upbeat pop", FakeClient(resp))
    assert result.error is None
    assert result.profile.favorite_genre == "pop"
    assert result.profile.target_energy == 0.9
    assert result.intro == "Bangers incoming:"


def test_extract_profile_error_message_is_surfaced():
    resp = json.dumps({"error": "Tell me a bit more."})
    result = extract_profile("asdfgh", FakeClient(resp))
    assert result.profile is None
    assert result.error == "Tell me a bit more."


def test_extract_profile_failure_is_quiet_fallback():
    # Empty output (model failure) -> all-None result (caller falls back).
    result = extract_profile("whatever", FakeClient(""))
    assert result.profile is None and result.error is None


# --------------------------------------------------------------------------
# select_recommendations (LLM call 2)
# --------------------------------------------------------------------------
def test_select_known_gate_and_id_constraint():
    songs = make_songs()
    profile = validate_profile({"target_energy": 0.9, "favorite_genre": "pop"})
    candidates = recommend_songs({"target_energy": 0.9, "favorite_genre": "pop"}, songs, k=15)

    resp = json.dumps({"picks": [
        {"id": 1, "known": True, "reasons": ["a great synth hook", "certified banger"]},
        {"id": 999, "known": True, "reasons": ["invented song"]},   # invalid id -> dropped
        {"id": 3, "known": False, "reasons": ["should be ignored"]},  # not known -> deterministic
    ]})
    picks = select_recommendations(profile, candidates, FakeClient(resp))
    ids = [song["id"] for song, _, _ in picks]

    assert 999 not in ids                                   # invented id dropped
    assert 1 in ids and 3 in ids
    cand_ids = {song["id"] for song, _, _ in candidates}
    assert set(ids) <= cand_ids                             # subset of retrieved candidates

    expl1 = next(e for s, _, e in picks if s["id"] == 1)
    assert "great synth hook" in expl1                      # known=True -> LLM prose used
    expl3 = next(e for s, _, e in picks if s["id"] == 3)
    assert "should be ignored" not in expl3                 # known=False -> deterministic
    assert "energy" in expl3

    assert len(picks) == min(FINAL_K, len(candidates))      # backfilled to FINAL_K


def test_select_scores_are_deterministic_not_invented():
    songs = make_songs()
    profile = validate_profile({"target_energy": 0.9})
    candidates = recommend_songs({"target_energy": 0.9}, songs, k=15)
    truth = {song["id"]: score for song, score, _ in candidates}

    resp = json.dumps({"picks": [{"id": 1, "known": True, "reasons": ["x"]}]})
    picks = select_recommendations(profile, candidates, FakeClient(resp))
    for song, score, _ in picks:
        assert score == truth[song["id"]]                   # score stays the deterministic one


def test_select_falls_back_to_deterministic_top_k():
    songs = make_songs()
    profile = validate_profile({"target_energy": 0.5})
    candidates = recommend_songs({"target_energy": 0.5}, songs, k=15)
    picks = select_recommendations(profile, candidates, FakeClient(""))  # model fails
    assert picks == candidates[:FINAL_K]


# --------------------------------------------------------------------------
# recommend_from_text (orchestrator)
# --------------------------------------------------------------------------
def test_recommend_from_text_success():
    songs = make_songs()
    profile_resp = json.dumps({"favorite_genre": "pop", "target_energy": 0.9,
                               "note": "Here you go:"})
    select_resp = json.dumps({"picks": [{"id": 1, "known": True, "reasons": ["hook"]}]})
    run = recommend_from_text("upbeat pop", songs, FakeClient(profile_resp, select_resp))

    assert run.error is None
    assert run.intro == "Here you go:"
    assert run.picks and run.picks[0][0]["id"] == 1
    catalog_ids = {s["id"] for s in songs}
    assert all(song["id"] in catalog_ids for song, _, _ in run.picks)


def test_recommend_from_text_surfaces_error():
    run = recommend_from_text("asdfgh", make_songs(),
                              FakeClient(json.dumps({"error": "Say more."})))
    assert run.error == "Say more."
    assert not run.picks


# --------------------------------------------------------------------------
# scorer properties (deterministic ground truth the LLM sits on top of)
# --------------------------------------------------------------------------
def test_scoring_is_deterministic():
    song = make_songs()[0]
    prefs = {"target_energy": 0.8, "favorite_genre": "pop"}
    assert score_song(prefs, song) == score_song(prefs, song)


def test_energy_term_is_monotonic():
    song = make_songs()[0]  # energy 0.9
    near = score_song({"target_energy": 0.9}, song)[0]
    far = score_song({"target_energy": 0.3}, song)[0]
    assert near >= far
