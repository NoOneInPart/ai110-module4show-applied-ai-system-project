"""
Command line runner for the Music Recommender Simulation.

Usage:
    python -m src.main         natural-language mode (the AI feature): describe
                               what you want and Gemini builds a taste profile
                               and picks songs. With GEMINI_API_KEY set it calls
                               Gemini; without a key it falls back to a short
                               manual questionnaire (Enter accepts defaults).
    python -m src.main demo    the deterministic sample profiles below
                               (reproducible, no key needed).

Any LLM failure, a no-signal prompt, or missing input degrades to manual entry /
deterministic results, so the app never breaks.
"""

import sys
from dataclasses import asdict

from .recommender import load_songs, recommend_songs, UserProfile
from .llm import (
    recommend_from_text,
    validate_profile,
    MODEL,
    TEMPERATURE,
    API_KEY_ENV,
)

CATALOG = "data/song2.csv"


# A handful of profiles to try in the non-interactive demo. Keys match the
# UserProfile fields; any key left out (or set to None) is simply skipped by the
# scorer, which is what the edge-case profiles below exercise.
PROFILES = {
    "high-energy pop": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.95,
        "likes_acoustic": False,
    },
    "chill lofi": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.4,
        "likes_acoustic": True,
    },
    "deep intense rock": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.95,
        "likes_acoustic": False,
    },
    # Edge case: acoustic + high energy pull in opposite directions.
    "edge: acoustic headbanger": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.95,
        "likes_acoustic": True,
    },
    # Edge case: genre and mood that rarely co-occur on one song.
    "edge: happy classical": {
        "favorite_genre": "classical",
        "favorite_mood": "happy",
        "target_energy": 0.5,
        "likes_acoustic": True,
    },
}


def print_recommendations(name: str, user_prefs: dict, songs: list) -> None:
    recommendations = recommend_songs(user_prefs, songs, k=5)

    print()
    print("=" * 60)
    print(f"  TOP RECOMMENDATIONS - {name}")
    print("=" * 60)
    print(
        f"  Profile: genre={user_prefs.get('favorite_genre')}, "
        f"mood={user_prefs.get('favorite_mood')}, "
        f"energy={user_prefs.get('target_energy')}, "
        f"likes_acoustic={user_prefs.get('likes_acoustic')}"
    )
    print("-" * 60)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  {rank}. {song['artist']} - {song['title']}  (score: {score:.2f})")
        print(f"       [{song['genre']} / {song['mood']}]")
        print("       why:")
        for reason in explanation.split("\n"):
            print(f"         - {reason}")
        print()


def _print_llm_header(profile_intro) -> None:
    """Printed the moment the profile call returns, before selection runs, so the
    user sees the reflection immediately instead of waiting for the second call."""
    print()
    print("=" * 60)
    print("  YOUR RECOMMENDATIONS")
    print("=" * 60)
    if profile_intro:
        print(f"  {profile_intro}")
    sys.stdout.flush()


def _print_llm_picks(selection_intro, picks) -> None:
    """Printed once selection returns: the picks-grounded intro, then the songs."""
    print("-" * 60)
    if selection_intro:
        print(f"  {selection_intro}")
    print()
    for rank, (song, score, explanation) in enumerate(picks, start=1):
        print(f"  {rank}. {song['artist']} - {song['title']}  (score: {score:.2f})")
        print(f"       [{song['genre']} / {song['mood']}]")
        print("       why:")
        for reason in explanation.split("\n"):
            print(f"         - {reason}")
        print()


def _read_line(prompt: str, default: str = "") -> str:
    """Read one line; return `default` on empty input or no stdin (EOF), so the
    app stays reproducible when run non-interactively."""
    try:
        value = input(prompt).strip()
    except EOFError:
        return default
    return value if value else default


def manual_entry() -> UserProfile:
    """Ask for each preference, offering a [default] you can accept with Enter.
    Everything funnels through validate_profile, the same guardrail the LLM path
    uses. Pressing Enter through all of it yields a valid default profile."""
    print("\nDescribe your taste (press Enter to accept the [default]):")
    raw = {
        "favorite_genre": _read_line("  favorite genre [none]: ", ""),
        "favorite_mood": _read_line("  favorite mood [none]: ", ""),
        "target_energy": _read_line("  target energy 0-1 [0.5]: ", "0.5"),
        "target_valence": _read_line("  target valence 0-1 [none]: ", ""),
        "target_danceability": _read_line("  target danceability 0-1 [none]: ", ""),
        "target_tempo_bpm": _read_line("  target tempo bpm [none]: ", ""),
    }
    acoustic = _read_line("  like acoustic tracks? y/n [none]: ", "").lower()
    if acoustic in ("y", "yes", "true"):
        raw["likes_acoustic"] = True
    elif acoustic in ("n", "no", "false"):
        raw["likes_acoustic"] = False
    else:
        raw["likes_acoustic"] = None
    return validate_profile(raw)


def _make_client():
    """Build a GeminiClient if a key is available, else None. Importing here
    keeps google-genai out of the deterministic / no-key path entirely."""
    try:
        from .llm_client import GeminiClient
        return GeminiClient(model_name=MODEL, temperature=TEMPERATURE)
    except Exception:
        return None


def _manual_flow(songs: list) -> None:
    profile = manual_entry()
    print_recommendations("your profile", asdict(profile), songs)


def run_interactive(songs: list) -> None:
    client = _make_client()
    if client is None:
        print(
            f"\n(No {API_KEY_ENV} found - using manual entry. Set the key to "
            "describe what you want in plain language.)"
        )
        _manual_flow(songs)
        return

    text = _read_line("\nWhat would you like to hear? ", "")
    if text:
        # The header + profile intro print via the callback the instant the
        # first call returns; the picks print here once selection comes back.
        run = recommend_from_text(text, songs, client, on_profile=_print_llm_header)
        if run.error:
            print("\n" + run.error)              # pipe Gemini's message through
        elif run.picks:
            _print_llm_picks(run.intro, run.picks)
            return
        else:
            print("\n(Couldn't read a taste from that - let's do it manually.)")
    # No input, an error message, or a silent failure -> manual entry.
    _manual_flow(songs)


def run_demo(songs: list) -> None:
    for name, user_prefs in PROFILES.items():
        print_recommendations(name, user_prefs, songs)


def main() -> None:
    # Song titles (e.g. 残酷な天使のテーゼ) and Gemini's replies can contain
    # non-ASCII characters; force UTF-8 so a Windows console can't crash on them.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    songs = load_songs(CATALOG)
    # Interactive natural-language mode (the AI feature) is the default;
    # `demo` runs the reproducible deterministic sample profiles.
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if mode in ("demo", "--demo"):
        run_demo(songs)
    else:
        run_interactive(songs)


if __name__ == "__main__":
    main()
