import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# --- scoring weights (the "algorithm recipe" knobs) ---
# Heavy continuous backbone; light categorical bonuses on top.
W_ENERGY = 5.0        # heavy: energy similarity does most of the work
W_GENRE = 1.0         # slight boost for an exact genre match
W_MOOD = 1.0          # slight boost for an exact mood match
W_ACOUSTIC = 0.5      # slight boost when acoustic-ness agrees with the user
# Optional continuous preferences (each fires only if the user supplies it).
# Kept lighter than W_ENERGY so energy stays the backbone, but heavy enough to
# add real specificity when the LLM (or a user) sets them.
W_VALENCE = 1.5       # moderate: valence/positivity closeness (0-1 scale)
W_DANCEABILITY = 1.5  # moderate: danceability closeness (0-1 scale)
W_TEMPO = 1.5         # moderate: tempo closeness (BPM, normalized by TEMPO_SCALE)
TEMPO_SCALE = 60.0    # BPM gap at which tempo closeness reaches 0

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.

    Every field is optional (defaults to None). score_song skips any preference
    that is None, so a sparse profile still scores on whatever is supplied. This
    lets the LLM emit only the tastes it can infer from free text and leave the
    rest unset, without breaking scoring.

    Required by tests/test_recommender.py
    """
    favorite_genre: Optional[str] = None
    favorite_mood: Optional[str] = None
    target_energy: Optional[float] = None
    likes_acoustic: Optional[bool] = None
    target_tempo_bpm: Optional[float] = None
    target_valence: Optional[float] = None
    target_danceability: Optional[float] = None

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # Reuse the shared scoring rule so OOP and functional paths agree.
        # asdict(user) yields the exact keys score_song expects.
        prefs = asdict(user)
        ranked = sorted(
            self.songs,
            key=lambda song: score_song(prefs, asdict(song))[0],
            reverse=True,
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = score_song(asdict(user), asdict(song))
        detail = ", ".join(reasons) if reasons else "it didn't strongly match your preferences"
        return f"'{song.title}' by {song.artist} scored {score:.2f}: {detail}"

# Which CSV columns to convert to which numeric type. Anything not listed
# stays a str (title, artist, genre, mood).
_INT_FIELDS = {"id"}
_FLOAT_FIELDS = {
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
}


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file into a list of dicts.

    Numeric columns are converted to the appropriate type: `id` becomes an
    int, the audio-feature columns (energy, tempo_bpm, valence,
    danceability, acousticness) become floats, and text columns (title,
    artist, genre, mood) stay as strings. Downstream scoring can therefore
    rely on numbers being real numbers, not strings.

    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song: Dict = {}
            for key, value in row.items():
                if key in _INT_FIELDS:
                    song[key] = int(value)
                elif key in _FLOAT_FIELDS:
                    song[key] = float(value)
                else:
                    song[key] = value
            songs.append(song)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    Scoring recipe (see README):
      - Heavy continuous backbone: closeness of the song's energy to the
        user's preferred energy (weight W_ENERGY).
      - Slight categorical bonuses: +W_GENRE / +W_MOOD for an exact
        genre / mood match.
      - Slight acoustic bonus (pure bonus, never a penalty): up to
        +W_ACOUSTIC, scaled by how strongly the song's acoustic-ness agrees
        with the user's preference (fully acoustic earns the whole bonus for
        an acoustic-lover; fully electric earns it for someone who isn't).

    Any preference the user hasn't supplied is simply skipped, so an
    unspecified taste never helps or hurts a song.

    Returns (score, reasons), where reasons explains which terms fired.
    """
    score = 0.0
    reasons: List[str] = []

    # HEAVY: energy closeness (both values are on a 0-1 scale)
    target_energy = user_prefs.get("target_energy")
    if target_energy is not None:
        song_energy = float(song["energy"])
        closeness = 1 - abs(song_energy - float(target_energy))
        score += W_ENERGY * closeness
        reasons.append(
            f"energy {song_energy:.2f} is close to your target {float(target_energy):.2f}"
        )

    # SLIGHT: exact genre match
    if user_prefs.get("favorite_genre") and song.get("genre") == user_prefs["favorite_genre"]:
        score += W_GENRE
        reasons.append(f"matches your favorite genre ({song['genre']})")

    # SLIGHT: exact mood match
    if user_prefs.get("favorite_mood") and song.get("mood") == user_prefs["favorite_mood"]:
        score += W_MOOD
        reasons.append(f"matches your mood ({song['mood']})")

    # SLIGHT: acoustic preference as a pure bonus (never subtracts), scaled by
    # how acoustic the song actually is. An acoustic-lover gets the full
    # W_ACOUSTIC for a fully-acoustic song and nothing for a fully-electric
    # one; the reverse holds for someone who doesn't like acoustic.
    likes_acoustic = user_prefs.get("likes_acoustic")
    if likes_acoustic is not None:
        acousticness = float(song["acousticness"])
        agreement = acousticness if likes_acoustic else 1 - acousticness
        score += W_ACOUSTIC * agreement
        reasons.append(
            f"acousticness {acousticness:.2f} " + (
                "leans acoustic like you prefer"
                if likes_acoustic
                else "leans electric, matching your preference"
            )
        )

    # OPTIONAL: valence closeness (0-1 scale, same recipe as energy)
    target_valence = user_prefs.get("target_valence")
    if target_valence is not None:
        song_valence = float(song["valence"])
        closeness = 1 - abs(song_valence - float(target_valence))
        score += W_VALENCE * closeness
        reasons.append(
            f"valence {song_valence:.2f} is close to your target {float(target_valence):.2f}"
        )

    # OPTIONAL: danceability closeness (0-1 scale, same recipe as energy)
    target_danceability = user_prefs.get("target_danceability")
    if target_danceability is not None:
        song_danceability = float(song["danceability"])
        closeness = 1 - abs(song_danceability - float(target_danceability))
        score += W_DANCEABILITY * closeness
        reasons.append(
            f"danceability {song_danceability:.2f} is close to your target {float(target_danceability):.2f}"
        )

    # OPTIONAL: tempo closeness. Unlike the others, tempo is raw BPM (not 0-1),
    # so the gap is normalized by TEMPO_SCALE and clamped to [0, 1] before
    # weighting — a gap of TEMPO_SCALE BPM or more contributes nothing.
    target_tempo = user_prefs.get("target_tempo_bpm")
    if target_tempo is not None:
        song_tempo = float(song["tempo_bpm"])
        closeness = 1 - min(abs(song_tempo - float(target_tempo)) / TEMPO_SCALE, 1.0)
        score += W_TEMPO * closeness
        reasons.append(
            f"tempo {song_tempo:.0f} bpm is close to your target {float(target_tempo):.0f} bpm"
        )

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Ranks the catalog for a user and returns the top-k recommendations.

    Scoring (per song) and ranking (over the list) are kept as two steps:
    `score_song` judges one song, and this function orders those scores.

    Returns a list of (song, score, explanation) tuples, sorted from
    highest score to lowest and truncated to k items.

    Required by src/main.py
    """
    # Score every song. A list comprehension is the Pythonic way to build a
    # new list from an existing iterable in one clear expression.
    scored = [
        (song, *score_song(user_prefs, song))  # -> (song, score, reasons)
        for song in songs
    ]

    # Rank: sorted() returns a NEW list (leaving `songs` untouched); the key
    # picks the score, and reverse=True gives highest-to-lowest order.
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)

    # Join reasons with newlines so callers can print one per line, then
    # keep only the top k.
    return [
        (song, score, "\n".join(reasons) if reasons else "no strong matches")
        for song, score, reasons in ranked[:k]
    ]
