"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

#FIX: make importing relative so that it actually works
from .recommender import load_songs, recommend_songs


# A handful of profiles to try. Keys match the UserProfile fields; any key
# left out (or set to None) is simply skipped by the scorer, which is what the
# edge-case profiles below exercise.
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
    # Edge case: acoustic + high energy pull in opposite directions. In this
    # catalog acoustic tracks are all low-energy, so the heavy energy term and
    # the acoustic bonus can never both fire strongly on the same song.
    "edge: acoustic headbanger": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.95,
        "likes_acoustic": True,
    },
    # Edge case: genre and mood that never co-occur on one song. No track is
    # both "classical" and "happy", so at most one categorical bonus can fire.
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


def main() -> None:
    songs = load_songs("data/song2.csv")

    for name, user_prefs in PROFILES.items():
        print_recommendations(name, user_prefs, songs)


if __name__ == "__main__":
    main()
