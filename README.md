# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
  > In this setup, each song contains metadata about the id, artist, genre, mood, energy, tempo, valence, danceability, and acousticness. The catalog (`data/song2.csv`, 88 songs) draws its audio features (energy, tempo, valence, danceability, acousticness) from the public-domain (CC0) [Spotify Tracks Dataset – Audio Features](https://www.kaggle.com/datasets/saichaitanyareddyai/spotify-tracks-dataset-audio-features) on Kaggle. `mood` is derived from each track's valence and energy, since the source dataset has no mood field.
- What information does your `UserProfile` store
  > The UserProfile stores a user's favorite genre, mood, their preferred energy level, and whether they like acoustic tracks (people don't?).
- How does your `Recommender` compute a score for each song
  > As of this commit, the score is built around a base score (called the backbone) and some bonuses for categorical matches. The backbone (weight 5.0) rewards how closely a song's energy matches the user's preferred energy level, so a song that "sounds like" what the user wants ranks highly even if its labels don't match exactly. On top of that, small bonuses nudge the ranking: +1.0 if the song's genre matches the user's favorite, +1.0 if the mood matches, and up to +0.5 depending on how strongly the song's acoustic-ness agrees with the user's `likes_acoustic` preference. The acoustic term is a pure bonus that scales with how acoustic a song is (never a penalty), so a fully-acoustic song earns the whole +0.5 for an acoustic-lover while an electric one earns little or nothing, and a user who simply hasn't asked for acoustic tracks isn't punished for songs that happen to be acoustic. This setup prevents the algorithm from overwhelmingly recommending songs that fit into specific categories based on the specific genre name or mood, but it does mean most of the songs recommended have similar energy levels.
- How do you choose which songs to recommend
  > Every song in the catalog is scored with the formula above, then the list is ranked from highest to lowest score and the top `k` songs (default 5) are returned. The scoring rule judges one song at a time; the ranking rule turns those scores into an ordered shortlist.

You can include a simple diagram or bullet list if helpful.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main        # interactive: describe what you want in plain language
python -m src.main demo   # deterministic sample profiles (reproducible, no key needed)
```

Natural-language mode uses Google AI Studio (Gemini) to turn your description
into a taste profile and pick songs. To enable it, set `GEMINI_API_KEY` in your
environment or a `.env` file (`GEMINI_API_KEY=...`). Without a key, `nl` mode
falls back to a short manual questionnaire (press Enter to accept defaults), so
the app always runs.

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
============================================================
  TOP RECOMMENDATIONS - high-energy pop
============================================================
  Profile: genre=pop, mood=happy, energy=0.95, likes_acoustic=False
------------------------------------------------------------
  1. David Guetta;Bebe Rexha - I'm Good (Blue)  (score: 6.42)
       [pop / intense]
       why:
         - energy 0.96 is close to your target 0.95
         - matches your favorite genre (pop)
         - acousticness 0.00 leans electric, matching your preference

  2. Rick Astley - Never Gonna Give You Up  (score: 6.39)
       [disco / happy]
       why:
         - energy 0.94 is close to your target 0.95
         - matches your mood (happy)
         - acousticness 0.12 leans electric, matching your preference

  3. Beast In Black - One Night in Tokyo  (score: 6.37)
       [heavy metal / happy]
       why:
         - energy 0.98 is close to your target 0.95
         - matches your mood (happy)
         - acousticness 0.00 leans electric, matching your preference

  4. Rammstein - Du hast  (score: 6.36)
       [industrial metal / happy]
       why:
         - energy 0.92 is close to your target 0.95
         - matches your mood (happy)
         - acousticness 0.00 leans electric, matching your preference

  5. IVE - After LIKE  (score: 6.31)
       [k-pop / happy]
       why:
         - energy 0.92 is close to your target 0.95
         - matches your mood (happy)
         - acousticness 0.10 leans electric, matching your preference


============================================================
  TOP RECOMMENDATIONS - chill lofi
============================================================
  Profile: genre=lofi, mood=chill, energy=0.4, likes_acoustic=True
------------------------------------------------------------
  1. Jinsang - Affection  (score: 6.42)
       [lofi / chill]
       why:
         - energy 0.19 is close to your target 0.40
         - matches your favorite genre (lofi)
         - matches your mood (chill)
         - acousticness 0.91 leans acoustic like you prefer

  2. Maroon 5 - Memories  (score: 6.06)
       [pop / chill]
       why:
         - energy 0.33 is close to your target 0.40
         - matches your mood (chill)
         - acousticness 0.84 leans acoustic like you prefer

  3. Idealism - Controlla  (score: 6.04)
       [lofi / groovy]
       why:
         - energy 0.45 is close to your target 0.40
         - matches your favorite genre (lofi)
         - acousticness 0.55 leans acoustic like you prefer

  4. potsu - i'm closing my eyes  (score: 5.81)
       [lofi / chill]
       why:
         - energy 0.11 is close to your target 0.40
         - matches your favorite genre (lofi)
         - matches your mood (chill)
         - acousticness 0.53 leans acoustic like you prefer

  5. Lewis Capaldi - Someone You Loved  (score: 5.35)
       [pop / groovy]
       why:
         - energy 0.41 is close to your target 0.40
         - acousticness 0.75 leans acoustic like you prefer
```

> Note: the high-energy example shows a quirk the catalog deliberately keeps — *Du hast* is scored "happy" (mood is derived from Spotify's valence, which is high for the track), and it sits comfortably beside disco and k-pop because energy dominates the score. The lofi example matches cleanly on genre now that real lofi tracks (potsu, Idealism, Jinsang) are in the catalog.

---

## Experiments You Tried

Use this section to document the experiments you ran.
> I picked the weighting shift experiment where the weighting of energy is doubled while the weighting of genre is halved. For user profiles like "chill lofi" and "acoustic headbanger" the results remained identical because song energy was the defining factor in these ranking choices, but "high-energy pop" experienced a large change as high energy songs were prioritized more heavily over songs within the pop genre.

---

## Limitations and Risks

Summarize some limitations of your recommender.

> This model prioritizes recommending based on energy, this is by design to reduce the effects of locking in on a particular genre or mood but this does mean all recommended songs are roughly the same energy level. Categories like genre are matched by exact-string, so a user that prefers "pop" would miss "indie pop" songs. Tempo, valence, and danceability are also not used in scoring as the current user profile does not contain preference info for those categories.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



