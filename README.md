# 🎵 musicmachine 0.001a-rc2 (yes, that's what I'm calling it)

## Original Project (Modules 1-3)

This project began as **Music Recommender Simulation**, a small rule-based recommender built to explore how AI systems turn data into predictions. The original goal was to represent songs and a listener's taste as structured data, design a transparent scoring rule (not a black-box model) that ranks a catalog against that taste profile, and reflect on where bias or unfairness could creep into a system like this. It shipped as a CLI tool: a handful of hardcoded taste profiles were scored against a small song catalog using a deterministic formula, and the top matches were printed with a plain-language explanation of why each song scored the way it did.

---

## Module 4: What's New

This module adds two AI-powered elements. 

The *specialized model* element uses a prompt-tuned Gemini Flash Lite model to allow the user to textually describe the vibes of the songs they want to hear, and Gemini builds a user taste profile based on that prompt. 

The *Retrieval-Augmented Generation* element uses a scoring tool to pick the top 15 songs that match the generated user taste profile, and they are handed over to Gemini to pick the top 5 candidates that make sense for the prompt.

It makes for a good demonstration on how AI can be used to augment the functionality of an existing program.

---

## Architecture Overview

See [`diagrams/architecture.mmd`](diagrams/architecture.mmd) for the full system diagram. In short, this is a two-call constrained RAG pipeline with the deterministic scorer as the retriever:

```
your text
   |
   v
[LLM call 1: Profile Extraction]  ->  validated UserProfile + a short, generic
   (src/llm.py: extract_profile)      "vibe" intro (or a friendly error if the
                                       text carries no musical signal at all)
   |                                  ^ printed to you immediately
   v
[Retriever: the ORIGINAL deterministic scorer]
   (src/recommender.py: score_song / recommend_songs)
   scores the whole catalog, returns a WIDE candidate set (~15 songs)
   |
   v
[LLM call 2: Grounded Selection]  ->  final top-5, in ranked order, with an
   (src/llm.py: select_recommendations)  intro grounded in the ACTUAL picks
   |
   v
final recommendations: real score + real match facts (always shown),
                        plus personified description when the model
                        actually recognizes the song
```

---

## Setup Instructions

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```
   You may have to execute ``Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`` in PowerShell on Windows.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Get a Google AI Studio API key at
   https://aistudio.google.com/apikey, then copy `.env.example` to `.env` and
   fill it in:

   ```
   GEMINI_API_KEY=your_key_here
   ```

4. Run the app:

   ```bash
   python -m src.main        # interactive: describe what you want in plain language
   python -m src.main demo   # deterministic: sample profiles (reproducible, no key needed)
   ```

   Natural-language mode uses Gemini Flash Lite to turn your description
   into a taste profile and pick songs. Without a key (or if the call fails), it
   falls back to a short manual questionnaire — press Enter to accept the
   defaults shown in `[brackets]` — so the app always runs, key or no key.

### Running Tests

```bash
pytest
```

`tests/test_recommender.py` covers the original deterministic scorer.
`tests/test_llm.py` covers the AI layer entirely offline, using a fake LLM
client — no API key or network needed to run the suite.

---

## Sample Interactions

Two real, end-to-end runs against the live system (Gemini as the LLM, `data/song2.csv` as the catalog):

```
What would you like to hear? music to write README.md files to

============================================================
  YOUR RECOMMENDATIONS
============================================================
  Let's keep you locked in with some steady, low-key background grooves to power through the documentation.
------------------------------------------------------------
  Get into the zone and power through your documentation with this focused selection of mellow grooves and electronic textures.

  1. Lewis Capaldi - Someone You Loved  (score: 6.39)
       [pop / groovy]
       why:
         - A soaring, emotional vocal performance that anchors a steady, driving rhythm.
         - Provides a dramatic backdrop when you need to push through tedious documentation blocks.
         - energy 0.41 is close to your target 0.40
         - valence 0.45 is close to your target 0.50

  2. Idealism - Controlla  (score: 6.20)
       [lofi / groovy]
       why:
         - Dusty vinyl crackles and smooth hip-hop beats create an effortless, distraction-free bubble for coding.
         - The ultimate background loop for deep-focus writing sessions.
         - energy 0.45 is close to your target 0.40
         - valence 0.55 is close to your target 0.50

  3. Four Tet - Two Thousand and Seventeen  (score: 6.15)
       [electronic / groovy]
       why:
         - Intricate electronic textures and gentle, hypnotic pulses keep your brain engaged without breaking your train of thought.
         - Immaculate ambient pacing that makes drafting markdown sections feel like second nature.
         - energy 0.47 is close to your target 0.40
         - valence 0.50 is close to your target 0.50

  4. Arctic Monkeys - Hello You  (score: 6.01)
       [indie rock / groovy]
       why:
         - Laid-back indie rock instrumentation with a breezy, sophisticated melody.
         - Brings an upbeat yet composed energy that helps beat writer's block.
         - energy 0.45 is close to your target 0.40
         - valence 0.65 is close to your target 0.50

  5. Maroon 5 - Memories  (score: 5.99)
       [pop / chill]
       why:
         - A gentle acoustic guitar loop paired with a soft, comforting vocal delivery.
         - Keeps the mood light and steady as you polish the finishing touches on your project.
         - energy 0.33 is close to your target 0.40
         - valence 0.59 is close to your target 0.50
```

```
What would you like to hear? provocative songs for staying awake

============================================================
  YOUR RECOMMENDATIONS
============================================================
  Let's keep things sharp, bold, and high-energy to help you power through.
------------------------------------------------------------
  Keep the energy surging and ward off sleep with this high-octane mix of driving beats and infectious hooks.

  1. OneRepublic - I Ain't Worried  (score: 6.48)
       [rock / happy]
       why:
         - That whistling hook and sun-drenched whistle-along chorus are impossible to sleep through.
         - A breezy, driving rhythm that keeps your eyelids wide open.
         - energy 0.80 is close to your target 0.80
         - danceability 0.70 is close to your target 0.70

  2. Blur - Song 2 - 2012 Remaster  (score: 6.41)
       [britpop / happy]
       why:
         - An absolute jolt of adrenaline with that explosive, shout-along chorus.
         - Pure chaotic britpop energy that instantly shakes off any lingering fatigue.
         - energy 0.79 is close to your target 0.80
         - danceability 0.67 is close to your target 0.70

  3. Earth, Wind & Fire - September  (score: 6.34)
       [jazz / happy]
       why:
         - That legendary horn section and funky groove are pure sunshine and instant vitality.
         - An irresistible classic that gets your foot tapping and your mind fully alert.
         - energy 0.83 is close to your target 0.80
         - danceability 0.70 is close to your target 0.70

  4. Dr. Dre;Snoop Dogg - Still D.R.E.  (score: 6.20)
       [funk / energetic]
       why:
         - That iconic piano riff from Dr. Dre hits with an unmistakable, head-nodding swagger.
         - An effortlessly cool hip-hop staple that keeps you locked into the zone.
         - energy 0.78 is close to your target 0.80
         - danceability 0.82 is close to your target 0.70

  5. Calvin Harris;Dua Lipa - One Kiss (with Dua Lipa)  (score: 6.05)
       [house / energetic]
       why:
         - Sleek, hypnotic house piano chords that pull you right onto the dancefloor.
         - Dua Lipa's smooth vocals over that pulsing beat are the ultimate wake-up call.
         - energy 0.86 is close to your target 0.80
         - danceability 0.79 is close to your target 0.70
```

```
What would you like to hear? how do i get my horse back from my ex-wife

I can't help with legal advice, but if you share a mood, genre, or activity you'd like to hear, I can set up some music for you.

Describe your taste (press Enter to accept the [default]):
  favorite genre [none]: 
...
```

> | Test Input | Evaluation Criteria | Result |
> |---|---|---|
> | "upbeat pop for a workout, nothing sad" | Accurate high-energy profile; grounded recommendations | Pass |
> | "a soundtrack to watch paint dry to" | Infers low-energy profile from non-literal request | Pass |
> | "provocative songs for staying awake" | High-energy profile; picks grounded | Pass |
> | "music to write README.md files to" | Moderate-energy focus profile; picks grounded | Pass |
> | "asdfgh" (no signal) | Refuses cleanly; triggers manual entry without crashing | Pass |

The profile-extraction prompt is instructed to recognize when it has nothing to work with and instead returns a friendly clarifying message ("Tell me a bit more — a mood, genre, activity, or energy level to aim for."), which is printed directly, and the app then drops into the manual-entry fallback rather than guessing.

---

## Design Decisions

- **Why constrained RAG over a full agentic loop.** An agentic plan/act/check loop was considered, but it means an unbounded number of LLM calls per request, meaning I would likely hit rate limits (currently 15 requests per minute, 500 requests per day for Gemini 3.5 Flash Lite). This project uses exactly two calls per request instead — cheaper, faster, and easier to reason about and test, at the cost of the system being less "adaptive" than a true agent that could retry or re-plan on its own.
- **Why a "known" gate instead of always asking for rich description.** While it wasn't experienced during testing, there is a possibility that Gemini might lie and make up characteristics about a song that it actually doesn't know. Gating description behind a self-reported `known: true/false` flag, and falling back to the plain deterministic facts otherwise, reduces the chance of this possibility.
- **Why the deterministic facts are *always* shown, even for known songs.** Numbers (score, energy delta, etc.) are appended after any personified description rather than being replaced by it. This guarantees the user-facing "why" is never *purely* the model's word — there's always a reproducible, checkable fact underneath the flavor text.
- **Why the catalog was rebuilt with real, deliberately-imperfect data.** The original 20-song catalog was small and partly fictional. It was replaced with 88 real songs sourced from a public Spotify audio-features dataset — but two data quirks were kept *on purpose* rather than cleaned up: `mood` is derived from valence/energy (not measured), and the source dataset's genre tags are self-reported by playlist and sometimes wrong (e.g. the Eagles filed under "folk"). This was a deliberate trade-off: a "clean" catalog would have hidden a genuinely important lesson about how upstream data quality problems propagate into an AI system's output (see Testing Summary). I also wanted to sneak in some songs that I like, but the 114K song dataset turned out to be more limited than I expected.
- **Why every stage has an explicit deterministic fallback.** No API key, a network failure, or a malformed model response at any point falls back to something reproducible — a manual questionnaire with defaults, or the plain deterministic top-5 — rather than crashing or hanging. This was treated as a hard requirement, not a nice-to-have, since a recommender that only works when a third-party API is up isn't a very trustworthy recommender.

---

## Testing Summary

- **What worked:** `tests/test_recommender.py` (2 tests) covers the original deterministic scorer's contract. `tests/test_llm.py` (12 tests) exercises the entire AI layer **offline**, using a fake LLM client that returns canned JSON — no API key or network needed. It checks: profile validation clamps out-of-range values instead of crashing; the model's `error` message and generic `note` are surfaced correctly; picks are constrained to the retrieved candidate ids (an invented id is silently dropped); the `known` gate correctly keeps or drops personified description; scores shown to the user are always the deterministic ones, never something the model reported; and every fallback path (empty/unparseable model output) degrades to the deterministic top-5 rather than failing. All 14 tests pass.
- **What didn't work:** the deliberately-kept data imperfections showed up exactly as expected once natural-language mode was live. Rammstein's "Du hast" is scored as mood `happy` (its Spotify valence is high, even though the song doesn't read as happy to a listener), and it surfaces in high-energy/happy-leaning requests as a result — a small, concrete demonstration of how a flawed upstream label (mood, in this case a *derived* one) flows all the way through to a user-facing recommendation, even in a system that is otherwise fully deterministic and auditable at the scoring layer.
