"""
Transport layer for the LLM feature.

`GeminiClient` knows *how to reach Gemini* and nothing about music, profiles, or
prompts. src/llm.py owns the prompts and the recommendation logic and calls this
to actually talk to the model. Keeping the provider confined here means:
  - the no-key / deterministic path imports none of this,
  - google-genai lives in exactly one module,
  - llm.py can be tested by injecting a fake client with the same `complete()`.
"""

import os


class GeminiClient:
    """
    Minimal single-shot wrapper around google-genai.

    Requirements:
      - google-genai installed (see requirements.txt)
      - GEMINI_API_KEY set in the environment or a .env file
    """

    def __init__(
        self,
        model_name: str = "gemini-flash-lite-latest",
        temperature: float = 0.3,
    ) -> None:
        # Load a .env if present so GEMINI_API_KEY can live in a git-ignored
        # file. python-dotenv is optional here: skip quietly if it's missing.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set it in your environment or a .env "
                "file (GEMINI_API_KEY=...). Without a key, use the deterministic "
                "path instead of constructing this client."
            )

        # Import here so the offline path never needs the dependency installed.
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = float(temperature)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_output: bool = False,
    ) -> str:
        """
        Send one request and return the model's text.

        The system prompt is passed as Gemini's real `system_instruction` (the
        durable role/rules), while only the per-call input goes in `contents` —
        so the model treats the listener's text as data, not as new rules.

        When `json_output` is True, the response is constrained to JSON mode
        (`response_mime_type="application/json"`), so the caller gets parseable
        JSON to `json.loads` without scraping it out of prose. The concrete
        shape is described in the prompt and enforced by the caller's validators
        (llm.validate_profile and the candidate-id constraint), which keeps this
        robust across google-genai versions.

        Returns "" on ANY error (missing text, safety block, network, quota) so
        the caller can fall back to its deterministic path instead of crashing.
        """
        from google.genai import types

        config_kwargs = {
            "temperature": self.temperature,
            "system_instruction": system_prompt,
        }
        if json_output:
            config_kwargs["response_mime_type"] = "application/json"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            # response.text can be None if the output was blocked by filters.
            return response.text or ""
        except Exception:
            # Empty string is the agreed failure signal; llm.py degrades to the
            # deterministic recommendation instead.
            return ""
