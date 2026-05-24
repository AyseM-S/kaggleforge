"""
AI Brain Module
Handles all interactions with the local Ollama LLM.
No API key needed — runs entirely on your machine.
"""

import ollama
from typing import Generator, Optional
from core.prompt_templates import (
    SYSTEM_PROMPT,
    ANALYSIS_PROMPT,
    EXPLAIN_PROMPT,
    CHAT_PROMPT,
)


class AIBrain:
    """Local AI brain powered by Ollama. No API key, no rate limits."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.chat_history: list[dict] = []
        self._competition_rules = ""
        self._data_summary = ""
        self._solution_status = "No solution generated yet."

    @staticmethod
    def check_connection() -> tuple[bool, str]:
        """Check if Ollama server is running and accessible."""
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models] if models.models else []
            if model_names:
                return True, f"Connected. Available models: {', '.join(model_names)}"
            else:
                return True, "Connected but no models installed. Run: ollama pull llama3.2:3b"
        except Exception as e:
            return False, f"Cannot connect to Ollama: {e}. Make sure Ollama is running (ollama serve)."

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of locally available Ollama models."""
        try:
            models = ollama.list()
            return [m.model for m in models.models] if models.models else []
        except Exception:
            return []

    def set_context(self, rules: str = "", data_summary: str = "", solution_status: str = ""):
        """Update the context for chat conversations."""
        if rules:
            self._competition_rules = rules
        if data_summary:
            self._data_summary = data_summary
        if solution_status:
            self._solution_status = solution_status

    def analyze_competition(self, rules: str, data_summary: str) -> str:
        """Analyze competition rules + data and return a strategy."""
        self._competition_rules = rules
        self._data_summary = data_summary

        prompt = ANALYSIS_PROMPT.format(rules=rules, data_summary=data_summary)

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        return response["message"]["content"]

    def explain_results(self, rules: str, data_summary: str, model_results: str, best_model: str) -> str:
        """Explain AutoML results in plain language."""
        prompt = EXPLAIN_PROMPT.format(
            rules=rules,
            data_summary=data_summary,
            model_results=model_results,
            best_model=best_model,
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        self._solution_status = f"Best model: {best_model}\n{model_results}"
        return response["message"]["content"]

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """Stream a chat response. Yields tokens one by one."""
        # Build system message with current context
        system_content = SYSTEM_PROMPT + "\n\n"
        if self._competition_rules:
            system_content += CHAT_PROMPT.format(
                rules=self._competition_rules[:2000],  # Truncate to save context
                data_summary=self._data_summary[:2000],
                solution_status=self._solution_status[:1000],
            )

        # Build messages list
        messages = [{"role": "system", "content": system_content}]
        # Add recent chat history (last 10 exchanges to save memory)
        messages.extend(self.chat_history[-20:])
        messages.append({"role": "user", "content": user_message})

        # Stream response
        full_response = ""
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            yield token

        # Save to history
        self.chat_history.append({"role": "user", "content": user_message})
        self.chat_history.append({"role": "assistant", "content": full_response})

    def chat_sync(self, user_message: str) -> str:
        """Non-streaming chat (returns full response at once)."""
        result = ""
        for token in self.chat(user_message):
            result += token
        return result

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
