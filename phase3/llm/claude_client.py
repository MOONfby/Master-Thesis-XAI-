"""
Claude API client with streaming support and conversation history management.
"""
from __future__ import annotations

from typing import Generator

import anthropic

from phase3.config import LLM_MODEL, MAX_TOKENS


class ClaudeExplanationClient:
    """
    Thin wrapper around the Anthropic messages API.

    Handles streaming responses and conversation history.
    """

    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    def stream_response(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        """
        Stream a response from Claude.

        Parameters
        ----------
        system_prompt : str
            The system-level instructions (persona + grounding rules).
        messages : list[dict]
            Conversation history as [{"role": "user"|"assistant", "content": str}, ...].

        Yields
        ------
        str — text chunks as they arrive.
        """
        with self._client.messages.stream(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    @staticmethod
    def append_turn(
        messages: list[dict],
        role: str,
        content: str,
    ) -> list[dict]:
        """
        Append a message turn to the conversation history.

        Returns a new list (does not mutate the input).
        """
        return messages + [{"role": role, "content": content}]

    @staticmethod
    def build_first_user_message(context_block: str, opening_question: str) -> str:
        """
        Combine the grounded context block and the opening question into the
        first user message. The context block is labelled so the LLM knows
        these are grounded facts that must not be contradicted.
        """
        return (
            "=== EXPLANATION CONTEXT (GROUNDED DATA — DO NOT CONTRADICT) ===\n"
            + context_block
            + "\n\n=== YOUR QUESTION ===\n"
            + opening_question
        )
