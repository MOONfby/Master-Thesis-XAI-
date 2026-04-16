"""
OpenAI API client with streaming support and conversation history management.
"""
from __future__ import annotations

from typing import Generator

from openai import OpenAI

from phase3.config import LLM_MODEL, MAX_TOKENS


class ClaudeExplanationClient:
    """
    Wrapper around the OpenAI chat completions API.

    Kept the class name unchanged so no other files need to be modified.
    Handles streaming responses and conversation history.
    """

    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    def stream_response(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        """
        Stream a response from the model.

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
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        stream = self._client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            messages=full_messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

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
        first user message.
        """
        return (
            "=== EXPLANATION CONTEXT (GROUNDED DATA — DO NOT CONTRADICT) ===\n"
            + context_block
            + "\n\n=== YOUR QUESTION ===\n"
            + opening_question
        )
