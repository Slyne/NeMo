"""
Handler for calling the backend LangGraph agent service from the S2S pipeline.

When the S2S model detects a SPECIAL_13 token (function-call trigger), this handler:
1. Extracts conversation history from the gen_text and gen_asr_text token streams
2. Sends the history to the backend agent service via /inject
3. Calls /query with the latest user utterance
4. Returns the agent's response as token IDs wrapped by SPECIAL_15/SPECIAL_16,
   ready to be injected into the LLM's KV cache via prefill.

Usage:
    handler = BackendAgentHandler(tokenizer, agent_url="http://localhost:8100")
    handler.on_special_13_detected(gen_text, gen_asr_text, current_frame_idx)
    # ... later, when response_ready is True:
    token_ids = handler.get_all_response_token_ids()
"""

import os
import re
import uuid
import logging
import threading
import requests
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_for_speech(text: str) -> str:
    """Convert symbols and formatting into speech-friendly text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    text = text.replace('°C', ' degrees Celsius')
    text = text.replace('°F', ' degrees Fahrenheit')
    text = text.replace('°', ' degrees')

    text = text.replace('%', ' percent')
    text = text.replace('&', ' and ')
    text = text.replace('+', ' plus ')
    text = text.replace('$', ' dollars ')
    text = text.replace('€', ' euros ')
    text = text.replace('£', ' pounds ')

    text = text.replace('\u202f', ' ')
    text = text.replace('\u00a0', ' ')
    text = text.replace('\xa0', ' ')

    text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

    text = text.replace('\n', ' ')
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


class BackendAgentHandler:
    def __init__(
        self,
        tokenizer,
        agent_url: str | None = None,
        session_id: str | None = None,
        timeout: float = 120.0,
    ):
        self.tokenizer = tokenizer
        self.agent_url = agent_url or os.environ.get("AGENT_SERVICE_URL", "http://localhost:8100")
        self.timeout = timeout

        self.session_id = session_id or str(uuid.uuid4())
        logger.info(f"BackendAgentHandler initialized (session={self.session_id}, url={self.agent_url})")

        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = getattr(tokenizer, 'pad_id', None)
        if self.pad_id is None:
            self.pad_id = tokenizer.text_to_ids('<SPECIAL_12>')[0]

        self.special_15_id = tokenizer.text_to_ids('<SPECIAL_15>')[0]
        self.special_16_id = tokenizer.text_to_ids('<SPECIAL_16>')[0]
        self._response_token_ids: list[int] = []
        self._response_idx: int = 0
        self._pending: bool = False
        self._request_sent: bool = False

    def extract_conversation_turns(self, gen_text, gen_asr_text, up_to_frame: int) -> list[dict]:
        """Extract user and agent conversation turns from the token streams.

        gen_text contains agent tokens with BOS/EOS delimiters.
        gen_asr_text contains user (ASR) tokens — no BOS marker (it's injected as
        an embedding), pad tokens (12) interspersed between words, EOS (2) at turn end.
        """
        agent_segments = self._extract_segments(gen_text[0, :up_to_frame])
        user_segments = self._extract_user_segments(gen_asr_text[0, :up_to_frame])

        all_turns = []
        for start, end, text in user_segments:
            all_turns.append({"start": start, "role": "user", "content": text})
        for start, end, text in agent_segments:
            all_turns.append({"start": start, "role": "assistant", "content": text})

        all_turns.sort(key=lambda t: t["start"])
        return [{"role": t["role"], "content": t["content"]} for t in all_turns if t["content"].strip()]

    def _extract_segments(self, token_ids_tensor) -> list[tuple[int, int, str]]:
        """Extract agent text segments delimited by BOS (1) / EOS (2) from a 1D token tensor."""
        token_ids = token_ids_tensor.tolist()
        segments = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.bos_id:
                start = i
                j = i + 1
                toks = []
                while j < len(token_ids) and token_ids[j] != self.eos_id:
                    if token_ids[j] != self.pad_id:
                        toks.append(token_ids[j])
                    j += 1
                if toks:
                    text = self.tokenizer.ids_to_text(toks)
                    segments.append((start, j, text))
                i = j + 1
            else:
                i += 1
        return segments

    def _extract_user_segments(self, token_ids_tensor) -> list[tuple[int, int, str]]:
        """Extract user/ASR text segments from a 1D token tensor.

        User tokens have no discrete BOS — it is injected as an embedding.
        Pad tokens (12) are interspersed between words. EOS (2) marks turn end.
        Segments are split on EOS boundaries.
        """
        token_ids = token_ids_tensor.tolist()
        segments = []
        toks = []
        start = None
        for i, tid in enumerate(token_ids):
            if tid == self.eos_id:
                if toks:
                    text = self.tokenizer.ids_to_text(toks)
                    segments.append((start, i, text))
                toks = []
                start = None
            elif tid != self.pad_id:
                if start is None:
                    start = i
                toks.append(tid)
        if toks:
            text = self.tokenizer.ids_to_text(toks)
            segments.append((start, len(token_ids), text))
        return segments

    def on_special_13_detected(self, gen_text, gen_asr_text, current_frame_idx: int):
        """Called when SPECIAL_13 is first detected. Fires the backend request in a background thread
        so the inference loop can keep generating tokens without blocking."""
        if self._request_sent:
            return

        self._request_sent = True

        turns = self.extract_conversation_turns(gen_text, gen_asr_text, current_frame_idx)

        if not turns:
            logger.warning("No conversation turns found to send to backend agent.")
            return

        history = turns[:-1]
        last_turn = turns[-1]

        logger.info(f"Launching async backend request (history={len(history)} turns, query={last_turn})")
        logger.info("Conversation history being sent to backend agent:")
        for i, turn in enumerate(turns):
            logger.info(f"  [{i}] {turn['role']}: {turn['content']}")

        thread = threading.Thread(
            target=self._backend_request_worker,
            args=(history, last_turn),
            daemon=True,
        )
        thread.start()

    def _backend_request_worker(self, history: list[dict], last_turn: dict):
        """Runs in a background thread — sends inject + query to the backend agent."""
        try:
            if history:
                resp = requests.post(
                    f"{self.agent_url}/inject",
                    json={"session_id": self.session_id, "turns": history},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                logger.info(f"Injected {len(history)} turns into session {self.session_id}")

            query_role = last_turn["role"] if last_turn["role"] == "system" else "user"
            message = (
                last_turn["content"]
                + "\n\nIMPORTANT: Respond in one or two short conversational sentences only."
            )
            resp = requests.post(
                f"{self.agent_url}/query",
                json={
                    "session_id": self.session_id,
                    "message": message,
                    "role": query_role,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            response_text = result.get("response", "")

            logger.info(f"Backend agent response (raw): {response_text[:200]}")
            response_text = normalize_for_speech(response_text)
            logger.info(f"Backend agent response (normalized): {response_text[:200]}")

            self._response_token_ids = (
                [self.special_15_id]
                + self.tokenizer.text_to_ids(response_text)
                + [self.special_16_id]
            )
            self._response_idx = 0
            self._pending = True

        except Exception as e:
            logger.error(f"Backend agent call failed: {e}")
            fallback = "I'm sorry, I couldn't process that request right now."
            self._response_token_ids = (
                [self.special_15_id]
                + self.tokenizer.text_to_ids(fallback)
                + [self.special_16_id]
            )
            self._response_idx = 0
            self._pending = True

    def get_next_token(self) -> Optional[int]:
        """Get the next token from the backend response, or None if done."""
        if not self._pending or self._response_idx >= len(self._response_token_ids):
            return None
        token = self._response_token_ids[self._response_idx]
        self._response_idx += 1
        if self._response_idx >= len(self._response_token_ids):
            self._pending = False
        return token

    @property
    def response_ready(self) -> bool:
        """True when the async backend response has arrived and is available for prefill."""
        return self._pending and len(self._response_token_ids) > 0

    def get_all_response_token_ids(self) -> Optional[list[int]]:
        """Return all response token IDs (including SPECIAL_15/16 wrapping) for prefill injection.
        Returns None if not ready. Consumes the response (single use)."""
        if not self._pending:
            return None
        self._pending = False
        return self._response_token_ids

    @property
    def is_injecting(self) -> bool:
        return self._pending

    @property
    def is_done(self) -> bool:
        return self._request_sent and not self._pending

    def reset(self):
        """Reset for the next tool call within the same session."""
        self._response_token_ids = []
        self._response_idx = 0
        self._pending = False
        self._request_sent = False

    def reset_session(self):
        """Full reset with a new session ID."""
        self.reset()
        self.session_id = str(uuid.uuid4())
