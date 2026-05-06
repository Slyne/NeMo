"""
Handler for calling the backend LangGraph agent service from the S2S pipeline.

When the S2S model detects a SPECIAL_13 token (function-call trigger), this handler:
1. Extracts conversation history from the gen_text and gen_asr_text token streams
2. Sends the history to the backend agent service via /inject
3. Submits an async task via /tasks
4. Waits for an external reinjection request to deliver the final text response
5. Returns the response as token IDs wrapped by SPECIAL_15/SPECIAL_16,
   ready to be injected into the LLM's KV cache via prefill.

Usage:
    handler = BackendAgentHandler(tokenizer, agent_url="http://localhost:8100")
    handler.tool_call_detected(gen_text, gen_asr_text, current_frame_idx)
    # ... later, when response_ready is True:
    token_ids = handler.get_all_response_token_ids()
"""

import os
import uuid
import logging
import threading
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class BackendAgentHandler:
    _handlers_by_session: dict[str, "BackendAgentHandler"] = {}

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
        self.callback_url = os.environ.get("AGENT_CALLBACK_URL", "").strip() or None

        self.session_id = session_id or str(uuid.uuid4())
        self._handlers_by_session[self.session_id] = self
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
        self._task_id: str | None = None
        self._last_delivery_error: str | None = None
        self._synced_turn_count: int = 0

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
        """Backward-compatible alias for async task submission."""
        self.tool_call_detected(gen_text, gen_asr_text, current_frame_idx)

    def tool_call_detected(self, gen_text, gen_asr_text, current_frame_idx: int):
        """Submit an async backend task when SPECIAL_13/14 is first detected."""
        if self._request_sent:
            return

        self._request_sent = True

        turns = self.extract_conversation_turns(gen_text, gen_asr_text, current_frame_idx)

        if not turns:
            logger.warning("No conversation turns found to send to backend agent.")
            return

        history_end = len(turns) - 1
        history_start = min(self._synced_turn_count, history_end)
        history = turns[history_start:history_end]
        last_turn = turns[-1]

        if history_start < history_end:
            logger.info(
                "Launching async backend task (%s new history turns, query=%s)",
                len(history),
                last_turn,
            )
        else:
            logger.info("Launching async backend task (no new history turns, query=%s)", last_turn)
        logger.info("Conversation history being sent to backend agent:")
        for i, turn in enumerate(history, start=history_start):
            logger.info(f"  [{i}] {turn['role']}: {turn['content']}")
        logger.info(f"  [{len(turns) - 1}] {last_turn['role']}: {last_turn['content']} (query)")

        thread = threading.Thread(
            target=self._backend_task_worker,
            args=(history, last_turn, len(turns)),
            daemon=True,
        )
        thread.start()

    def _backend_task_worker(self, history: list[dict], last_turn: dict, observed_turn_count: int):
        """Runs in a background thread — injects history and submits /tasks."""
        try:
            if history:
                resp = requests.post(
                    f"{self.agent_url}/inject",
                    json={"session_id": self.session_id, "turns": history},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                logger.info(f"Injected {len(history)} turns into session {self.session_id}")

            if not self.callback_url:
                raise RuntimeError(
                    "AGENT_CALLBACK_URL is not configured. Set it to the external client callback "
                    "endpoint that will receive the async backend result."
                )

            query_role = last_turn["role"] if last_turn["role"] == "system" else "user"
            message = last_turn["content"]
            resp = requests.post(
                f"{self.agent_url}/tasks",
                json={
                    "session_id": self.session_id,
                    "message": message,
                    "role": query_role,
                    "callback_url": self.callback_url,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            self._task_id = result.get("task_id")
            self._last_delivery_error = None
            # The backend session now has all turns observed up through the
            # current query turn. The tool response itself is not part of the
            # gen_text/gen_asr_text streams, so it must not advance this cursor.
            self._synced_turn_count = observed_turn_count
            logger.info(
                "Submitted async backend task %s for session %s",
                self._task_id,
                self.session_id,
            )

        except Exception as e:
            logger.error(f"Backend agent task submission failed: {e}")
            fallback = "I'm sorry, I couldn't process that request right now."
            self.tool_call_response(fallback)
            self._last_delivery_error = str(e)

    def tool_call_response(self, text: str):
        """Accept an externally delivered backend response and make it prefill-ready."""
        response_text = (text or "").strip()
        if not response_text:
            response_text = "I'm sorry, I couldn't process that request right now."

        logger.info(f"Backend agent response: {response_text[:200]}")
        self._response_token_ids = (
            [self.special_15_id]
            + self.tokenizer.text_to_ids(response_text)
            + [self.special_16_id]
        )
        self._response_idx = 0
        self._pending = True
        self._request_sent = True

    @classmethod
    def inject_response_for_session(cls, session_id: str, text: str) -> bool:
        """Push an externally delivered response into the live handler for a session."""
        handler = cls._handlers_by_session.get(session_id)
        if handler is None:
            logger.warning("No BackendAgentHandler found for session %s", session_id)
            return False
        handler.tool_call_response(text)
        return True

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

    @property
    def task_id(self) -> str | None:
        return self._task_id

    def reset(self):
        """Reset for the next tool call within the same session."""
        self._response_token_ids = []
        self._response_idx = 0
        self._pending = False
        self._request_sent = False
        self._task_id = None
        self._last_delivery_error = None

    def reset_session(self):
        """Full reset with a new session ID."""
        self.reset()
        self._handlers_by_session.pop(self.session_id, None)
        self.session_id = str(uuid.uuid4())
        self._handlers_by_session[self.session_id] = self
        self._synced_turn_count = 0
