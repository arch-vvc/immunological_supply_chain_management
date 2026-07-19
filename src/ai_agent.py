"""
AI Agent — Local LLM Immune System Analyst
============================================
A zero-cost, fully local natural-language layer on top of the immune response
engine's structured chain-of-thought trace (data/stream/immune_decisions.jsonl).

Talks to a local Ollama server (ollama.com) — no API keys, no cloud calls,
no per-token cost. If Ollama isn't running, every method falls back to a
deterministic template so the dashboard never breaks or shows an error.

One-time setup:
    brew install ollama              # or see ollama.com/download
    ollama serve                     # local server on :11434
    ollama pull llama3.2:1b          # ~1.3GB, runs fine on a laptop CPU

Everything downstream of that is just HTTP calls to localhost — same
zero-cost / local-model idea taught in the "Zero to AI Builder" sprint,
applied on top of a real running pipeline instead of a toy demo.
"""

from __future__ import annotations

import json
import time
import requests
from typing import Any, Callable, Optional

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"
TIMEOUT = 30


class ImmuneAIAgent:
    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL):
        self.host = host.rstrip("/")
        self.model = model
        self._checked = False
        self._available = False

    def available(self, force: bool = False) -> bool:
        """Ping the local Ollama server. Cached after the first check."""
        if self._checked and not force:
            return self._available
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        self._checked = True
        return self._available

    def _generate(self, prompt: str, system: str = "", json_mode: bool = False) -> Optional[str]:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 600},
            }
            if json_mode:
                # Ollama's grammar-constrained JSON mode — guarantees syntactically
                # valid JSON even from a small model, unlike asking it to "please
                # output JSON" in plain text.
                payload["format"] = "json"
                payload["options"]["temperature"] = 0.1
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            return None

    def _generate_stream(self, prompt: str, system: str = ""):
        """Same as _generate but yields text chunks as they arrive, for
        st.write_stream(). Ollama's streaming endpoint sends one JSON object
        per line (not SSE) — {"response": "<chunk>", "done": bool, ...}.
        On any failure mid-stream, yields nothing further; the caller falls
        back to a template if no chunks were produced at all."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": True,
                "options": {"temperature": 0.4, "num_predict": 600},
            }
            with requests.post(f"{self.host}/api/generate", json=payload,
                                timeout=TIMEOUT, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break
        except Exception:
            return

    # ── Event narration ──────────────────────────────────────────────────

    def narrate_event(self, decision: dict) -> str:
        """Turn one immune-response decision (thinking trace + verdict) into
        a short plain-English incident briefing."""
        if self.available():
            out = self._generate(
                self._build_narration_prompt(decision),
                system=(
                    "You are a supply chain resilience analyst. Explain the system's "
                    "automated incident response for a non-technical reader in AT MOST 4 "
                    "sentences and under 100 words total — this is a hard limit, stop well "
                    "before you run out of room. Cover only: what was disrupted, the single "
                    "top recommended action, and the estimated recovery time. Be concrete: "
                    "name the entities and numbers given in the data. No headers, no bullet "
                    "points, no markdown, just prose. Do not invent facts that are not "
                    "present in the data."
                ),
            )
            if out:
                return out
        return self._template_narration(decision)

    def _build_narration_prompt(self, decision: dict) -> str:
        v = decision.get("verdict", {})
        lines = [
            f"Manufacturer: {decision.get('manufacturer', '?')}",
            f"Distributor (disrupted node): {decision.get('distributor', '?')}",
            f"Retailer: {decision.get('retailer', '?')}",
            f"Z-score: {decision.get('z_score', 0):.2f}",
            f"Severity: {v.get('severity', '?')}",
            f"Risk rating: {self._risk_rating_10(decision.get('z_score', 0))}/10",
            f"Signals activated: {v.get('signals_activated', '?')}/4",
            f"Estimated recovery: {v.get('recovery_estimate_days', '?')} days",
            "",
            "Reasoning trace:",
        ]
        for step in decision.get("thinking", []):
            lines.append(f"- [{step.get('phase')}] {step.get('reasoning', '')}")
        actions = decision.get("actions", [])
        if actions:
            lines.append("")
            lines.append("Ranked recommended actions:")
            for a in actions:
                lines.append(
                    f"- (priority {a.get('priority')}) {a.get('action')}: {a.get('label')} "
                    f"— confidence {a.get('confidence')}, ETA {a.get('eta_days')} days"
                )
        return "\n".join(lines)

    def _template_narration(self, decision: dict) -> str:
        v = decision.get("verdict", {})
        actions = decision.get("actions", [])
        top = actions[0] if actions else None
        rating = self._risk_rating_10(decision.get("z_score", 0))
        parts = [
            f"A {str(v.get('severity', 'MODERATE')).lower()}-severity disruption was detected at "
            f"[{decision.get('distributor', '?')}] on the route from [{decision.get('manufacturer', '?')}] "
            f"to [{decision.get('retailer', '?')}] (z-score {decision.get('z_score', 0):.2f}, "
            f"risk rating {rating}/10)."
        ]
        if top:
            parts.append(
                f"The top recommended response is to {str(top.get('label', 'take manual action')).lower()}, "
                f"with confidence {top.get('confidence', '?')} and a recovery ETA of {top.get('eta_days', '?')} days."
            )
        rec = v.get("recovery_estimate_days")
        if rec:
            parts.append(f"Historical memory recall estimates full recovery in about {rec} days.")
        parts.append(
            "[Local model unavailable — showing a template summary. Run `ollama serve` for full narration.]"
        )
        return " ".join(parts)

    # ── Chat ────────────────────────────────────────────────────────────

    _CHAT_SYSTEM_PROMPT = (
        "You are the AI analyst for a supply chain immune-response system. "
        "Answer the user's question using ONLY the context provided below. The "
        "context already contains concrete, computed numbers — a risk rating out "
        "of 10, an estimated recovery time in days, a severity tier, and confidence "
        "scores for each recommended action. When asked for a rating, a risk score, "
        "or how long something will take to resolve, READ THE NUMBER DIRECTLY FROM "
        "THE CONTEXT and state it confidently — do not say it's difficult to estimate "
        "or that more information is needed when a number is already given. Only say "
        "information is missing if the specific fact is truly absent below. IMPORTANT: "
        "composite_risk and risk rating are on a scale where HIGHER = MORE risky, LESS "
        "safe. A value near 1.0 (or 10/10) means high danger, not safety — double-check "
        "that your conclusion about whether something is safe/risky/stable points the "
        "same direction as the number before stating it. Keep answers under 150 words, "
        "no markdown headers."
    )

    _TOOL_RESULT_SYSTEM_PROMPT = (
        "Answer using ONLY the verified tool result given above. State numbers "
        "exactly as given — do not round differently or invent additional facts not "
        "present in the result. Do not write code, code blocks, or unrelated examples. "
        "IMPORTANT: composite_risk is on a scale where HIGHER = MORE risky, LESS safe — "
        "a value near 1.0 means high danger, not safety; double-check your conclusion "
        "points the same direction as the number. Under 80 words, plain prose only, no "
        "markdown, no hedging."
    )

    def chat_stream(self, question: str, context: dict, history: list,
                     tools: Optional[dict] = None, meta: Optional[dict] = None):
        """Streaming counterpart to chat() for st.write_stream(). Yields text
        chunks as they arrive from the local model instead of returning the
        full answer at once.

        Since a generator can't also return a value, populate `meta` (pass an
        empty dict in) with the same fields chat() returns in its dict —
        "source", "tool_used", "trace" — as a side channel. Read `meta` after
        the generator is exhausted (st.write_stream() does this for you)."""
        if meta is None:
            meta = {}
        meta.setdefault("source", "template")
        meta.setdefault("tool_used", None)
        meta.setdefault("trace", [])
        trace = meta["trace"]

        if not self.available():
            yield self._template_chat(context)
            return

        if tools:
            routed = self._route_tool(question, tools, trace)
            if routed and not self._is_dead_end(routed[1]):
                tool_name, tool_result = routed
                meta["tool_used"] = tool_name
                meta["source"] = "tool"
                t0 = time.monotonic()
                produced = False
                for chunk in self._answer_with_tool_result_stream(question, tool_name, tool_result):
                    produced = True
                    yield chunk
                trace.append({
                    "step": "Compose answer", "duration_ms": round((time.monotonic() - t0) * 1000),
                    "detail": "Asked the model to phrase the verified tool result in plain English",
                })
                if not produced:
                    yield self._template_tool_result(tool_name, tool_result)
                return
            elif routed:
                # Small local models sometimes route to a tool for a phrasing
                # that doesn't actually name a valid target (e.g. "how risky
                # is the supply chain overall" routed to lookup_entity_risk
                # with entity="supply_chain"). Rather than dead-ending on that
                # wrong guess, fall through to the context-grounded answer —
                # the context already covers general risk/severity/recovery
                # questions.
                trace.append({
                    "step": "Tool found no match — falling back to context",
                    "duration_ms": 0,
                    "detail": f'{routed[0]} found nothing for this phrasing; answering from context instead',
                })

        meta["source"] = "llm"
        t0 = time.monotonic()
        ctx_block = self._build_context_block(context)
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        prompt = f"CONTEXT:\n{ctx_block}\n\nCONVERSATION SO FAR:\n{convo}\n\nQuestion: {question}"
        produced = False
        for chunk in self._generate_stream(prompt, system=self._CHAT_SYSTEM_PROMPT):
            produced = True
            yield chunk
        trace.append({
            "step": "Compose answer from context", "duration_ms": round((time.monotonic() - t0) * 1000),
            "detail": "No tool matched — answered directly from the risk leaderboard and recent events given in context",
        })
        if not produced:
            meta["source"] = "template"
            yield self._template_chat(context)

    @staticmethod
    def _is_dead_end(tool_result: Any) -> bool:
        """True if a routed tool found nothing to answer with (no matching
        entity, or a runtime error) — a signal the routing was probably
        wrong for this phrasing, not that the question is unanswerable."""
        return isinstance(tool_result, dict) and (
            tool_result.get("error") or tool_result.get("found") is False
        )

    def _answer_with_tool_result_stream(self, question: str, tool_name: str, tool_result: Any):
        """Streaming counterpart to _answer_with_tool_result(). Error/not-found
        results are still deterministic (no LLM call, no hallucination risk) —
        yielded as a single chunk rather than streamed token-by-token."""
        if isinstance(tool_result, dict) and (tool_result.get("error") or tool_result.get("found") is False):
            reason = tool_result.get("error") or tool_result.get("reason") or "no match found"
            queried = tool_result.get("queried")
            subject = f' for "{queried}"' if queried else ""
            yield f"{tool_name} could not answer this{subject}: {reason}."
            return

        prompt = (
            f"Tool called: {tool_name}\n"
            f"Verified result (computed directly from live pipeline data, not invented):\n"
            f"{json.dumps(tool_result, default=str)}\n\n"
            f"User question: {question}\n\n"
            f"Write a short, direct answer to the question using ONLY this verified data."
        )
        yield from self._generate_stream(prompt, system=self._TOOL_RESULT_SYSTEM_PROMPT)

    def chat(self, question: str, context: dict, history: list, tools: Optional[dict] = None) -> dict:
        """Answer a free-form question grounded in current pipeline state.

        Returns {"answer": str, "source": "tool"|"llm"|"template", "tool_used": str|None}.
        `source` lets the UI show a confidence cue: "tool" answers are computed
        directly by real pipeline code (deterministic, verified); "llm" answers
        are the model's own synthesis from context (should be flagged as such);
        "template" means no local model was reachable at all.

        `tools` is an optional {name: callable} dict. Each callable's docstring
        is shown to the model as the tool's description, and its keyword
        arguments become the JSON schema the model is asked to fill in. If the
        model selects a tool, it is called for real and its (deterministic,
        verified) result is what the final answer is grounded in — the model
        is choosing *which capability to invoke*, not inventing the answer.
        """
        trace = []
        if self.available():
            if tools:
                routed = self._route_tool(question, tools, trace)
                if routed and not self._is_dead_end(routed[1]):
                    tool_name, tool_result = routed
                    t0 = time.monotonic()
                    answer = self._answer_with_tool_result(question, tool_name, tool_result)
                    trace.append({
                        "step": "Compose answer", "duration_ms": round((time.monotonic() - t0) * 1000),
                        "detail": "Asked the model to phrase the verified tool result in plain English",
                    })
                    return {"answer": answer, "source": "tool", "tool_used": tool_name, "trace": trace}
                elif routed:
                    # See chat_stream() for why: fall back to context instead
                    # of dead-ending on a probably-wrong tool routing guess.
                    trace.append({
                        "step": "Tool found no match — falling back to context",
                        "duration_ms": 0,
                        "detail": f'{routed[0]} found nothing for this phrasing; answering from context instead',
                    })

            t0 = time.monotonic()
            ctx_block = self._build_context_block(context)
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
            prompt = f"CONTEXT:\n{ctx_block}\n\nCONVERSATION SO FAR:\n{convo}\n\nQuestion: {question}"
            out = self._generate(prompt, system=self._CHAT_SYSTEM_PROMPT)
            trace.append({
                "step": "Compose answer from context", "duration_ms": round((time.monotonic() - t0) * 1000),
                "detail": "No tool matched — answered directly from the risk leaderboard and recent events given in context",
            })
            if out:
                return {"answer": out, "source": "llm", "tool_used": None, "trace": trace}
        return {"answer": self._template_chat(context), "source": "template", "tool_used": None, "trace": trace}

    def _route_tool(self, question: str, tools: dict, trace: Optional[list] = None) -> Optional[tuple]:
        """Ask the model which tool (if any) answers this question, using
        Ollama's grammar-constrained JSON mode so even a small model produces
        syntactically valid output. Returns (tool_name, result) or None.
        Appends timed steps to `trace` if given, for the UI's Execution Trace panel."""
        tool_lines = "\n".join(
            f'- "{name}": {(fn.__doc__ or "no description").strip().splitlines()[0]}'
            for name, fn in tools.items()
        )
        system = (
            "You are a routing layer for a supply chain analysis system. Given the user's "
            "question and the list of available tools below, decide whether ONE tool should "
            "be called to answer it precisely. Respond with ONLY a JSON object of the form "
            '{"tool": "<exact tool name or none>", "args": {<arguments the tool needs>}}. '
            "Use \"none\" if no listed tool clearly answers the question — general questions "
            "about severity, recovery time, or risk rating of the current incident should use "
            "\"none\" since those are already answered elsewhere. Only ever use tool names "
            "exactly as spelled in the list, never invent a new one."
        )
        prompt = f"Available tools:\n{tool_lines}\n\nUser question: {question}"
        t0 = time.monotonic()
        raw = self._generate(prompt, system=system, json_mode=True)
        routing_ms = round((time.monotonic() - t0) * 1000)
        if not raw:
            if trace is not None:
                trace.append({"step": "Route", "duration_ms": routing_ms, "detail": "No response from local model — routing skipped"})
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            if trace is not None:
                trace.append({"step": "Route", "duration_ms": routing_ms, "detail": f"Model returned invalid JSON, ignored: {raw[:80]}"})
            return None
        name = parsed.get("tool") if isinstance(parsed, dict) else None
        if not name or name == "none" or name not in tools:
            if trace is not None:
                trace.append({"step": "Route", "duration_ms": routing_ms, "detail": f'Model decided no tool applies (chose "{name}")'})
            return None
        args = parsed.get("args") if isinstance(parsed, dict) else {}
        if not isinstance(args, dict):
            args = {}
        if trace is not None:
            trace.append({
                "step": "Route", "duration_ms": routing_ms,
                "detail": f'Selected tool "{name}" with args {args}',
            })
        t1 = time.monotonic()
        try:
            result = self._call_tool_leniently(tools[name], args)
            call_detail = f"{name}() returned {'a result' if not (isinstance(result, dict) and result.get('error')) else 'an error'}"
        except Exception as e:
            result = {"error": str(e)}
            call_detail = f"{name}() raised: {e}"
        if trace is not None:
            trace.append({
                "step": "Execute tool", "duration_ms": round((time.monotonic() - t1) * 1000),
                "detail": call_detail,
            })
        return name, result

    @staticmethod
    def _call_tool_leniently(fn: Callable, raw_args: dict) -> Any:
        """A small local model reliably picks the right tool far more often
        than it reproduces the tool's exact Python parameter name in its JSON
        args (e.g. it'll send {"entity_name": ...} for a param literally
        called `entity`). Map whatever values it provided onto the tool's
        real parameters *by position*, not by key, so routing isn't defeated
        by a spelling mismatch the model was never going to get exactly right."""
        import inspect
        params = list(inspect.signature(fn).parameters.keys())
        if not params:
            return fn()
        values = [v for v in raw_args.values() if v is not None]
        if not values:
            raise TypeError(f"{fn.__name__} needs an argument (e.g. an entity name) but none was given")
        return fn(*values[: len(params)])

    def explain_tool_result(self, question: str, tool_name: str, tool_result: Any) -> str:
        """Public entry point for when the calling app already knows which
        tool applies (e.g. a dedicated "what if X fails" UI control, not a
        free-text chat question) and just wants the verified result narrated.
        Skips the LLM routing step entirely — routing is where a small local
        model is least reliable, so anything the UI can determine
        deterministically should never go through it."""
        if self.available():
            return self._answer_with_tool_result(question, tool_name, tool_result)
        return self._template_tool_result(tool_name, tool_result)

    def _answer_with_tool_result(self, question: str, tool_name: str, tool_result: Any) -> str:
        # A tiny local model asked to explain a failed lookup tends to invent a
        # plausible-sounding but completely fabricated explanation rather than
        # admitting the lookup failed. For error/not-found results, skip the
        # model entirely and report the failure verbatim — deterministic, no
        # hallucination risk, and more honest than a confident-sounding guess.
        if isinstance(tool_result, dict) and (tool_result.get("error") or tool_result.get("found") is False):
            reason = tool_result.get("error") or tool_result.get("reason") or "no match found"
            queried = tool_result.get("queried")
            subject = f' for "{queried}"' if queried else ""
            return f"{tool_name} could not answer this{subject}: {reason}."

        prompt = (
            f"Tool called: {tool_name}\n"
            f"Verified result (computed directly from live pipeline data, not invented):\n"
            f"{json.dumps(tool_result, default=str)}\n\n"
            f"User question: {question}\n\n"
            f"Write a short, direct answer to the question using ONLY this verified data."
        )
        out = self._generate(prompt, system=self._TOOL_RESULT_SYSTEM_PROMPT)
        return out or self._template_tool_result(tool_name, tool_result)

    def _template_tool_result(self, tool_name: str, tool_result: Any) -> str:
        return f"[Local model unavailable] {tool_name} result: {json.dumps(tool_result, default=str)}"

    @staticmethod
    def _risk_rating_10(z_score: float) -> int:
        """Deterministic 1-10 risk rating derived from the same z-score the
        pipeline already uses for severity tiers (MODERATE/HIGH/CRITICAL) —
        not left to the LLM to invent. z=2.5 (detection threshold) -> ~4,
        z=5 (CRITICAL threshold) -> ~8, z>=6.7 -> 10."""
        try:
            return max(1, min(10, round(float(z_score) * 1.5)))
        except (TypeError, ValueError):
            return 0

    def _build_context_block(self, context: dict) -> str:
        lines = []
        if context.get("top_risk"):
            lines.append("Top risk entities (composite_risk, 0-1 scale):")
            for r in context["top_risk"][:5]:
                lines.append(f"  - {r['entity']}: {r['composite_risk']:.3f}")
        if context.get("recent_events"):
            lines.append("")
            lines.append("Recent immune response events (most recent last):")
            for e in context["recent_events"][:5]:
                v = e.get("verdict", {})
                z = e.get("z_score", 0)
                rating = self._risk_rating_10(z)
                actions = e.get("actions", [])
                top_action = actions[0] if actions else {}
                lines.append(
                    f"  - {e.get('timestamp', '')[:19]}: [{e.get('distributor', '?')}] disrupted "
                    f"(mfr={e.get('manufacturer', '?')}, retailer={e.get('retailer', '?')}). "
                    f"z-score={z:.2f}, severity={v.get('severity', '?')}, "
                    f"RISK RATING = {rating}/10, "
                    f"signals activated={v.get('signals_activated', '?')}/4, "
                    f"ESTIMATED RECOVERY = {v.get('recovery_estimate_days', '?')} DAYS. "
                    f"Top recommended action: {top_action.get('label', 'none')} "
                    f"(confidence {top_action.get('confidence', '?')})."
                )
        if context.get("anomaly_summary"):
            lines.append("")
            lines.append(f"Anomaly summary: {context['anomaly_summary']}")
        wi = context.get("whatif_result")
        if wi and wi.get("found"):
            lines.append("")
            lines.append(
                f"Most recent what-if simulation: if [{wi.get('node')}] fails, "
                f"{wi.get('reroutable_dependents')} of {wi.get('downstream_dependents')} downstream "
                f"dependents can still be reached via an alternate route, and "
                f"{wi.get('unreachable_dependents')} cannot (sample: {wi.get('sample_unreachable')})."
            )
        return "\n".join(lines) if lines else "No live data available yet."

    def _template_chat(self, context: dict) -> str:
        events = context.get("recent_events", [])
        if not events:
            return (
                "[Local model unavailable] I don't have any live decision data to answer from yet — "
                "start the stream simulator and consumer, then ask again."
            )
        latest = events[-1]
        v = latest.get("verdict", {})
        rating = self._risk_rating_10(latest.get("z_score", 0))
        return (
            f"[Local model unavailable — template answer] The most recent event was a "
            f"{v.get('severity', '?')} disruption at [{latest.get('distributor', '?')}] — "
            f"risk rating {rating}/10, estimated recovery {v.get('recovery_estimate_days', '?')} days. "
            f"Run `ollama serve` for full conversational answers."
        )
