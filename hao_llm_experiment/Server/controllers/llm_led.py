# basil_module/controllers/llm_led.py
from __future__ import annotations
import os, json, subprocess, requests, datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np

DEFAULT_MODEL = os.getenv('LLM_MODEL', 'gpt-oss:120b')
OLLAMA_BASE = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
LLM_TIMEOUT_SECONDS = int(os.getenv('LLM_TIMEOUT_SECONDS', '180'))
OLLAMA_KEEP_ALIVE = os.getenv('OLLAMA_KEEP_ALIVE', '30m')

@dataclass
class LLMLEDPolicy:
    temp_min: float = 20.0
    temp_max: float = 30.0

@dataclass
class LLMControlLimits:
    target_ppfd_min: float = 0.0
    target_ppfd_max: float = 400.0  # Adjust based on your max hardware capability

@dataclass
class LLMLoggerConfig:
    enabled: bool = True
    log_dir: str = "logs/llm_led"
    session_name: Optional[str] = None
    log_full_context: bool = True
    timezone_name: Optional[str] = None


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _normalize_explanation(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        lines = [line.strip(" -•\t") for line in value.splitlines()]
        cleaned = [line for line in lines if line]
        return cleaned or [value]
    if value is None:
        return []
    return [str(value)]

def _build_prompt(observations: Dict[str, Any], context: Dict[str, Any], forecast: Dict[str, Any], policy: Dict[str, Any]) -> str:
    llm_observations = dict(observations)
    # Day/night control is derived from the edge-computed is_day flag, not from
    # the clock string. Keep local_time out of the prompt to avoid semantic drift.
    llm_observations.pop("local_time", None)
    temp_min = policy.get("temp_min")
    temp_max = policy.get("temp_max")
    temp_range_text = (
        f"[{temp_min}, {temp_max}]"
        if temp_min is not None and temp_max is not None
        else "[temp_min, temp_max]"
    )
    blocks = [
        "You are a greenhouse controls engineer. Choose a target PPFD to maximize photosynthesis (Pn) while minimizing energy $/kWh cost.",
        'Return STRICT JSON only with keys { "target_ppfd", "rationale", "explanation" }. No extra text. No markdown.',
        '\n[Observations]\n' + json.dumps(llm_observations),
        '\n[Physics_Context]\n' + json.dumps(context),
        '\n[Temperature_Forecast]\n' + json.dumps(forecast),
        '\n[Objectives_And_Penalties]\n' + json.dumps(policy),
        "\n[Actuator_Constraints]\n"
        "- LED actuator limits are hard bounds.\n"
        "- Red PWM range: 0 to 100 percent.\n"
        "- Blue PWM range: 0 to 100 percent.\n"
        "- Values above 100 percent cannot be executed by the hardware.\n"
        "- If the requested target_ppfd would require either LED channel to exceed 100 percent PWM, the edge node will saturate at the hardware limit.\n"
        "- Once PWM is saturated, increasing target_ppfd further may not increase actual PPFD.\n"
        "- Choose a realistic target_ppfd that respects actuator saturation and current plant conditions.\n",
        "\n[Reasoning_Instructions]\n"
        "- Respect actuator limits.\n"
        "- Treat 'is_day' as the only authoritative day/night signal.\n"
        "- Ignore clock-time semantics for day/night classification.\n"
        "- If is_day == 0, set target_ppfd to 0.0 for dark respiration.\n"
        f"- Keep indoor temperature within {temp_range_text}.\n"
        "- If tleaf_now >= temp_max or next_1h_temp_estimate >= temp_max, do not increase PPFD.\n"
        "- If temperature is at or above temp_max, prefer reducing PPFD below the current level.\n"
        "- When temperature is near the upper limit, prioritize thermal relief over photosynthesis gains.\n"
        "- Consider electricity_price_$per_kWh to weigh energy cost.\n"
        "- OUTPUT STRICT JSON ONLY with fields: target_ppfd, rationale, explanation.\n"
        "- 'explanation' must be a short step-by-step list (3-6 bullets, <=20 tokens each)."
    ]
    return "\n".join(blocks)

def call_llm(prompt: str) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    data = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.2},
    }
    try:
        r = requests.post(url, json=data, timeout=LLM_TIMEOUT_SECONDS, stream=True)
        r.raise_for_status()
        return ''.join(json.loads(line)['response'] for line in r.iter_lines() if line and 'response' in json.loads(line))
    except Exception as e:
        print(f"HTTP call failed, trying CLI. Error: {e}")
        cmd = ["ollama", "run", DEFAULT_MODEL]
        try:
            proc = subprocess.run(cmd, input=prompt.encode('utf-8'), capture_output=True, timeout=LLM_TIMEOUT_SECONDS)
            return proc.stdout.decode('utf-8', errors='ignore')
        except Exception as cli_error:
            raise RuntimeError(f"LLM call failed via HTTP and CLI: {cli_error}") from cli_error

def _validate_and_parse(raw: str, limits: LLMControlLimits) -> Dict[str, Any]:
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end == -1: raise ValueError("No JSON found")
    js = json.loads(raw[start:end+1])

    ppfd = float(js.get('target_ppfd', 0.0))
    js['target_ppfd'] = float(np.clip(ppfd, limits.target_ppfd_min, limits.target_ppfd_max))
    js.setdefault('rationale', '')
    js['explanation'] = _normalize_explanation(js.get('explanation', []))
    return js

class LLMLEDController:
    def __init__(self, limits=None, policy=None, model_name=None, logger: Optional[LLMLoggerConfig] = None):
        self.limits = limits or LLMControlLimits()
        self.policy = policy or LLMLEDPolicy()
        self.logger = logger or LLMLoggerConfig(enabled=False)
        if model_name:
            global DEFAULT_MODEL
            DEFAULT_MODEL = model_name

    def _now(self) -> datetime.datetime:
        if self.logger.timezone_name:
            return datetime.datetime.now(ZoneInfo(self.logger.timezone_name))
        return datetime.datetime.now()

    def _write_log(self, payload: Dict[str, Any]) -> None:
        if not self.logger.enabled:
            return

        os.makedirs(self.logger.log_dir, exist_ok=True)
        now = self._now()
        session = self.logger.session_name or now.strftime("%Y%m%d")
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.logger.log_dir, f"{session}_{timestamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _apply_temperature_guard(
        self,
        decision: Dict[str, Any],
        obs: Dict[str, Any],
        physics_ctx: Dict[str, Any],
        forecast: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        temp_max = _safe_float(self.policy.temp_max)
        if temp_max is None:
            return decision, None

        monitored_temps = {
            "tleaf_now": _safe_float(obs.get("tleaf_now")),
            "next_1h_temp_estimate": _safe_float(forecast.get("next_1h_temp_estimate")),
        }
        valid_temps = {name: value for name, value in monitored_temps.items() if value is not None}
        if not valid_temps:
            return decision, None

        trigger_name, trigger_temp = max(valid_temps.items(), key=lambda item: item[1])
        if trigger_temp < temp_max:
            return decision, None

        ceiling_candidates = [
            _safe_float(obs.get("ppfd_now")),
            _safe_float(physics_ctx.get("last_target_ppfd")),
        ]
        valid_ceilings = [value for value in ceiling_candidates if value is not None]
        guard_ceiling = min(valid_ceilings) if valid_ceilings else float(decision.get("target_ppfd", 0.0))
        guarded_ppfd = float(min(float(decision.get("target_ppfd", 0.0)), guard_ceiling))

        if guarded_ppfd >= float(decision.get("target_ppfd", 0.0)):
            return decision, None

        updated = dict(decision)
        previous_ppfd = float(decision["target_ppfd"])
        updated["target_ppfd"] = guarded_ppfd
        updated["rationale"] = (
            f"{decision.get('rationale', '').strip()} "
            f"Hard temperature guard reduced target_ppfd because {trigger_name}={trigger_temp:.2f}C >= {temp_max:.2f}C."
        ).strip()
        explanation = list(_normalize_explanation(updated.get("explanation", [])))
        explanation.append(
            f"Hard guard: {trigger_name}={trigger_temp:.2f}C >= {temp_max:.2f}C."
        )
        explanation.append(
            f"Prevent increase above current ceiling {guard_ceiling:.1f}."
        )
        updated["explanation"] = explanation
        guard_info = {
            "trigger_metric": trigger_name,
            "trigger_temp": trigger_temp,
            "temp_max": temp_max,
            "guard_ceiling_ppfd": guard_ceiling,
            "previous_target_ppfd": previous_ppfd,
            "guarded_target_ppfd": guarded_ppfd,
        }
        return updated, guard_info

    def decide(self, obs: Dict, physics_ctx: Dict, forecast: Dict) -> Dict[str, Any]:
        if int(obs.get("is_day", 1)) == 0:
            decision = {
                "target_ppfd": 0.0,
                "rationale": "is_day=0, so lights stay off for the dark period.",
                "explanation": [
                    "Edge marked this interval as night.",
                    "Night control is determined strictly by is_day.",
                    "Set PPFD to zero for dark respiration.",
                ],
            }
            self._write_log({
                "model": DEFAULT_MODEL,
                "observations": obs if self.logger.log_full_context else None,
                "physics_context": physics_ctx if self.logger.log_full_context else None,
                "forecast": forecast if self.logger.log_full_context else None,
                "prompt": None,
                "raw_response": None,
                "decision": decision,
                "decision_source": "is_day_guard",
            })
            return decision

        prompt = _build_prompt(obs, physics_ctx, forecast, self.policy.__dict__)
        raw = ""
        try:
            raw = call_llm(prompt)
            decision = _validate_and_parse(raw, self.limits)
            decision, temperature_guard = self._apply_temperature_guard(decision, obs, physics_ctx, forecast)
            self._write_log({
                "model": DEFAULT_MODEL,
                "observations": obs if self.logger.log_full_context else None,
                "physics_context": physics_ctx if self.logger.log_full_context else None,
                "forecast": forecast if self.logger.log_full_context else None,
                "prompt": prompt,
                "raw_response": raw,
                "decision": decision,
                "temperature_guard": temperature_guard,
            })
            return decision
        except Exception as e:
            print(f"LLM Error: {e}")
            # Safe Fallback
            fallback = {
                "target_ppfd": 0.0,
                "rationale": "fallback",
                "explanation": [f"{type(e).__name__}: {e}"],
            }
            self._write_log({
                "model": DEFAULT_MODEL,
                "observations": obs if self.logger.log_full_context else None,
                "physics_context": physics_ctx if self.logger.log_full_context else None,
                "forecast": forecast if self.logger.log_full_context else None,
                "prompt": prompt,
                "raw_response": raw,
                "error": str(e),
                "decision": fallback,
            })
            return fallback
