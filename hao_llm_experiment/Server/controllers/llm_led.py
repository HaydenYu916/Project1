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

def _build_prompt(observations: Dict[str, Any], context: Dict[str, Any], forecast: Dict[str, Any], policy: Dict[str, Any]) -> str:
    blocks = [
        "You are a greenhouse controls engineer. Choose a target PPFD to maximize photosynthesis (Pn) while minimizing energy $/kWh cost.",
        'Return STRICT JSON only with keys { "target_ppfd", "rationale", "explanation" }. No extra text. No markdown.',
        '\n[Observations]\n' + json.dumps(observations),
        '\n[Physics_Context]\n' + json.dumps(context),
        '\n[Outdoor Temperature]\n' + json.dumps(forecast),
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
        "- Check 'local_time' and 'is_day'. If night, set target_ppfd to 0.0 for dark respiration.\n"
        "- Keep indoor temperature within [temp_min, temp_max].\n"
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
    js['explanation'] = [str(x) for x in js.get('explanation', [])]
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

    def decide(self, obs: Dict, physics_ctx: Dict, forecast: Dict) -> Dict[str, Any]:
        prompt = _build_prompt(obs, physics_ctx, forecast, self.policy.__dict__)
        raw = ""
        try:
            raw = call_llm(prompt)
            decision = _validate_and_parse(raw, self.limits)
            self._write_log({
                "model": DEFAULT_MODEL,
                "observations": obs if self.logger.log_full_context else None,
                "physics_context": physics_ctx if self.logger.log_full_context else None,
                "forecast": forecast if self.logger.log_full_context else None,
                "prompt": prompt,
                "raw_response": raw,
                "decision": decision,
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
