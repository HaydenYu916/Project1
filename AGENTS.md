# Repository Guidelines

## Project Structure & Module Organization
- Root modules: `LED_MPPI_Controller/`, `Shelly/`, `Sensor/`, plus legacy `Collect_SP_Solar_PWM/` scripts.
- Code lives under each module’s `src/`; tests and runnable demos are under `tests/` and `examples/`.
- Cross‑module imports rely on relative paths used in tests (e.g., `LED_MPPI_Controller/tests/test_system.py` inserts `Sensor/riotee_sensor` and `Shelly/src` to `sys.path`). Avoid moving these directories without updating tests.

## Build, Test, and Development Commands
- LED MPPI controller
  - `cd LED_MPPI_Controller`
  - Run model tests: `python tests/test_all_models.py`
  - System test: `python tests/test_system.py`
  - Demos: `python examples/ppfd_demo.py`
- Shelly device control
  - `cd Shelly`
  - One‑off RPC: `python src/shelly_controller.py Red on`
  - Scheduler (foreground): `python tests/pwm_scheduler.py`
  - Service mode: `python tests/pwm_service.py start`
- Sensor (Riotee)
  - `cd Sensor/riotee_sensor`
  - Start manager: `python riotee_system_manager.py start`
  - Status: `python riotee_system_manager.py status`

## Coding Style & Naming Conventions
- Python, PEP 8 with 4‑space indentation; UTF‑8 source.
- Names: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Prefer explicit imports; keep module boundaries stable (don’t rename `src/` or test entry points).
- Add brief docstrings and type hints for public functions.

## Testing Guidelines
- Tests are executable scripts in each module’s `tests/` and may be `pytest`‑discoverable (`test_*.py`).
- Keep tests fast and deterministic; mock networked Shelly calls in unit tests.
- Place fixtures/test data under `tests/` (e.g., `Shelly/tests/src/data/`).
- From a module root, run: `python -m pytest -q` (if pytest installed) or execute individual test files as above.

## Commit & Pull Request Guidelines
- Use Conventional Commits where possible: `feat:`, `fix:`, `refactor:`, etc. Examples in history: `feat: 添加MPPI控制循环日志记录功能`.
- Commits: present‑tense, focused, include scope when helpful (e.g., `fix(shelly): clamp brightness`).
- PRs: clear description, linked issues, steps to test, and sample logs/output (e.g., MPPI result and RPC payload). Include screenshots only when UI/plots are relevant.

## Security & Configuration Tips
- Device IPs live in `Shelly/config/device_config.py`; avoid committing secrets or credentials.
- Don’t commit generated logs, `.pid`, or large data; prefer `logs/` and `data/` in gitignore.
- Networked commands may affect real devices—use examples/dry‑runs in development environments.
