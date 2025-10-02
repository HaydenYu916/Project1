# Repository Guidelines

## Project Structure & Module Organization
The repository is split into domain modules: `LED_MPPI_Controller/`, `Shelly/`, `Sensor/`, and the legacy toolkit in `Collect_SP_Solar_PWM/`. Within each module, production code stays in `src/`, executable demos and utilities live in `examples/`, and automated checks live in `tests/`. Some tests (for example `LED_MPPI_Controller/tests/test_system.py`) adjust `sys.path` to reach both `Sensor/riotee_sensor` and `Shelly/src`, so keep these directories stable or update the path logic alongside any move.

## Build, Test, and Development Commands
Run module workflows from their roots. LED MPPI controller: `cd LED_MPPI_Controller` then `python tests/test_all_models.py` for model sanity checks, `python tests/test_system.py` for end-to-end validation, and `python examples/ppfd_demo.py` to inspect the PPFD demo. Shelly device control: `cd Shelly` then `python src/shelly_controller.py Red on` for a direct RPC, `python tests/pwm_scheduler.py` to exercise the scheduler loop, and `python tests/pwm_service.py start` for service mode. Sensor manager: `cd Sensor/riotee_sensor` and use `python riotee_system_manager.py start` or `status` to control the Riotee stack.

## Coding Style & Naming Conventions
All Python follows PEP 8, 4-space indentation, UTF-8 source, and explicit imports. Modules use `snake_case.py`, classes use `CamelCase`, and functions or variables remain `snake_case`. Add concise docstrings and type hints on public APIs, and keep module boundaries intact when introducing new helpers.

## Testing Guidelines
Favor fast, deterministic tests. Place fixtures under the module `tests/` tree (for example `Shelly/tests/src/data/`). Use `python -m pytest -q` from the module root when pytest is available, or call the individual `tests/*.py` scripts directly. Mock Shelly network interactions in unit tests to avoid hitting physical devices.

## Commit & Pull Request Guidelines
Adopt Conventional Commits such as `feat:`, `fix:`, or `refactor:` and keep messages present tense with optional scopes (e.g. `fix(shelly): clamp brightness`). Pull requests should link issues, summarise behaviour changes, list validation steps, and include relevant logs or RPC payloads; screenshots are only necessary for plots or visual assets.

## Security & Configuration Tips
Sensitive device configuration stays in `Shelly/config/device_config.py`; never hardcode credentials in tests or examples. Ignore generated logs, `.pid` files, and bulky data by placing them under the existing gitignore. When experimenting with networked routines, prefer dry-run commands or isolated demos to avoid unintentionally driving production hardware.
