install:
	@pip install -e .
	@uv run pre-commit install --install-hooks


test:
	@uv run pre-commit run --all-files


run:
	uv sync --all-extras
	uv run python -m src.key_macro.app


clean:
	rm -rf .venv


run_adk:
	uv run adk run src/first_adk

web_adk:
	uv run adk web --port 8000 src/first_sdk
