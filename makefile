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
