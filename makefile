install:
	@pip install -e .
	@python tools/aliases/install.py
	@uv run pre-commit install --install-hooks
	# possibly use this as the installer instead
	# alias km="cd ~/repository/tools && make run"


test:
	@uv run pre-commit run --all-files


run:
	uv sync --all-extras
	uv run python -m src.key_macro.app


clean:
	rm -rf .venv
