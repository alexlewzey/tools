install:
	@pip install -e .
	@python tools/aliases/install.py
	@uv run pre-commit install --install-hooks


test:
	@uv run pre-commit run --all-files
