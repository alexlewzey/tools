install:
	@pip install -e .
	@python src/aliases/install.py
	@poetry install
	@poetry run pre-commit install --install-hooks


test:
	@poetry run pre-commit run --all-files
