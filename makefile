install:
	@poetry install
	@poetry run pre-commit install
	@pip install -e .


test:
	@echo "Running tests"
	@poetry run pre-commit run --all-files

