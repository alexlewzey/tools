install:
	@poetry install
	@poetry run pre-commit install
	@pip install -e .


test:
	@echo "Running tests"
	@poetry run pre-commit run --all-files

clean:
	@echo "Cleaning project"
	@find . -type d \
		\( -name '.venv' -o \
		-name 'tmp' -o \
		-name 'data' -o \
		-name '.*_cache' -o \
		-name '__pycache__' \) \
		-exec rm -rf {} + \
		2>/dev/null || true
