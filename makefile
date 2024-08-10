install:
	@pip install -e .
	@python src/aliases/install.py


test:
	@poetry run pre-commit run --all-files
