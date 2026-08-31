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

grpc_compile:
	uv run python -m grpc_tools.protoc -Isrc/grpc --python_out=src/grpc --grpc_python_out=src/grpc src/grpc/inference.proto
