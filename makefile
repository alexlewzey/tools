install:
	uv sync --all-groups
	@uv run pre-commit install --install-hooks

test:
	@uv run pre-commit run --all-files

run:
	uv run python -m src.key_macro.app

clean:
	rm -rf .venv

adk_run:
	uv run adk run src/adk

adk_web:
	uv run adk web src/adk
	open http://localhost:8000/

grpc_compile:
	uv run python -m grpc_tools.protoc -Isrc/grpc --python_out=src/grpc --grpc_python_out=src/grpc src/grpc/inference.proto
