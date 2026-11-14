.PHONY: sync install-hooks lint format format-check typecheck test check clean

sync:
	uv sync --dev

install-hooks:
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check: lint format-check typecheck test

clean:
	uv run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.mypy_cache', '.pytest_cache', '.ruff_cache', 'htmlcov', 'dist', 'build')]; [p.unlink(missing_ok=True) for p in [pathlib.Path('.coverage'), pathlib.Path('coverage.xml')]]"
