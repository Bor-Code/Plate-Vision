# Contributing to Plate-Vision

## Engineering Standards and Strict Constraints

Every contribution to Plate-Vision must adhere to the following non-negotiable standards:

1. **No Emojis**: Do not include emojis anywhere in source code, configuration files, commit messages, pull request titles or descriptions, logs, or documentation.
2. **Self-Documenting Code Without Comments or Docstrings**: Do not use `#` comments or `"""` / `'''` docstrings in Python, YAML, Dockerfile, Makefile, TOML, or GitHub Actions workflow files. Express intent through descriptive names, small functions, and strict static type annotations. Architectural and usage explanations belong in Markdown files under `docs/`.
3. **Professional English**: All identifiers, commit messages, pull request descriptions, logs, error messages, and documentation must be written in professional English.
4. **Conventional Commits**: All commit messages must follow Conventional Commits (`feat`, `fix`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`, `perf`).
5. **Pull Request Workflow**:
   - Branch from up-to-date `main` using `<type>/<short-description>` naming.
   - Keep each pull request scoped to a single roadmap item or fix.
   - Never commit directly to `main`.
6. **Empirical Metrics Only**: Never fabricate metrics, benchmark numbers, or dataset statistics. Only report values produced by executing the code.

## Local Verification Commands

Run all quality checks locally before pushing:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Total test coverage must remain at or above `90%`.
