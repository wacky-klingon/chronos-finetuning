# tachyon-model

Config-driven downloader for Chronos base models.

## Requirements

- Python 3.10 to 3.12
- Poetry

## Setup

```bash
poetry lock
poetry install
```

## Project Structure

- `src/tachyon_model_downloader/`: package source
- `config/`: YAML configuration files
- `scripts/`: helper shell scripts

## Notes

This README exists so packaging metadata in `pyproject.toml` can resolve
`readme = "README.md"` during `poetry install`.
