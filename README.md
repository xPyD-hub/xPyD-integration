# xPyD Integration Tests

Cross-project integration and stress tests for the xPyD ecosystem.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
# E2E tests
pytest tests/e2e/ -v

# Stress tests
pytest tests/stress/ -v -m stress
```

## Trigger

This repo's CI is triggered by:
- Push to main (nightly)
- `repository_dispatch` from sub-projects (on PR merge or PR check)
