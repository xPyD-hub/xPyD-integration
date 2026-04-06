# xPyD-integration

**Integration tests for the xPyD ecosystem.**

Cross-component tests that validate proxy + sim + bench work together correctly. Organized in a 3×3 matrix: topology (single / 1P1D / multi-PD) × depth (basic / advanced / stress).

## Test Structure

```
xpyd_integration/
  single_basic/              # sim(dual) direct — basic API
  single_advanced/           # sim(dual) direct — advanced features
  single_concurrent_stress/  # sim(dual) direct — concurrency & stress

  1p1d_basic/                # 1P+1D+proxy — basic PD flow
  1p1d_advanced/             # 1P+1D+proxy — advanced features
  1p1d_concurrent_stress/    # 1P+1D+proxy — concurrency & stress

  xpyd_basic/                # NP+MD+proxy — basic multi-node
  xpyd_advanced/             # NP+MD+proxy — advanced features
  xpyd_concurrent_stress/    # NP+MD+proxy — concurrency & stress
```

## Install

```bash
pip install -e ".[dev]"
```

## Run Tests

```bash
# All tests
pytest xpyd_integration/ -v

# Single category
pytest xpyd_integration/single_basic/ -v
```

## CI

- **PR trigger**: sub-repos dispatch integration CI on every PR
- **Nightly**: cron at 2am UTC, all repos at HEAD
- **Release**: sub-repos dispatch on release
- **Result reporting**: integration CI writes pass/fail back to source PR

## License

Apache 2.0 — see [LICENSE](LICENSE)
