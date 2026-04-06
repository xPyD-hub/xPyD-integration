# Contributing to xPyD-integration

## Adding Tests

Place tests in the appropriate directory based on topology × depth:

| | Basic | Advanced | Stress |
|---|---|---|---|
| Single node | `single_basic/` | `single_advanced/` | `single_concurrent_stress/` |
| 1P+1D | `1p1d_basic/` | `1p1d_advanced/` | `1p1d_concurrent_stress/` |
| Multi-PD | `xpyd_basic/` | `xpyd_advanced/` | `xpyd_concurrent_stress/` |

## Running Locally

```bash
pip install -e ".[dev]"
pytest xpyd_integration/ -v --timeout=120
```

## Bot Development

See [bot/](bot/) for automated development policies.
