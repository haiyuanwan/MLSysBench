from mlsysbench.simai_bench.cli import main
from mlsysbench.simai_bench.io import ConfigError


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from exc
