from mlsysbench.simai_bench.cli import main
from mlsysbench.simai_bench.io import ConfigError


def entrypoint() -> None:
    """Run the CLI with consistent configuration-error rendering."""
    try:
        main()
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    entrypoint()
