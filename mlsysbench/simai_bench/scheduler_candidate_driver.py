"""JSON-lines adapter for untrusted scheduler-policy task submissions."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import resource
import sys


def _install_landlock(solution_dir: pathlib.Path) -> None:
    repository_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from mlsysbench.simai_bench.landlock import restrict_current_process

    system_paths = [
        pathlib.Path(path)
        for path in ("/bin", "/lib", "/lib64", "/usr")
        if pathlib.Path(path).exists()
    ]
    restrict_current_process(
        read_only_paths=[*system_paths, solution_dir],
        read_write_paths=[],
    )


def _install_resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def main() -> None:
    solution_dir = pathlib.Path(sys.argv[1]).resolve()
    if "--already-isolated" not in sys.argv[2:]:
        _install_landlock(solution_dir)
    _install_resource_limits()
    scheduler_path = solution_dir / "scheduler.py"
    spec = importlib.util.spec_from_file_location("candidate_scheduler", scheduler_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scheduler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schedule = getattr(module, "schedule", None)
    if not callable(schedule):
        raise RuntimeError("scheduler.py must define schedule(requests, limits)")

    for line in sys.stdin:
        try:
            request = json.loads(line)
            decisions = schedule(request["requests"], request["limits"])
            response = {"decisions": decisions}
            rendered = json.dumps(response, separators=(",", ":"))
            if len(rendered.encode("utf-8")) > 65_536:
                raise ValueError("scheduler response exceeds 65536 bytes")
        except Exception as exc:  # noqa: BLE001 - return a bounded error to the evaluator.
            rendered = json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"},
                separators=(",", ":"),
            )
        sys.stdout.write(rendered + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
