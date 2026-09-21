"""Optional, loopback-only browser interface (install ``hmsm[web]``)."""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Local HMSM roll player")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--static-dir", type=Path, help="Vite build directory (web/dist)"
    )
    args = parser.parse_args()
    try:
        import uvicorn
        from .server import create_app
    except ImportError as exc:
        raise SystemExit(
            'Install the browser extras with: pip install -e ".[web]"'
        ) from exc
    uvicorn.run(create_app(args.static_dir), host="127.0.0.1", port=args.port)
