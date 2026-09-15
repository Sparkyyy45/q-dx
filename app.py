#!/usr/bin/env python3
"""
CardioQ: Hybrid Quantum-Classical Cardiovascular Disease Diagnostic Platform.
Main application entry point for launching the clinical decision support server and UI.

Usage:
    python app.py [--host 127.0.0.1] [--port 8080] [--fastapi]
"""

from __future__ import annotations

import argparse
import sys

from config.settings import settings
from src.api.server import run_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CardioQ: Hybrid Quantum-Classical CVD Diagnostic Platform Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.HOST,
        help=f"Host address to bind server (default: {settings.HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.PORT,
        help=f"Port to run server (default: {settings.PORT})",
    )
    parser.add_argument(
        "--fastapi",
        action="store_true",
        help="Run high-throughput asynchronous FastAPI server via Uvicorn",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print(" CARDIOQ: HYBRID QUANTUM-CLASSICAL CVD PREDICTION PLATFORM")
    print(" Smart India Hackathon (SIH) Problem Statement 3 Solution")
    print("=" * 80)
    print(f" Web Dashboard URL:        http://{args.host}:{args.port}/")
    print(f" REST API Health:          http://{args.host}:{args.port}/health")
    print(f" REST API Predict:         http://{args.host}:{args.port}/api/predict (POST)")
    print(f" SQLite Records History:   http://{args.host}:{args.port}/api/records (GET)")
    print(f" ABDM ABHA Verification:   http://{args.host}:{args.port}/api/records/validate-abha (POST)")
    print(f" Qiskit Aer Hardware Sim:  http://{args.host}:{args.port}/api/quantum/qiskit/execute (POST)")
    print(f" OpenQASM Circuit Export:  http://{args.host}:{args.port}/api/quantum/circuit/qasm (GET)")
    print("=" * 80)

    if args.fastapi:
        try:
            import uvicorn
            print(f"Starting ASGI FastAPI server on http://{args.host}:{args.port}...")
            uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)
            return 0
        except ImportError:
            print("uvicorn not available, falling back to native ClinicalPlatformServer...")

    try:
        run_server(host=args.host, port=args.port)
    except Exception as exc:
        sys.stderr.write(f"Server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

