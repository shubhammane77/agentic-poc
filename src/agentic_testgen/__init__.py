"""Agentic test generation platform for GitLab Maven repositories."""

from __future__ import annotations


def _bootstrap_system_certs() -> None:
    """Best-effort trust-store bootstrap before any network clients are used."""
    try:
        import pip_system_certs.wrapt_requests as wrapt_requests  # type: ignore
    except Exception:
        return
    try:
        wrapt_requests.inject_truststore()
    except Exception:
        return


_bootstrap_system_certs()

__all__ = ["__version__"]

__version__ = "0.1.0"
