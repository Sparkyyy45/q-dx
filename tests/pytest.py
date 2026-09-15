"""
Lightweight pytest compatibility module for environments where pytest is not pre-installed.
Provides pytest.raises and fixtures compatibility.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Optional, Type


class ExceptionInfo:
    """Wrapper holding caught exception instance, mirroring pytest.ExceptionInfo."""
    def __init__(self):
        self.value: Optional[BaseException] = None


@contextmanager
def raises(expected_exception: Type[BaseException], match: Optional[str] = None):
    """Context manager imitating pytest.raises supporting 'as exc_info' and 'match'."""
    info = ExceptionInfo()
    try:
        yield info
    except expected_exception as exc:
        info.value = exc
        if match is not None:
            if not re.search(match, str(exc)):
                raise AssertionError(
                    f"Pattern '{match}' does not match exception message: '{exc}'"
                )
    except Exception as exc:
        raise AssertionError(
            f"Expected exception {expected_exception}, but got {type(exc)}: '{exc}'"
        )
    else:
        raise AssertionError(f"Did not raise expected exception: {expected_exception}")


def fixture(func):
    """Decorator imitating pytest.fixture."""
    return func
