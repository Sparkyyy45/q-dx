"""
Ayushman Bharat Health Account (ABHA ID) verification and validation.
Standard format: 14-digit numeric identifier formatted as 'XX-XXXX-XXXX-XXXX'
Governed by National Health Authority (NHA) specification.
"""

from __future__ import annotations

import random
import re
from typing import Optional, Tuple


def _luhn_checksum(digits: str) -> bool:
    """Validate standard Luhn mod-10 checksum."""
    digits_list = [int(d) for d in digits if d.isdigit()]
    if not digits_list:
        return False
    check = 0
    reverse_digits = digits_list[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            check += (doubled - 9) if doubled > 9 else doubled
        else:
            check += d
    return (check % 10) == 0


def validate_abha_id(abha_id: Optional[str]) -> Tuple[bool, str, Optional[str]]:
    """
    Validate ABHA ID format and digit integrity.
    Returns: (is_valid, error_message_or_success, cleaned_formatted_id)
    """
    if not abha_id:
        return False, "ABHA ID is empty.", None

    raw = re.sub(r"[\s-]", "", str(abha_id).strip())

    if not raw.isdigit():
        return False, "ABHA ID must contain only digits.", None

    if len(raw) != 14:
        return False, f"ABHA ID must be exactly 14 digits (provided {len(raw)} digits).", None

    # First two digits represent state / series (cannot be 00)
    if raw.startswith("00"):
        return False, "ABHA ID cannot begin with '00'.", None

    # Check Luhn checksum for official format validity
    if not _luhn_checksum(raw):
        return False, "ABHA ID failed mathematical Luhn checksum validation.", None

    formatted = f"{raw[:2]}-{raw[2:6]}-{raw[6:10]}-{raw[10:14]}"
    return True, "Valid ABHA ID.", formatted


def generate_demo_abha(state_code: str = "91") -> str:
    """Generate a mathematically valid 14-digit ABHA ID with verified Luhn checksum for testing."""
    prefix = f"{state_code}{random.randint(1000000000, 9999999999):011d}"[:13]
    digits_list = [int(d) for d in prefix]
    
    # Calculate required check digit
    check = 0
    # The check digit will be at index 0 in reverse_digits, so prefix digits will be at indices 1..13
    for i, d in enumerate(digits_list[::-1], start=1):
        if i % 2 == 1:
            doubled = d * 2
            check += (doubled - 9) if doubled > 9 else doubled
        else:
            check += d

    check_digit = (10 - (check % 10)) % 10
    raw_14 = f"{prefix}{check_digit}"
    return f"{raw_14[:2]}-{raw_14[2:6]}-{raw_14[6:10]}-{raw_14[10:14]}"
