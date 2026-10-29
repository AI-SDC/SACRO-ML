"""Reporting utilities for SACRO-ML attack reports.

This subpackage contains the canonical attack-report JSON schema, a set of
common catalog definitions, and tooling to convert legacy ``report.json``
files produced by older SACRO-ML versions into the new nested,
catalog-enriched report format.
"""

from __future__ import annotations

from sacroml.reporting.convert import (
    ConversionResult,
    convert_report,
    convert_report_file,
)

__all__ = [
    "ConversionResult",
    "convert_report",
    "convert_report_file",
]
