"""Privileged label modules for observability audits (not training data)."""

from embodied_grasp_insertion.labels.privileged_schema import (
    EXCLUDED_FIELDS,
    PROTOCOL,
    SCHEMA_VERSION,
    VELOCITY_CONTRACT,
    extract_privileged_frame,
    labels_bit_digest,
    schema_document,
    validate_privileged_label,
)

__all__ = [
    "EXCLUDED_FIELDS",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "VELOCITY_CONTRACT",
    "extract_privileged_frame",
    "labels_bit_digest",
    "schema_document",
    "validate_privileged_label",
]
