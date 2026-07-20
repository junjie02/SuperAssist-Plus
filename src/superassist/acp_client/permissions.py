"""Build ACP RequestPermissionResponse objects under a configurable policy."""

from __future__ import annotations

from enum import Enum
from typing import Any


class PermissionPolicy(str, Enum):
    """How to respond to permission prompts from the ACP agent."""

    AUTO_APPROVE = "auto_approve"
    DENY = "deny"


def build_permission_response(options: list[Any], *, policy: PermissionPolicy) -> Any:
    """Resolve a permission prompt according to *policy*.

    AUTO_APPROVE picks the first option whose ``kind`` is ``allow_once`` or
    ``allow_always``. DENY (and any unmatched AUTO_APPROVE prompt) returns a
    cancelled outcome so the agent receives a denial.
    """

    from acp import RequestPermissionResponse
    from acp.schema import AllowedOutcome, DeniedOutcome

    if policy == PermissionPolicy.AUTO_APPROVE:
        for preferred_kind in ("allow_once", "allow_always"):
            for option in options:
                if getattr(option, "kind", None) != preferred_kind:
                    continue
                option_id = getattr(option, "option_id", None) or getattr(option, "optionId", None)
                if option_id is not None:
                    return RequestPermissionResponse(
                        outcome=AllowedOutcome(outcome="selected", optionId=option_id),
                    )
    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


__all__ = ["PermissionPolicy", "build_permission_response"]
