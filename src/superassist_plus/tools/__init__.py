from __future__ import annotations

from langchain_core.tools import BaseTool

from superassist_plus.tools.basic import current_time, echo


def default_tools() -> list[BaseTool]:
    return [current_time, echo]

