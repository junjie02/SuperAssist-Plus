from __future__ import annotations

from langchain_core.tools import BaseTool

from superassist_plus.tools.basic import echo
from superassist_plus.tools.files import delete_path, list_files, read_file, write_file
from superassist_plus.tools.task import make_task_tool, task
from superassist_plus.tools.web import web_fetch, web_search


def default_tools(include_task: bool = True, run_event_reporter=None) -> list[BaseTool]:
    tools = [
        echo,
        list_files,
        read_file,
        write_file,
        delete_path,
        web_search,
        web_fetch,
    ]
    if include_task:
        tools.append(make_task_tool(run_event_reporter) if run_event_reporter is not None else task)
    return tools
