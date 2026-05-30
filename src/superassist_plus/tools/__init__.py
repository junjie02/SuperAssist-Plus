from __future__ import annotations

from langchain_core.tools import BaseTool

from superassist_plus.tools.basic import echo
from superassist_plus.tools.files import delete_path, list_files, read_file, write_file
from superassist_plus.tools.shell import shell as shell_tool
from superassist_plus.tools.task import make_task_tool, task
from superassist_plus.tools.team import team_task
from superassist_plus.tools.web import web_fetch, web_search


def default_tools(include_task: bool = True, include_team_task: bool = False, run_event_reporter=None) -> list[BaseTool]:
    tools = [
        echo,
        list_files,
        read_file,
        write_file,
        delete_path,
        web_search,
        web_fetch,
        shell_tool,
    ]
    if include_task:
        tools.append(make_task_tool(run_event_reporter) if run_event_reporter is not None else task)
    if include_team_task:
        tools.append(team_task)
    return tools
