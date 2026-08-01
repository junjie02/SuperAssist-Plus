from __future__ import annotations

from langchain_core.tools import BaseTool

from superassist.tools.basic import echo
from superassist.tools.files import delete_path, list_files, read_file, write_file
from superassist.tools.images import image_search, inspect_image, present_images
from superassist.tools.shell import shell as shell_tool
from superassist.tools.task import make_task_tool, task
from superassist.tools.team import team_task
from superassist.tools.web import web_fetch, web_search


def default_tools(
    include_task: bool = True,
    include_team_task: bool = False,
    include_images: bool = True,
    run_event_reporter=None,
) -> list[BaseTool]:
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
    if include_images:
        tools.extend([image_search, inspect_image, present_images])
    if include_task:
        tools.append(make_task_tool(run_event_reporter) if run_event_reporter is not None else task)
    if include_team_task:
        tools.append(team_task)
    return tools
