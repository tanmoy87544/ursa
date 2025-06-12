import os

# from langchain_core.runnables.graph import MermaidDrawMethod
import subprocess
from typing import Annotated, Literal

import coolname
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    # TavilySearchResults,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from typing_extensions import TypedDict

from ..prompt_library.execution_prompts import executor_prompt, summarize_prompt
from .base import BaseAgent

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ExecutionState(TypedDict):
    messages: Annotated[list, add_messages]
    current_progress: str
    code_files: list[str]
    workspace: str


class ExecutionAgent(BaseAgent):
    def __init__(
        self, llm: str | BaseChatModel = "openai/gpt-4o-mini", **kwargs
    ):
        super().__init__(llm, **kwargs)
        self.executor_prompt = executor_prompt
        self.summarize_prompt = summarize_prompt
        self.tools = [run_cmd, write_code, search_tool]
        self.tool_node = ToolNode(self.tools)
        self.llm = self.llm.bind_tools(self.tools)

        self._initialize_agent()

    # Define the function that calls the model
    def query_executor(self, state: ExecutionState) -> ExecutionState:
        new_state = state.copy()
        if "workspace" not in new_state.keys():
            new_state["workspace"] = coolname.generate_slug(2)
            print(
                f"{RED}Creating the folder {BLUE}{BOLD}{new_state['workspace']}{RESET}{RED} for this project.{RESET}"
            )
        os.makedirs(new_state["workspace"], exist_ok=True)

        messages = state["messages"]
        if type(new_state["messages"][0]) == SystemMessage:
            new_state["messages"][0] = SystemMessage(
                content=self.executor_prompt
            )
        else:
            new_state["messages"] = [
                SystemMessage(content=self.executor_prompt)
            ] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response], "workspace": new_state["workspace"]}

    # Define the function that calls the model
    def summarize(self, state: ExecutionState) -> ExecutionState:
        messages = [SystemMessage(content=summarize_prompt)] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response.content]}

    # Define the function that calls the model
    def safety_check(self, state: ExecutionState) -> ExecutionState:
        new_state = state.copy()
        if state["messages"][-1].tool_calls[0]["name"] == "run_cmd":
            query = state["messages"][-1].tool_calls[0]["args"]["query"]
            safety_check = self.llm.invoke(
                "Assume commands to run/install python and Julia files are safe because the files are from a trusted source. Answer only either [YES] or [NO]. Is this command safe to run: "
                + query
            )
            if "[NO]" in safety_check.content:
                print(f"{RED}{BOLD} [WARNING] {RESET}")
                print(
                    f"{RED}{BOLD} [WARNING] That command deemed unsafe and cannot be run: {RESET}",
                    query,
                    " --- ",
                    safety_check,
                )
                print(f"{RED}{BOLD} [WARNING] {RESET}")
                return {
                    "messages": [
                        ToolMessage(
                            content="[UNSAFE] That command deemed unsafe and cannot be run: "
                            + query,
                            tool_call_id=state["messages"][-1].tool_calls[0][
                                "id"
                            ],
                        )
                    ]
                }

            print(f"{GREEN}[PASSED] the safety check: {RESET}" + query)
        elif state["messages"][-1].tool_calls[0]["name"] == "write_code":
            fn = (
                state["messages"][-1]
                .tool_calls[0]["args"]
                .get("filename", None)
            )
            if "code_files" in new_state:
                new_state["code_files"].append(fn)
            else:
                new_state["code_files"] = [fn]

        return new_state

    def _initialize_agent(self):
        self.graph = StateGraph(ExecutionState)

        self.graph.add_node("agent", self.query_executor)
        self.graph.add_node("action", self.tool_node)
        self.graph.add_node("summarize", self.summarize)
        self.graph.add_node("safety_check", self.safety_check)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.graph.add_edge(START, "agent")

        self.graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "safety_check",
                "summarize": "summarize",
            },
        )

        self.graph.add_conditional_edges(
            "safety_check",
            command_safe,
            {
                "safe": "action",
                "unsafe": "agent",
            },
        )

        self.graph.add_edge("action", "agent")
        self.graph.add_edge("summarize", END)

        self.action = self.graph.compile()
        # self.action.get_graph().draw_mermaid_png(output_file_path="execution_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)
    
    def run(self, prompt, recursion_limit = 1000):
        inputs = {
            "messages": [HumanMessage(content=prompt)]
        }
        return self.action.invoke(inputs, {"recursion_limit": recursion_limit})



@tool
def run_cmd(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Run a commandline command from using the subprocess package in python

    Args:
        query: commandline command to be run as a string given to the subprocess.run command.
    """
    workspace_dir = state["workspace"]
    print("RUNNING: ", query)
    try:
        result = subprocess.run(query,
                                text = True,
                                shell=True,
                                timeout=60000,
                                capture_output=True,
                                cwd=workspace_dir)
        stdout, stderr = result.stdout, result.stderr
    except KeyboardInterrupt:
        print("Keyboard Interrupt of command: ", query)
        stdout, stderr = "", "KeyboardInterrupt:"
    
    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    return f"STDOUT: {stdout} and STDERR: {stderr}"


@tool
def write_code(code: str, filename: str, state: Annotated[dict, InjectedState]):
    """
    Writes python or Julia code to a file in the given workspace as requested.

    Args:
        code: The code to write
        filename: the filename with an appropriate extension for programming language (.py for python, .jl for Julia, etc.)

    Returns:
        Execution results
    """
    workspace_dir = state["workspace"]
    print("Writing filename ", filename)
    try:
        # Extract code if wrapped in markdown code blocks
        if "```" in code:
            code_parts = code.split("```")
            if len(code_parts) >= 3:
                # Extract the actual code
                if "\n" in code_parts[1]:
                    code = "\n".join(code_parts[1].strip().split("\n")[1:])
                else:
                    code = code_parts[2].strip()

        # Write code to a file
        code_file = os.path.join(workspace_dir, filename)

        with open(code_file, "w") as f:
            f.write(code)
        print(f"Written code to file: {code_file}")

        return f"File {filename} written successfully."

    except Exception as e:
        print(f"Error generating code: {str(e)}")
        # Return minimal code that prints the error
        return f"Failed to write {filename} successfully."


search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(max_results=10, search_depth="advanced", include_answer=True)


# Define the function that determines whether to continue or not
def should_continue(state: ExecutionState) -> Literal["summarize", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "summarize"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that determines whether to continue or not
def command_safe(state: ExecutionState) -> Literal["safe", "unsafe"]:
    messages = state["messages"]
    last_message = messages[-1]
    if "[UNSAFE]" in last_message.content:
        return "unsafe"
    else:
        return "safe"


def main():
    execution_agent = ExecutionAgent()
    problem_string = (
        "Write and execute a python script to print the first 10 integers."
    )
    inputs = {
        "messages": [HumanMessage(content=problem_string)]
    }  # , "workspace":"dummy_test"}
    result = execution_agent.action.invoke(inputs)
    print(result["messages"][-1].content)
    return result


if __name__ == "__main__":
    main()
