import os
from typing            import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.tools    import tool
from langgraph.prebuilt      import ToolNode

from langgraph.graph         import END, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_community.tools      import DuckDuckGoSearchResults
from langchain_community.tools      import TavilySearchResults
# from langchain_core.runnables.graph import MermaidDrawMethod

import subprocess

from .base                              import BaseAgent
from ..prompt_library.execution_prompts import executor_prompt, summarize_prompt

workspace_dir = "./workspace/"
os.makedirs(workspace_dir, exist_ok=True)

class ExecutionState(TypedDict):
    messages: Annotated[list, add_messages]

class ExecutionAgent(BaseAgent):
    def __init__(self, llm = "OpenAI/gpt-4o", *args, **kwargs):
        super().__init__(llm, args, kwargs)
        self.executor_prompt  = executor_prompt
        self.summarize_prompt = summarize_prompt
        self.tools            = [run_cmd, write_code, search_tool]
        self.tool_node        = ToolNode(self.tools)
        self.llm              = self.llm.bind_tools(self.tools)

        self._initialize_agent()    

    # Define the function that calls the model
    def query_executor(self, state: ExecutionState) -> ExecutionState:
        messages = state["messages"]
        if type(state["messages"][0]) == SystemMessage:
            state["messages"][0] = SystemMessage(content=self.executor_prompt)
        else:
            state["messages"]    = [SystemMessage(content=self.executor_prompt)] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    # Define the function that calls the model
    def summarize(self, state: ExecutionState) -> ExecutionState:
        messages = [SystemMessage(content=summarize_prompt)] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response.content]}

    # Define the function that calls the model
    def safety_check(self, state: ExecutionState) -> ExecutionState:
        if state["messages"][-1].tool_calls[0]["name"] == "run_cmd":
            query        = state["messages"][-1].tool_calls[0]["args"]["query"]
            safety_check = self.llm.invoke("Assume commands to run python and Julia are safe because the files are from a trusted source. Answer only either [YES] or [NO]. Is this command safe to run: " + query)
            if "[NO]" in safety_check.content:
                print("[WARNING]")
                print("[WARNING] That command deemed unsafe and cannot be run: ", query, " --- ",safety_check)
                print("[WARNING]")
                return {"messages": ["[UNSAFE] That command deemed unsafe and cannot be run: "+ query]}
            
            print("[PASSED] the safety check: "+query)
        return state

    def _initialize_agent(self):
        self.graph = StateGraph(ExecutionState)

        self.graph.add_node("agent",       self.query_executor)
        self.graph.add_node("action",           self.tool_node)
        self.graph.add_node("summarize",        self.summarize)
        self.graph.add_node("safety_check",  self.safety_check)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.graph.add_edge(START, "agent")

        self.graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue":  "safety_check",
                "summarize":    "summarize",
            },
        )

        self.graph.add_conditional_edges(
            "safety_check",
            command_safe,
            {
                "safe":   "action",
                "unsafe":  "agent",
            },
        )

        self.graph.add_edge("action",    "agent")
        self.graph.add_edge("summarize",     END)

        self.action = self.graph.compile()
        # self.action.get_graph().draw_mermaid_png(output_file_path="execution_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

@tool
def run_cmd(query: str) -> str:
    """Run command from commandline"""
    
    print("RUNNING: ", query)
    process = subprocess.Popen(
        query.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace_dir
    )

    stdout, stderr = process.communicate(timeout=600)

    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    return f"STDOUT: {stdout} and STDERR: {stderr}"

@tool
def write_code(code: str, filename: str):
    """
    Writes python or Julia code to a file in the given workspace as requested.
    
    Args:
        code: The code to write
        filename: the filename with an appropriate extension for programming language (.py for python, .jl for Julia, etc.)
        
    Returns:
        Execution results
    """
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
    problem_string = "Write and execute a python script to print the first 10 integers." 
    inputs = {"messages": [HumanMessage(content=problem_string)]}
    result = execution_agent.action.invoke(inputs)
    print(result["messages"][-1].content)
    return result

if __name__ == "__main__":
    main()
