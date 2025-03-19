import os
from typing import Annotated
from typing import Literal

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.tools    import tool
from langgraph.prebuilt      import ToolNode

from langchain_openai        import ChatOpenAI
from langgraph.graph         import END, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_ollama.chat_models   import ChatOllama
from langchain_community.tools      import DuckDuckGoSearchResults
from langchain_community.tools      import TavilySearchResults

import subprocess

# Add messages essentially does this with more
# robust handling
# def add_messages(left: list, right: list):
#     return left + right

workspace_dir = "./workspace/"
os.makedirs(workspace_dir, exist_ok=True)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define the function that determines whether to continue or not
def should_continue(state: State) -> Literal["summarize", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "summarize"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state: State):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


summarize_prompt = '''
You are a summarizing agent.  You'll be provided a collection of user/assistant back
and forth messages as the we are working through a complex problem requiring multiple steps.
You are to take the text and summarize it to condense the amount of text while keeping salient
points.
'''
# Define the function that calls the model
def call_summarize(state: State):
        
    messages = [SystemMessage(content=summarize_prompt)] + state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response.content]}



@tool
def run_cmd(query: str) -> str:
    """Run command from commandline"""
    safety_check = model.invoke("Assume commands to run python are safe because the files are from a trusted source. Answer only either [YES] or [NO]. Is this command safe to run: " + query)
    if "[NO]" in safety_check.content:
        print("[WARNING]")
        print("[WARNING] That command deemed unsafe and cannot be run: ", query, " --- ",safety_check)
        print("[WARNING]")
        return ["[WARNING] That command deemed unsafe and cannot be run."]
    
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
def write_python(code: str, filename: str):
    """
    Writes python code to a file in the given workspace.
    
    Args:
        code: The python code to write
        filename: the filename with a .py extension for the file name
        
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

#  search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
search_tool = TavilySearchResults(max_results=10, search_depth="advanced",include_answer=True)

tools     = [run_cmd, write_python, search_tool]
tool_node = ToolNode(tools)

model = ChatOpenAI(
    model       = "o1",
    max_tokens  = 10000,
    timeout     = None,
    max_retries = 2
)

# model = ChatOllama(
#     model       = "llama3.1:8b",
#     max_tokens  = 4000,
#     timeout     = None,
#     max_retries = 2
# )

model = model.bind_tools(tools)


workflow = StateGraph(State)

workflow.add_node("agent",         call_model)
workflow.add_node("action",         tool_node)
workflow.add_node("summarize", call_summarize)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue":     "action",
        "summarize": "summarize",
    },
)

workflow.add_edge("action", "agent")
workflow.add_edge("summarize", END)

app = workflow.compile()


def main():
    # problem_string = '''
    # Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "log Yield".

    # Write and execute a python file to:
    # - Load that data into python.
    # - Split the data into a training and test set.
    # - Fit a neural network to the training data where "log Yield" is the output and the other variables are inputs.
    # - Assess the quality of fit by r-squared on the test set and iterate if the current model is not good enough.
    # '''
    problem_string = "Write and execute a python script to print the first 10 integers." 
    # inputs = {"messages": [HumanMessage(content="what is the weather in New Mexico?")]}
    inputs = {"messages": [HumanMessage(content=problem_string)]}
    result = app.invoke(inputs)
    print(result["messages"][-1].content)
    return result

if __name__ == "__main__":
    main()
