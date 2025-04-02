from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI
from langchain_core.messages      import HumanMessage


from reflection import PlanningState
from reflection import graph as planning_graph
from execute    import app   as execution_graph


# llm = ChatOllama(
#     model       = "gemma3:12b",
#     # model       = "phi4",
#     max_tokens  = 4000,
#     timeout     = None,
#     max_retries = 2
# )

llm = ChatOpenAI(
    model       = "o1",
    max_tokens  = 4000,
    timeout     = None,
    max_retries = 2
)

config = {"configurable": {"thread_id": "1"}}

# for event in graph.stream({"messages": [HumanMessage( content="Find a city with as least 10 vowels in its name."),],},config):
#     print(event)
#     print("---")

def run_planning_graph(instructions):
    res = planning_graph.invoke({"messages": [HumanMessage( content=instructions),],},config)
    print(res)
    return planning_graph.invoke({"messages": [HumanMessage( content=instructions),],},config)

def run_execution_graph(state:PlanningState) -> PlanningState:
    state["messages"].append(HumanMessage( content="Use search, write code, and execute code to carry out this plan."))
    res = execution_graph.invoke(state,config)
    print("#"*30)
    print(res.content)
    print("$"*30)
    return res

from langgraph.graph         import END, StateGraph, START
 
workflow = StateGraph(PlanningState)

workflow.add_node("planning", planning_graph)
workflow.add_node("action", run_execution_graph)


workflow.add_edge(START, "planning")
workflow.add_edge("planning", "action")
workflow.add_edge("action", END)

app = workflow.compile()

problem_string = "Find a city with as least 10 vowels in its name."
# problem_string = '''
# Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "log Yield".

# Write and execute a python file to:
#   - Load that data into python.
#   - Split the data into a training and test set.
#   - Fit a Gaussian process model with gpytorch to the training data where "log Yield" is the output and the other variables are inputs.
#   - Assess the quality of fit by r-squared on the test set and iterate if the current model is not good enough.
# '''

final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],"reflection_steps":3})
