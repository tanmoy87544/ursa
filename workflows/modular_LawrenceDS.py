from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI
from langchain_core.messages      import HumanMessage


from reflection import PlanningState
from reflection import graph as planning_graph
from execute    import app   as execution_graph
from policy_strategizer import build_agent_graph, AgentState


agent_graph = build_agent_graph()

# llm = ChatOllama(
#     model       = "gemma3:12b",
#     # model       = "phi4",
#     max_tokens  = 4000,
#     timeout     = None,
#     max_retries = 2
# )

llm = ChatOpenAI(
    model       = "o1",
    max_tokens  = 10000,
    timeout     = None,
    max_retries = 2
)

config = {"configurable": {"thread_id": "1"}}

# for event in graph.stream({"messages": [HumanMessage( content="Find a city with as least 10 vowels in its name."),],},config):
#     print(event)
#     print("---")

def run_agent_graph(state:PlanningState) -> PlanningState:
    question = state["messages"][0].content
    max_iterations = 3
    # Initialize the state
    initial_state = AgentState(
        question=question,
        current_iteration=0,
        max_iterations=max_iterations,
        agent1_solution=[],
        agent2_critiques=[],
        agent3_perspectives=[],
        final_solution="",
        llm_model="o1",
    )
    # Run the graph
    result = agent_graph.invoke(initial_state, {"recursion_limit": 9999999999999})
    summary_text = result["summary_report"]
    return {"messages": [HumanMessage( content=summary_text),],}

def run_execution_graph(state:PlanningState) -> PlanningState:
    state["messages"].append(HumanMessage( content="Use search, write code, and execute code to carry out this plan."))
    res = execution_graph.invoke(state,config)
    return res

from langgraph.graph         import END, StateGraph, START
 
workflow = StateGraph(PlanningState)

workflow.add_node("planning", run_agent_graph)
workflow.add_node("action", run_execution_graph)


workflow.add_edge(START, "planning")
workflow.add_edge("planning", "action")
workflow.add_edge("action", END)

app = workflow.compile()

# problem_string = "Find a city with as least 10 vowels in its name."
problem_string = '''
Find a recent paper by Lawrence and Murph named A New Method For Multinomial Inference Using Dempster-Shafer Theory

Write a python file to:
  - Replicate the example in detail. Do not simplify. 
  - Visualize the results.
  - Summarize how your example compares to the published paper.
  - Critique the paper and suggest future work.
'''

final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],})
