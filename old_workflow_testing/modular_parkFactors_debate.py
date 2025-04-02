from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI
from langchain_core.messages      import HumanMessage


from reflection import PlanningState
from reflection import graph as planning_graph
from execute    import app   as execution_graph
from agent_debate import build_agent_graph, AgentState


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

agent_graph = build_agent_graph()

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
    print("#"*30)
    print(res["messages"][-1].content)
    print("$"*30)
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
You have access to the pybaseball package for downloading baseball data and the numpyro package for Bayesian inference in python.

Write a python file to:
  - Download data on batters from 2021-2024
  - Build infer a partially pooled Bayesian hierarchical model for to estimate home run park factors
  - Visualize the posterior distribitons and summarize results
'''

final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],})

# Below is a concise summary of the multi-iteration process. Each iteration refines a Python script for estimating a “park factor” using MLB hitting data. Key steps are outlined, along with final code features and limitations:

# • Iteration 1:  
#   – Initial Poisson regression in NumPyro.  
#   – Used “Team” as a stand-in for “Park” and “qual=1” which filtered out many partial-season players.  
#   – Single MCMC chain with minimal diagnostics.  
#   – Critiqued for over-simplicity (Poisson often underestimates overdispersion), ignoring home/away splits, minimal data checks, etc.

# • Iteration 2:  
#   – Negative Binomial instead of Poisson to address overdispersion.  
#   – Removed “qual=1,” added multiple chains and more thorough MCMC checks.  
#   – Still used team-level IDs, not actual stadium IDs, and did not differentiate home vs. away.  
#   – Critiques mentioned zero-inflation, partial-season stints aggregation, no robust stadium-level modeling.

# • Iteration 3:  
#   – Switched to plate appearances (PA) as exposure.  
#   – Aggregated data at (Name, Team, Year) and permitted partial-season splits.  
#   – Checked zero-inflation explicitly, added a minimum PA filter.  
#   – Overdispersion partially pooled across teams.  
#   – Still conflated team with park and did not separate home/away.

# • Final Code:  
#   – Implements a hierarchical Gamma–Poisson approach (equivalent to a Negative Binomial) supporting team-level intercepts and overdispersion.  
#   – Loads data for 2021–2024 from pybaseball, validates columns, excludes stints with <10 PA.  
#   – Uses jax/numpyro for MCMC (NUTS), summarizes posterior draws, and plots exponentiated park effects.  
#   – Emphasizes the caveat that “Team != Park” and future enhancements (e.g., real park IDs, home/away splits, zero-inflation, robust MCMC diagnostics).