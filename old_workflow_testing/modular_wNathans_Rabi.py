from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI
from langchain_core.messages      import HumanMessage


from reflection   import PlanningState
from reflection   import graph as planning_graph
from execute      import app   as execution_graph
from agent_debate import build_agent_graph, AgentState


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
    max_tokens  = 4000,
    timeout     = None,
    max_retries = 2
)

config = {"configurable": {"thread_id": "1"}}

# for event in graph.stream({"messages": [HumanMessage( content="Find a city with as least 10 vowels in its name."),],},config):
#     print(event)
#     print("---")

def run_agent_graph(state:PlanningState) -> PlanningState:
    question = state["messages"][0].content
    max_iterations = 5
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

problem_string = '''
Design, run and visualize the effects of the counter-rotating states in the quantum Rabi model using the QuTiP
python package. Compare with the Rotating wave approximation.

Write a python file to:
  - Create a compelling example
  - Build the case in python with the QuTiP package, installing QuTiP if necessary.
  - Visualize the results.
'''

final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],}, {"recursion_limit": 500})


# Writing filename  quantum_rabi_simulation.py
# Written code to file: ./workspace/quantum_rabi_simulation.py
# RUNNING:  pip install qutip numpy matplotlib tqdm pkg_resources
# STDOUT:  Collecting qutip
#   Downloading qutip-5.0.4-cp39-cp39-macosx_11_0_arm64.whl.metadata (9.2 kB)
# Requirement already satisfied: numpy in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (1.26.4)
# Requirement already satisfied: matplotlib in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (3.8.4)
# Requirement already satisfied: tqdm in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (4.67.1)

# STDERR:  ERROR: Could not find a version that satisfies the requirement pkg_resources (from versions: none)
# ERROR: No matching distribution found for pkg_resources

# RUNNING:  python quantum_rabi_simulation.py --mode advanced --savefig
# STDOUT:  
# STDERR:  ERROR: A required module was not found. Please install 'qutip', 'numpy', 'matplotlib', and 'tqdm'.
# You can install them via: pip install qutip numpy matplotlib tqdm

# RUNNING:  pip install setuptools
# STDOUT:  Requirement already satisfied: setuptools in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (59.5.0)

# STDERR:  
# RUNNING:  pip install --upgrade pip setuptools
# STDOUT:  Requirement already satisfied: pip in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (25.0.1)
# Requirement already satisfied: setuptools in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (59.5.0)
# Collecting setuptools
#   Downloading setuptools-78.0.1-py3-none-any.whl.metadata (6.6 kB)
# Downloading setuptools-78.0.1-py3-none-any.whl (1.3 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 15.4 MB/s eta 0:00:00
# Installing collected packages: setuptools
#   Attempting uninstall: setuptools
#     Found existing installation: setuptools 59.5.0
#     Uninstalling setuptools-59.5.0:
#       Successfully uninstalled setuptools-59.5.0
# Successfully installed setuptools-78.0.1

# STDERR:  
# RUNNING:  pip install qutip numpy matplotlib tqdm
# STDOUT:  Collecting qutip
#   Using cached qutip-5.0.4-cp39-cp39-macosx_11_0_arm64.whl.metadata (9.2 kB)
# Requirement already satisfied: numpy in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (1.26.4)
# Requirement already satisfied: matplotlib in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (3.8.4)
# Requirement already satisfied: tqdm in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (4.67.1)
# Requirement already satisfied: scipy>=1.9 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from qutip) (1.13.1)
# Requirement already satisfied: packaging in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from qutip) (24.2)
# Requirement already satisfied: contourpy>=1.0.1 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (1.2.1)
# Requirement already satisfied: cycler>=0.10 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (0.12.1)
# Requirement already satisfied: fonttools>=4.22.0 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (4.56.0)
# Requirement already satisfied: kiwisolver>=1.3.1 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (1.4.7)
# Requirement already satisfied: pillow>=8 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (11.1.0)
# Requirement already satisfied: pyparsing>=2.3.1 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (3.2.1)
# Requirement already satisfied: python-dateutil>=2.7 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (2.9.0.post0)
# Requirement already satisfied: importlib-resources>=3.2.0 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from matplotlib) (6.5.2)
# Requirement already satisfied: zipp>=3.1.0 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from importlib-resources>=3.2.0->matplotlib) (3.21.0)
# Requirement already satisfied: six>=1.5 in /Users/mikegros/envs/agentic/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
# Downloading qutip-5.0.4-cp39-cp39-macosx_11_0_arm64.whl (8.1 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 13.9 MB/s eta 0:00:00
# Installing collected packages: qutip
# Successfully installed qutip-5.0.4

# STDERR:  
# RUNNING:  python quantum_rabi_simulation.py --mode advanced --savefig
# STDOUT:  
# STDERR:  ERROR: A required module was not found. Please install 'qutip', 'numpy', 'matplotlib', and 'tqdm'.
# You can install them via: pip install qutip numpy matplotlib tqdm

# RUNNING:  python -c "import pkg_resources"
# STDOUT:  
# STDERR:    File "<string>", line 1
#     "import
#            ^
# SyntaxError: EOL while scanning string literal

# Writing filename  quantum_rabi_simulation.py
# Written code to file: ./workspace/quantum_rabi_simulation.py
# RUNNING:  python quantum_rabi_simulation.py --mode advanced --savefig
# STDOUT:  
# STDERR:  INFO: Running simulation for g = 0.300
# INFO: Final <σz>: Full Rabi = -0.682, JC (RWA) = -1.000
# INFO: Final <n>:   Full Rabi = 0.920, JC (RWA) = 1.000
# INFO: Figure saved as 'quantum_rabi_simulation.png'.
# 2025-03-24 12:14:45.071 Python[37647:4366792] +[CATransaction synchronize] called within transaction
# INFO: Performing coupling strength sweep over 10 values...

# Sweeping g:   0%|          | 0/10 [00:00<?, ?it/s]
# Sweeping g:  30%|███       | 3/10 [00:00<00:00, 22.62it/s]
# Sweeping g:  60%|██████    | 6/10 [00:00<00:00, 22.37it/s]
# Sweeping g:  90%|█████████ | 9/10 [00:00<00:00, 22.14it/s]
# Sweeping g: 100%|██████████| 10/10 [00:00<00:00, 22.15it/s]
# INFO: Coupling sweep figure saved as 'coupling_sweep.png'.
# 2025-03-24 12:15:34.932 Python[37647:4366792] +[CATransaction synchronize] called within transaction
# Traceback (most recent call last):
#   File "/Users/mikegros/Projects/AIDI/lanl_scientific_agent/workflows/workspace/quantum_rabi_simulation.py", line 297, in <module>
#     main()
#   File "/Users/mikegros/Projects/AIDI/lanl_scientific_agent/workflows/workspace/quantum_rabi_simulation.py", line 294, in main
#     spectral_analysis(args, a, adag, sz, sp, sm)
#   File "/Users/mikegros/Projects/AIDI/lanl_scientific_agent/workflows/workspace/quantum_rabi_simulation.py", line 244, in spectral_analysis
#     from qutip import eigenstates
# ImportError: cannot import name 'eigenstates' from 'qutip' (/Users/mikegros/envs/agentic/lib/python3.9/site-packages/qutip/__init__.py)

# Writing filename  quantum_rabi_simulation.py
# Written code to file: ./workspace/quantum_rabi_simulation.py
# Traceback (most recent call last):
#   File "/Users/mikegros/Projects/AIDI/lanl_scientific_agent/workflows/modular_wNathans_Rabi.py", line 83, in <module>
#     final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],})
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/pregel/__init__.py", line 2336, in invoke
#     for chunk in self.stream(
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/pregel/__init__.py", line 1993, in stream
#     for _ in runner.tick(
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/pregel/runner.py", line 230, in tick
#     run_with_retry(
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/pregel/retry.py", line 40, in run_with_retry
#     return task.proc.invoke(task.input, config)
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/utils/runnable.py", line 546, in invoke
#     input = step.invoke(input, config, **kwargs)
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/utils/runnable.py", line 310, in invoke
#     ret = context.run(self.func, *args, **kwargs)
#   File "/Users/mikegros/Projects/AIDI/lanl_scientific_agent/workflows/modular_wNathans_Rabi.py", line 56, in run_execution_graph
#     res = execution_graph.invoke(state,config)
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/pregel/__init__.py", line 2336, in invoke
#     for chunk in self.stream(
#   File "/Users/mikegros/envs/agentic/lib/python3.9/site-packages/langgraph/pregel/__init__.py", line 2013, in stream
#     raise GraphRecursionError(msg)
# langgraph.errors.GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
# For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT