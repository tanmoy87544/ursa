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

# problem_string = "Find a city with as least 10 vowels in its name."
problem_string = '''
Search for and find a paper on Determinantal Point Processes in the statistics literature and identify a synthetic data 
example from that paper. 

Write a python file to:
  - Replicate the example
  - Visualize the results
  - Summarize how your example compares to the published paper.
'''

# problem_string = '''
# Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "log Yield".

# Write and execute a python file to:
#   - Load that data into python.
#   - Split the data into a training and test set.
#   - Fit a regression model to the training data where "log Yield" is the output and the other variables are inputs.
#   - Assess the quality of fit by r-squared on the test set and iterate if the current model is not good enough.
# '''

final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],})




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Model Response:  The nonsymmetric kernel example triggers an error in the dppy library because dppy requires the kernel to be a real symmetric positive semi-definite matrix in order to sample. Nonetheless, you can still visualize the nonsymmetric matrix (and its eigenvalues) as a synthetic example, but the library’s sampling routines will fail for a nonsymmetric matrix. If you need to sample from a more general NDPP (Nonsymmetric Determinantal Point Process), a different library or custom code would be required.

# Below is a summary of what happened:

# • With the symmetric kernel option (–kernel-type symmetric), you were able to sample from the DPP and visualize the results.  
# • With the nonsymmetric option (–kernel-type nonsymmetric), the code successfully formed a skew-symmetric perturbation, diagnosed the eigenvalues, and showed a large number of them have significant imaginary parts. However, dppy then refused to sample, giving the error “array not symmetric: M.T != M.”  

# This reflects a limitation of dppy’s design: it only supports standard (Hermitian) DPP kernels. While our demonstration code helps illustrate the idea of a nonsymmetric perturbation, actually sampling from such a kernel requires either specialized research code or a different approach.
# ##################################################
# ##################################################
# SUMMARY:  Below is a concise summary of the conversation’s key points:

# • Goal: Locate a paper on Determinantal Point Processes (DPPs), particularly involving nonsymmetric kernels, and replicate a synthetic-data example.  
# • A specific arXiv paper, “On determinantal point processes with nonsymmetric kernels” (arXiv:2406.03360), is identified as inspiration.  
# • A Python script is iteratively developed:  n dpp.list_of_samples[0]\n    return dpp.list_of_samples[0]\n\ndef main():\n    # Create candidate points on a grid.\n    points = create_grid(n_points_side=20)\n\n    # Compute the kernel matrix using a Gaussian kernel.\n    sigma = 0.1  # smaller sigma accentuates repulsion.\n    L = gaussian_kernel_matrix(points, sigma=sigma)\n    \n    # Sample a set of indices from the DPP.\n    sample_indices = sample_dpp(L)\n    sampled_points = points[sample_indices]\n    \n    # Visualization.\n    plt.figure(figsize=(8, 8))\n    # Plot all candidate points in light gray.\n    plt.scatter(points[:, 0], points[:, 1], color=\'lightgray\', label=\'Candidate points\')\n    # Plot sampled points in red and larger.\n    plt.scatter(sampled_points[:, 0], sampled_points[:, 1],\n                color=\'red\', s=100, label=\'DPP sampled points\', edgecolor=\'k\')\n    plt.title("Synthetic DPP Sample\\n(Gaussian Kernel with sigma = {})".format(sigma))\n    plt.xlabel("x")\n    plt.ylabel("y")\n    plt.legend()\n    plt.grid(True)\n    plt.tight_layout()\n    plt.show()\n\n    # Summary of the result compared with the published paper:\n    # --------------------------------------------------------------------\n    # In the published paper (e.g. “On Determinantal Point Processes with Nonsymmetric Kernels”),\n    # the authors demonstrate how DPPs can generate repulsive spatial point patterns.\n    # Here, our example uses a Gaussian similarity kernel defined on a grid of candidate points.\n    # Because of the repulsion property encoded in the determinant-based probability,\n    # the DPP sample (displayed in red) tends to spread out in the unit square,\n    # similar to the synthetic data experiments in the paper.\n    # Thus, our experiment mirrors the key qualitative phenomenon showcased by DPPs in the literature.\n    # --------------------------------------------------------------------\n    \nif __name__ == \'__main__\':\n    main()\n\\end{lstlisting}\n\n\\textbf{Critique:}\\\\\n\\begin{lstlisting}\nDetailed critique pointing out strengths but also many areas where the solution can be improved.\n1. Assumptions versus the Published Paper’s Goals\n   • Mismatch between the replication experiment and the focus on nonsymmetric kernels in the paper.\n   • Discussion on why a grid-based candidate set is chosen.\n2. Code Structure and Robustness\n   • Need for more robust error handling and checks.\n   • Parameter hard-coding issues.\n3. Clarity of Comparison and Explanation\n   • Suggested quantitative measures of repulsion (e.g., pairwise distances).\n4. Choice of Synthetic Data Example\n   • Possibility of exploring additional settings, control experiments.\n5. Visualization Enhancements\n   • Further visualization improvements and annotations.\n6. Discussion and Documentation\n   • Need for inline comments and greater context regarding dppy.\n7. Connection to Literature\n   • Broader discussion on DPP applications.\n\\end{lstlisting}\n\n\\textbf{Competitor Perspective:}\\\\\n\\begin{lstlisting}\nResponse outlining how to counter the Agent 1 approach:\n1. Mismatch with the Paper’s Focus on Nonsymmetry:\n   • Highlight the importance of a nonsymmetric kernel.\n2. Parameterization and Experimental Flexibility:\n   • Introduce command-line configuration.\n3. Robust Error Handling and Sampling Validation:\n   • Rigorously check outputs from dppy.\n4. Enhanced Visualization and Quantitative Metrics:\n   • Include quantitative measures (e.g., nearest-neighbor distances).\n5. Deeper Literature Integration:\n   • Provide explicit references to figures/experiments in literature.\n6. Extensible Code Architecture:\n   • Modularize the code for flexibility and testing.\n\\end{lstlisting}\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\\subsection*{Iteration 2}\n\\textbf{Solution:}\\\\\n\\begin{lstlisting}\n#!/usr/bin/env python3\n"""\nDPP Synthetic Data Experiment Replication\n\nThis script replicates a synthetic data experiment inspired by studies\non Determinantal Point Processes (DPPs) in the statistics literature.\nTwo kernel choices are provided:\n  • symmetric: a standard Gaussian similarity kernel.\n  • nonsymmetric: a Gaussian kernel perturbed by a skew-symmetric matrix.\nThe user can adjust grid size, kernel lengthscale (sigma), and kernel type from the command line.\n\nUsage example:\n    python dpp_synthetic_enhanced.py --grid-size 20 --sigma 0.1 --kernel-type symmetric\n    python dpp_synthetic_enhanced.py --grid-size 20 --sigma 0.1 --kernel-type nonsymmetric\n"""\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport argparse\n\n# Try importing the dppy library\ntry:\n    from dppy.finite_dpps import FiniteDPP\nexcept ImportError as error:\n    raise ImportError("The dppy package is required. Install with \'pip install dppy\'") from error\n\ndef create_grid(n_points_side=20):\n    # ... function definition as before ...\n    pass\n\ndef gaussian_kernel_matrix(points, sigma=0.1):\n    # ... function definition as before ...\n    pass\n\ndef gaussian_nonsymmetric_kernel(points, sigma=0.1, noise_level=0.05, random_state=None):\n    # ... create a nonsymmetric kernel ...\n    pass\n\ndef sample_dpp(L):\n    # ... sample from the DPP using dppy ...\n    pass\n\ndef compute_nearest_neighbor_distances(points):\n    # ... compute distances ...\n    pass\n\ndef parse_arguments():\n    # ... use argparse to parse grid-size, sigma, kernel-type ...\n    pass\n\ndef main():\n    # ... main function creating grid, computing kernel, sampling DPP, plotting two subplots and histogram ...\n    pass\n\nif __name__ == \'__main__\':\n    main()\n\\end{lstlisting}\n\n\\textbf{Critique:}\\\\\n\\begin{lstlisting}\nDetailed critique noting:\n1. Introduction of command-line interface using argparse to improve parameterization.\n2. Showing both a DPP sample and a uniform random sample for comparison.\n3. Adding a histogram of nearest-neighbor distances to quantitatively
