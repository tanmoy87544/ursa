from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from oppenai.agents import (
    ExecutionAgent,
    HypothesizerAgent,
    HypothesizerState,
    PlanningAgent,
    ResearchAgent,
)

problem_definition = """
Your goal is to map out the binary phase diagram for alloys of aluminum and nickel. 

For different mixing ratios of aluminum and nickel and at different tempertures, the alloy will be in different states.

We need to use the LAMMPS molecular dynamics model to evaluate configurations and map out the phase diagram.
Write python to:
    - Evaluate LAMMPS for different mixing ratios and temperatures
    - Use these results to generate plots of the different phases of the allow
    - Add more LAMMPS evaluations to refine the phase diagram if any regions are under-sampled.

No hypothetical examples! 
Evaluate LAMMPS as needed. You have a valid installation.

Summarize your results in a webpage with interactive visualization.
"""


def main():
    """Run a simple example of the scientific agent."""
    try:
        model_o4 = ChatOpenAI(
            model="o4-mini", max_tokens=50000, timeout=None, max_retries=2
        )
        model_o3 = ChatOpenAI(
            model="o3", max_tokens=30000, timeout=None, max_retries=2
        )
        # model = ChatOllama(
        #     model       = "llama3.1:8b",
        #     max_tokens  = 4000,
        #     timeout     = None,
        #     max_retries = 2
        # )

        print(f"\nSolving problem: {problem_definition}\n")

        # Initialize the agent
        # hypothesizer = HypothesizerAgent(llm  = model)
        planner = PlanningAgent(llm=model_o4)
        executor = ExecutionAgent(llm=model_o3)

        # Solve the problem
        initial_state = HypothesizerState(
            question=problem_definition,
            question_search_query="",
            current_iteration=0,
            max_iterations=2,
            agent1_solution=[],
            agent2_critiques=[],
            agent3_perspectives=[],
            final_solution="",
        )

        # hypothesis_results   = hypothesizer.action.invoke(initial_state)
        # Solve the problem
        planning_output = planner.action.invoke(
            {"messages": [HumanMessage(content=problem_definition)]}
        )
        print(planning_output["plan_steps"])
        last_step_string = "This is the first step."
        execute_string = """
        Execute this step and report results for the executor of the next step. 
        Do not use placeholders. Fully carry out each step.
        Only execute this step. Stay in your lane!
        """
        for x in planning_output["plan_steps"]:
            plan_string = str(x)
            step_prompt = f"""
                You are contributing to a larger solution aimed at {problem_definition}.
                If there are previous steps, the summary of the most recent step is: {last_step_string}.
                The current substep is: {plan_string}.

                {execute_string}
            """
            final_results = executor.action.invoke(
                {"messages": [HumanMessage(content=step_prompt)]},
                {"recursion_limit": 999999},
            )
            last_step_string = final_results["messages"][-1].content
            print(last_step_string)

        return final_results["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    main()

# Solving problem:
# Your goal is to map out the binary phase diagram for alloys of aluminum and nickel.

# For different mixing ratios of aluminum and nickel and at different tempertures, the alloy will be in different states.

# We need to use the LAMMPS molecular dynamics model to evaluate configurations and map out the phase diagram.
# Write python to:
#     - Evaluate LAMMPS for different mixing ratios and temperatures
#     - Use these results to generate plots of the different phases of the allow
#     - Add more LAMMPS evaluations to refine the phase diagram if any regions are under-sampled.


# No hypothetical examples!
# Evaluate LAMMPS as needed.

# Summarize your results in a webpage with interactive visualization.
