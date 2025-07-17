import sys

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from ursa.agents import (
    ExecutionAgent,
    HypothesizerAgent,
    HypothesizerState,
    PlanningAgent,
)

problem_definition = """
High-entropy alloys have potential to develop metals that are not brittle in the cold temperatures of space.

Hypothesize high-entropy alloys and identify the mixture weights for these metals for optimal material properties:
    - Research high-entropy alloys and their properties
    - Visualize materials properties of high-entropy alloys to communicate those with high potential.
    - Use physical and data-driven models to predict optimal high-entropy alloys for space travel.

Your only tools for identifying the materials are:
    - Writing and executing python code.
    - Acquiring materials data from reputable online resources.
        - Attempt to use freely available data that does not require an API KEY
    - Installing and evaluating repuatable, openly available materials models. 

You cannot perform any materials synthesis or experimental testing.

In the end we should have a list of high-entropy alloys that are not brittle at low temperature and a justification for this.

No hypothetical examples! Obtain what you need to perform the actual analysis, execute the steps, and get come to a defensible conclusion. 

Summarize your results in a webpage with interactive visualization.
"""


def main(mode: str):
    """Run a simple example of the scientific agent."""
    try:
        model = ChatLiteLLM(
            model="openai/o3-mini"
            if mode == "prod"
            else "ollama_chat/llama3.1:8b",
            max_tokens=40000 if mode == "prod" else 4000,
            max_retries=2,
        )

        print(f"\nSolving problem: {problem_definition}\n")

        # Initialize the agent
        hypothesizer = HypothesizerAgent(llm=model)
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)

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

        hypothesis_results = hypothesizer.action.invoke(initial_state)
        # Solve the problem
        planning_output = planner.action.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=problem_definition
                        + hypothesis_results["summary_report"]
                    )
                ]
            }
        )
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
                {
                    "messages": [HumanMessage(content=step_prompt)],
                    "workspace": "workspace_materials1",
                },
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
    main(sys.argv[-1])
