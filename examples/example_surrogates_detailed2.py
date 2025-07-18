from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

from ursa.agents import (
    ExecutionAgent,
    PlanningAgent,
)
from ursa.prompt_library.planning_prompts import (
    detailed_planner_prompt,
)

problem_definition = """
Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "log Yield".

Write and execute a python file to:
  - Load that data into python.
  - Split the data into a training and test set.
  - Fit a Gaussian process model with gpytorch to the training data where "log Yield" is the output and the other variables are inputs.
      - Make sure that the number of training iterations are sufficient to converge
  - Fit a Bayesian neural network with numpyro to the same data.
      - Make sure that the number of training iterations are sufficient to converge
  - Assess the quality of fits by r-squared on the test set and summarize the quality of the Gaussian process against the neural network.
  - Assess the uncertainty quantification of the two models by coverage on the test set and with visualization.
"""


def main():
    """Run a simple example of the scientific agent."""
    try:
        model = ChatLiteLLM(
            model="openai/o3-mini",
            max_tokens=20000,
            max_retries=2,
        )

        print(f"\nSolving problem: {problem_definition}\n")

        # Initialize the agent
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)

        inputs = {"messages": [HumanMessage(content=problem_definition)]}

        # Solve the problem
        planning_output = planner.action.invoke(
            inputs, {"recursion_limit": 999999}
        )
        print(planning_output["messages"][-1].content)
        last_step_string = "Beginning to break down step 1 of the plan. "
        detail_plan_string = "Flesh out the details of this step and generate substeps to handle the details."
        for x in planning_output["plan_steps"]:
            print(80 * "#")
            print(80 * "#")
            print(str(x))
            print(80 * "#")
            print(80 * "#")
            detail_planner = PlanningAgent(llm=model)
            plan_string = str(x)
            detail_planner.planner_prompt = detailed_planner_prompt
            step_prompt = f"""
                You are contributing to a larger solution aimed at {problem_definition}.
                If there are previous steps, the summary of the most recent step is: {last_step_string}.
                The current step is: {plan_string}.

                {detail_plan_string}
            """
            detail_output = detail_planner.action.invoke(
                {"messages": [HumanMessage(content=step_prompt)]},
                {"recursion_limit": 999999},
            )
            last_substep_string = "Beginning to break down of the plan. "
            for y in detail_output["plan_steps"]:
                print(80 * "$")
                print(80 * "$")
                print(str(y))
                print(80 * "$")
                print(80 * "$")
                execute_string = "Execute this step and report results for the executor of the next step. Do not use placeholders but fully carry out each step."
                substep_prompt = f"""
                    You are contributing to a larger solution aimed at {problem_definition}.
                    If there are previous steps, the summary of the most recent step is: {last_substep_string}.
                    The current step is: {str(y)}.

                    {execute_string}
                """
                final_results = executor.action.invoke(
                    {
                        "messages": [HumanMessage(content=substep_prompt)],
                        "workspace": "workspace_surrogate_detailed2",
                    },
                    {"recursion_limit": 999999},
                )
                last_substep_string = final_results["messages"][-1].content
                print(last_substep_string)
            last_step_string = last_substep_string

        return final_results["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    main()

