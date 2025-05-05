from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from oppenai.agents import ExecutionAgent, PlanningAgent

problem = """
Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "log Yield".

Write and execute a python file to:
  - Load that data into python.
  - Split the data into a training and test set.
  - Fit a Gaussian process model with gpytorch to the training data where "log Yield" is the output and the other variables are inputs.
  - Assess the quality of fit by r-squared on the test set and iterate if the current model is not good enough.
"""


def main():
    """
    Run an example where a planning agent generates a multistep plan and the execution agent is
    queried to solve the problem step by step.
    """
    try:
        model = ChatOpenAI(
            model="o3-mini", max_tokens=50000, timeout=None, max_retries=2
        )
        # model = ChatOllama(
        #     model       = "llama3.1:8b",
        #     max_tokens  = 4000,
        #     timeout     = None,
        #     max_retries = 2
        # )

        init = {"messages": [HumanMessage(content=problem)]}

        print(f"\nSolving problem: {problem}\n")

        # Initialize the agent
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)

        # Solve the problem
        planning_output = planner.action.invoke(init)
        print(planning_output["messages"][-1].content)
        last_step_string = "Beginning step 1 of the plan. "
        for x in planning_output["plan_steps"]:
            plan_string = str(x)
            final_results = executor.action.invoke(
                {
                    "messages": [
                        HumanMessage(content=last_step_string + plan_string)
                    ],
                    "workspace": "workspace_stepByStep_Surrogate",
                }
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
