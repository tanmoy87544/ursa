import os
import sys

from langchain_community.chat_models import ChatLiteLLM

from ursa.agents import CodeReviewAgent

problem_definition = """
Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "logYield".

Write and execute a python file to:
  - Load that data into python.
  - Split the data into a training and test set.
  - Visualize the training data for EDA.
  - Fit a Gaussian process model with gpytorch to the training data where "logYield" is the output and the other variables are inputs.
      - Visualize the quality of the fit.
  - Fit a Bayesian neural network with numpyro to the same data.
      - Visualize the quality of the fit.
  - Assess the quality of fits by r-squared on the test set and summarize the quality of the Gaussian process against the neural network.
  - Assess the uncertainty quantification of the two models by coverage on the test set and with visualization.
"""


def main(mode: str):
    try:
        model_o3 = ChatLiteLLM(
            model="openai/o3-mini"
            if mode == "prod"
            else "ollama_chat/llama3.1:8b",
            max_tokens=40000 if mode == "prod" else 4000,
            max_retries=2,
        )

        code_review_agent = CodeReviewAgent(llm=model_o3)

        initial_state = {
            "messages": [],
            "project_prompt": problem_definition,
            "code_files": [
                x
                for x in os.listdir("workspace/")
                if any([y in x for y in [".py", ".html"]])
            ],
            "edited_files": [],
            "iteration": 0,
        }

        result = code_review_agent.action.invoke(
            initial_state, {"recursion_limit": 999999}
        )
        # for x in result["messages"]:
        #     print(x.content)
        print(result["edited_files"])
        return result["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    print(main(sys.argv[-1]))
