import sys

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage
from pypdf import PdfReader

from ursa.agents import (
    ExecutionAgent,
    PlanningAgent,
)
from ursa.prompt_library.planning_prompts import (
    detailed_planner_prompt,
)

reader = PdfReader("/Users/mikegros/Downloads/2406.03360v2.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = "".join([x.extract_text() for x in reader.pages])

problem_definition = f"""
The following is a published paper about determinantal point processes. 
{text}

  - Replicate the examples in python
  - Visualize the results
  - Summarize how your example compares to the published paper.
"""


def main(mode: str):
    """Run a simple example of the scientific agent."""
    try:
        model = ChatLiteLLM(
            model="openai/o3-mini"
            if mode == "prod"
            else "ollama_chat/llama3.1:8b",
            max_tokens=20000 if mode == "prod" else 4000,
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
            detail_planner = PlanningAgent(llm=model)
            plan_string = str(x)
            detail_planner.planner_prompt = detailed_planner_prompt
            detail_output = detail_planner.action.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=last_step_string
                            + plan_string
                            + detail_plan_string
                        )
                    ]
                },
                {"recursion_limit": 999999},
            )
            last_substep_string = "Beginning to break down of the plan. "
            for y in detail_output["plan_steps"]:
                execute_string = "Execute this step and report results for the executor of the next step. Do not use placeholders but fully carry out each step."
                final_results = executor.action.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=last_substep_string
                                + str(y)
                                + execute_string
                            )
                        ],
                        "workspace": "workspace_DPPpaper",
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
    main(sys.argv[-1])


# execute_string   = "Flesh out the details of this step and report results for the executor of the next step. Do not use placeholders but fully carry out each step."
