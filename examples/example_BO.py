from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from oppenai.agents import ExecutionAgent


def main():
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = """ 
        "Optimize the six-hump camel function. 
            Start by evaluating that function at 10 locations.
            Then utilize Bayesian optimization to build a surrogate model 
                and sequentially select points until the function is optimized. 
            Carry out the optimization and report the results.
        """

        model = ChatLiteLLM(
            # model="openai/o3-mini",
            model="ollama_chat/llama3.1:8b",
            max_tokens=50000,
            max_retries=2,
        )

        init = {
            "messages": [HumanMessage(content=problem)],
            "workspace": "workspace_BO",
        }

        print(f"\nSolving problem: {problem}\n")

        # Initialize the agent
        executor = ExecutionAgent(llm=model)

        # Solve the problem
        final_results = executor.action.invoke(init)
        for x in final_results["messages"]:
            print(x.content)

        return final_results["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    main()
