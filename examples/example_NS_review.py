import sys

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from oppenai.agents import LiteratureAgent


def main():
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = """ 
        Browse the arxiv literature and provide a summary of the experimental constraints on the neutron star radius. 
        """
        model = ChatLiteLLM(
            model="openai/o3-mini",
            max_tokens=50000,
            max_retries=2,
        )

        init = {"messages": [HumanMessage(content=problem)]}

        # Initialize the agent
        literature_agent = LiteratureAgent(llm=model)

        # Solve the problem
        final_results = literature_agent.action.invoke(
            {"messages": [HumanMessage(content=problem)]}
        )

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
