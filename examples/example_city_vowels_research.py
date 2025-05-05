import sys

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from oppenai.agents import ResearchAgent


def main(mode: str):
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = "Find a city with as least 10 vowels in its name."
        model = ChatLiteLLM(
            model="openai/o3-mini"
            if mode == "prod"
            else "ollama_chat/llama3.1:8b",
            max_tokens=20000 if mode == "prod" else 4000,
            max_retries=2,
        )

        init = {"messages": [HumanMessage(content=problem)]}

        print(f"\nSolving problem: {problem}\n")

        # Initialize the agent
        researcher = ResearchAgent(llm=model)

        # Solve the problem
        research_output = researcher.action.invoke(init)
        print(research_output["messages"][-1].content)
        for x in research_output["messages"]:
            print(x.content)

        return research_output["messages"][-1].content, research_output.get(
            "urls_visited", None
        )

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    final_output = main(mode=sys.argv[-1])
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)
    print(final_output[0])
    print(final_output[1])

# Solving problem: Find a city with as least 10 vowels in its name.
