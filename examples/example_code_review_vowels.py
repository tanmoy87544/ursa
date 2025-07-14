import sys

from langchain_community.chat_models import ChatLiteLLM

from oppenai.agents import CodeReviewAgent


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
            "project_prompt": "Find a city with as least 10 vowels in its name.",
            "code_files": ["vowel_count.py"],
        }

        result = code_review_agent.action.invoke(initial_state)
        for x in result["messages"]:
            print(x.content)
        return result["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    main(sys.argv[-1])
