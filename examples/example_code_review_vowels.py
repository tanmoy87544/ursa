import sys
sys.path.append("../../.")

from lanl_scientific_agent.agents import CodeReviewAgent
from langchain_openai             import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

def main():
    try:
        model_o3 = ChatOpenAI(
            model       = "o3-mini",
            max_tokens  = 40000,
            timeout     = None,
            max_retries = 2)
        model_o1 = ChatOpenAI(
            model       = "o1",
            max_tokens  = 50000,
            timeout     = None,
            max_retries = 2)
        # model = ChatOllama(
        #     model       = "llama3.1:8b",
        #     max_tokens  = 4000,
        #     timeout     = None,
        #     max_retries = 2
        # )

        code_review_agent = CodeReviewAgent(llm = model_o3)

        initial_state = {
            "messages": [],
            "project_prompt": "Find a city with as least 10 vowels in its name.",
            "code_files": ["vowel_count.py"]
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
    main()
