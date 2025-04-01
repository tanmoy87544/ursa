import sys
sys.path.append("../../.")

from lanl_scientific_agent.agents import ExecutionAgent
from langchain_core.messages      import HumanMessage
from langchain_openai             import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

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
        model = ChatOpenAI(
            model       = "o3-mini",
            max_tokens  = 50000,
            timeout     = None,
            max_retries = 2)
        # model = ChatOllama(
        #     model       = "llama3.1:8b",
        #     max_tokens  = 4000,
        #     timeout     = None,
        #     max_retries = 2
        # )
        
        init = {"messages": [HumanMessage(content=problem)]}
        
        print(f"\nSolving problem: {problem}\n")
        
        # Initialize the agent
        executor = ExecutionAgent(llm=model)
        
        # Solve the problem
        final_results   = executor.action.invoke(init)
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
