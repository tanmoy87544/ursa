import sys
sys.path.append("../../.")

from lanl_scientific_agent.agents import ExecutionAgent, LiteratureAgent
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
            Then utilize Bayesian optimization to build a surrogate model and sequentially select points until the function is optimized. 
            Execute the code, carry out the optimization and report the results.
        """
        
        model = ChatOpenAI(
            model       = "o3-mini",
            max_tokens  = 50000,
            timeout     = None,
            max_retries = 2)
 
        
        init = {"messages": [HumanMessage(content=problem)]}
                
        executor         = ExecutionAgent(llm=model)
        literature_agent = LiteratureAgent(llm=model)

        
        execution_results   = executor.action.invoke(init)

        execution_results = execution_results["messages"][-1].content

        start_string = 'Here is the summary of results from a calculation: \n"'+ execution_results
        start_string = start_string+'"\n Browse the arxiv literature to see if this is reasonable.'
        
        final_results    = literature_agent.action.invoke({"messages": [HumanMessage(content=start_string)]})

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
