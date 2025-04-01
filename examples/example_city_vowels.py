import sys
sys.path.append("../../.")

from lanl_scientific_agent.agent.execution_agent import ExecutionAgent
from lanl_scientific_agent.agent.planning_agent  import PlanningAgent
from langchain_core.messages                     import HumanMessage
from langchain_openai                            import ChatOpenAI
from langchain_ollama.chat_models                import ChatOllama

def main():
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = "Find a city with as least 10 vowels in its name."
        model = ChatOpenAI(
            model       = "o3-mini",
            max_tokens  = 10000,
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
        planner  = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)
        
        # Solve the problem
        planning_output = planner.action.invoke(init)
        print(planning_output["messages"][-1].content)
        final_results   = executor.action.invoke(planning_output)
        for x in final_results["messages"]:
            print(x.content)
        # print(final_results["messages"][-1].content)
                
        return final_results["messages"][-1].content
    
    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    main()
