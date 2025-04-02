import sys
sys.path.append("../../.")

from lanl_scientific_agent.agents import ExecutionAgent, ResearchAgent, HypothesizerAgent
from lanl_scientific_agent.agents import HypothesizerState
# from lanl_scientific_agent.agent.execution_agent import ExecutionAgent
from langchain_core.messages                     import HumanMessage
from langchain_openai                            import ChatOpenAI
from langchain_ollama.chat_models                import ChatOllama

def main():
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = """ 
         - Research the SEPIA python package for doing uncertainty quantification
              - You may need to use search to find the package on github
         - Identify the general structure for fitting a calibration model and summarize it in a pedogogical format
         - Install the package locally and carry out one example problem to demonstrate the use. Do not use alternative packages!
         - Summarize the example
         - Suggest paths for using this package for science problems in the Department of Energy
        """
        model = ChatOpenAI(
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
        
        init = {"messages": [HumanMessage(content=problem)]}
        
        print(f"\nSolving problem: {problem}\n")
        
        # Initialize the agents
        researcher   = ResearchAgent(llm      = model)
        executor     = ExecutionAgent(llm     = model)
        hypothesizer = HypothesizerAgent(llm  = model)
        
        # Solve the problem
        inputs          = {"messages": [HumanMessage(content=problem)]}
        research_result = researcher.action.invoke(inputs)

        initial_state = HypothesizerState(
            question              = problem + "".join([str(x.content) for x in research_result["messages"]]),
            question_search_query = "",
            current_iteration     =  0,
            max_iterations        =  3,
            agent1_solution       = [],
            agent2_critiques      = [],
            agent3_perspectives   = [],
            final_solution        = "",
        )

        hypothesis_results   = hypothesizer.action.invoke(initial_state)

        executor_messages = research_result["messages"].append(HumanMessage(content=hypothesis_results["summary_report"]))
        final_results     = executor.action.invoke(executor_messages)
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
