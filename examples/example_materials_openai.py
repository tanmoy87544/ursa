import sys
sys.path.append("../../.")

from lanl_scientific_agent.agent.execution_agent import ExecutionAgent
from lanl_scientific_agent.agent.planning_agent  import PlanningAgent
from langchain_core.messages                     import HumanMessage
from langchain_openai                            import ChatOpenAI
from langchain_ollama.chat_models                import ChatOllama


problem_definition = '''
Developing materials that are able to stay brittle at low temperatures is a critical part of advancing space travel.

High-entropy alloys have potential to develop metals that are not brittle in the cold temperatures of space.

Hypothesize some metal combinations that may lead to useful alloys and identify the mixture weights for these metals for optimal alloys.

Your only tool for identifying the materials is through writing python code. You cannot perform any materials synthesis or experimental testing.

In the end we should have a list of high-entropy alloys that are not brittle at low temperature and a justification for this.
'''


def main():
    """Run a simple example of the scientific agent."""
    try:
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
        
        init = {"messages": [HumanMessage(content=problem_definition)]}
        
        print(f"\nSolving problem: {problem_definition}\n")
        
        # Initialize the agent
        planner  = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)
        
        # Solve the problem
        planning_output = planner.action.invoke(init)
        print(planning_output["messages"][-1].content)
        last_step_string = "Beginning step 1 of the plan. "
        for x in planning_output["plan_steps"]:
            plan_string      = str(x)
            final_results    = executor.action.invoke({"messages": [HumanMessage(content=last_step_string + plan_string)]})
            last_step_string = final_results["messages"][-1].content
            print(last_step_string)
                
        return final_results["messages"][-1].content
    
    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    main()
