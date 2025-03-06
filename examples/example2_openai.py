"""
A minimal example script to test the ScientificAgent with error handling.
"""

import os
import sys
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the ScientificAgent
try:
    from lanl_scientific_agent.agent.core import ScientificAgent
except ImportError as e:
    print(f"Error importing ScientificAgent: {e}")
    sys.exit(1)

def main():
    """Run a simple example of the scientific agent."""
    try:
        # Define a simple problem
        # problem = "How does temperature affect the rate of a chemical reaction?"
        # problem = "Propose an acquisition function for multi-fidelity Bayesian optimization that allows selection across fidelities."
        problem = "Find a city with as least 10 vowels in its name."
        
        print(f"\nSolving problem: {problem}\n")
        
        # Initialize the agent
        agent = ScientificAgent(
            llm_provider="openai",
            model_name="gpt-4o",
            workspace_dir="./workspace"
        )
        
        # Solve the problem
        result = agent.solve(problem)
        
        # Print some basic results
        print("\n" + "="*80)
        print("SOLUTION RESULTS")
        print("="*80)
        
        print(f"\nStatus: {result.get('status', 'N/A')}")
        print(f"Duration: {result.get('duration_seconds', 0):.2f} seconds")
        
        # Check if we have a summary
        if "summary" in result and result["summary"]:
            summary = result["summary"]
            
            print("\nPROBLEM OVERVIEW:")
            print(summary.get("problem_overview", "Not available"))
            
            print("\nKEY FINDINGS:")
            for finding in summary.get("key_findings", ["Not available"]):
                print(f"- {finding}")
        else:
            print("\nNo summary available in results.")
            print("Available keys in result:", ", ".join(result.keys()))
        
        return result
    
    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    main()
