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
        # Define a simple problem
        problem = problem_definition
        
        print(f"\nSolving problem: {problem}\n")
        
        # Initialize the agent
        agent = ScientificAgent(
            llm_provider="openai",
            model_name="o3-mini",
            workspace_dir="./materials_workspace"
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
