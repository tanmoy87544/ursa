"""
Example usage of the scientific agent for solving a specific problem.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent components
from lanl_scientific_agent.agent.core import ScientificAgent
from lanl_scientific_agent.utils.logging import setup_logging

def example_quantum_simulation():
    """Example: Optimizing a quantum simulation algorithm."""
    # Setup logging
    setup_logging(logging.INFO)
    
    # Define the problem
    problem = """
    I'm trying to optimize a quantum simulation algorithm for studying nuclear reactions. 
    The current simulation takes too long and consumes excessive computational resources. 
    We need to identify bottlenecks in the algorithm and suggest improvements that maintain accuracy 
    while reducing computational requirements. The algorithm uses a tensor network approach 
    for representing quantum states of nuclei undergoing reactions.
    """
    
    # Initialize agent
    agent = ScientificAgent(
        llm_provider="ollama",
        model_name="llama3",
        workspace_dir="./workspace/quantum_sim"
    )
    
    # Solve the problem
    print(f"Solving problem: Quantum Simulation Optimization")
    result = agent.solve(problem)
    
    # Print the summary
    print_summary(result)
    
    return result

def example_material_science():
    """Example: Analyzing crystal structure data."""
    # Setup logging
    setup_logging(logging.INFO)
    
    # Define the problem
    problem = """
    We have X-ray diffraction data from a novel superconducting material. 
    We need to analyze the crystal structure and relate it to the observed superconducting 
    properties. The data shows unexpected peaks that don't match known crystal structures 
    for similar materials. We want to develop a model that explains the relationship between 
    the crystal structure and the material's unusually high critical temperature.
    """
    
    # Initialize agent
    agent = ScientificAgent(
        llm_provider="ollama",
        model_name="llama3",
        workspace_dir="./workspace/materials"
    )
    
    # Solve the problem
    print(f"Solving problem: Crystal Structure Analysis")
    result = agent.solve(problem)
    
    # Print the summary
    print_summary(result)
    
    return result

def example_climate_modeling():
    """Example: Developing a regional climate model."""
    # Setup logging
    setup_logging(logging.INFO)
    
    # Define the problem
    problem = """
    We need to develop a regional climate model for the southwestern United States 
    that can predict drought conditions with higher accuracy than current models. 
    The model should incorporate data on precipitation patterns, temperature trends, 
    soil moisture levels, and atmospheric conditions. We want to test different machine 
    learning approaches and compare them with traditional physical models to see which 
    provides the most accurate predictions.
    """
    
    # Initialize agent
    agent = ScientificAgent(
        llm_provider="ollama",
        model_name="llama3",
        workspace_dir="./workspace/climate"
    )
    
    # Solve the problem
    print(f"Solving problem: Regional Climate Modeling")
    result = agent.solve(problem)
    
    # Print the summary
    print_summary(result)
    
    return result

def print_summary(result: Dict[str, Any]) -> None:
    """Print a summary of the solution."""
    print("\n" + "="*80)
    print("SOLUTION SUMMARY")
    print("="*80)
    
    # Check if there was an error
    if "error" in result:
        print(f"\nERROR:")
        print(f"{result['error']}")
        print(f"\nExecution time: {result.get('duration_seconds', 0):.2f} seconds")
        return
    
    # Display solution summary
    summary = result.get("summary", {})
    
    print(f"\nPROBLEM OVERVIEW:")
    print(f"{summary.get('problem_overview', 'Not available')}")
    
    print(f"\nAPPROACH:")
    print(f"{summary.get('approach_summary', 'Not available')}")
    
    print(f"\nKEY FINDINGS:")
    for finding in summary.get("key_findings", ["Not available"]):
        print(f"- {finding}")
    
    print(f"\nPROBLEM RESOLUTION:")
    print(f"{summary.get('problem_resolution', 'Not available')}")
    
    print(f"\nExecution time: {result.get('duration_seconds', 0):.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    # Run the quantum simulation example
    example_quantum_simulation()
