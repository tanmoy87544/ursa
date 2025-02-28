#!/usr/bin/env python
"""
Main entry point for the scientific agent framework.
Demonstrates setup and usage of the agent for solving scientific problems.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agent components
from lanl_scientific_agent.agent.core import ScientificAgent
from lanl_scientific_agent.utils.logging import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Scientific Agent Framework")
    
    parser.add_argument("--problem", "-p", type=str, help="Scientific problem to solve")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--llm", "-l", type=str, default="ollama", help="LLM provider (ollama or langchain)")
    parser.add_argument("--model", "-m", type=str, default="llama3", help="LLM model name")
    parser.add_argument("--workspace", "-w", type=str, help="Workspace directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def run_agent(args) -> Dict[str, Any]:
    """Initialize and run the scientific agent."""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Create workspace directory
    workspace_dir = args.workspace or os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Initialize agent
    agent = ScientificAgent(
        config_path=args.config,
        llm_provider=args.llm,
        model_name=args.model,
        workspace_dir=workspace_dir
    )
    
    # Solve the problem
    if args.problem:
        result = agent.solve(args.problem)
    else:
        # Use a default example problem if none provided
        example_problem = "Analyze the relationship between temperature and bacterial growth rate in a laboratory setting. Determine the optimal temperature range for maximum growth."
        print(f"No problem specified, using example: \"{example_problem}\"")
        result = agent.solve(example_problem)
    
    return result

def display_result(result: Dict[str, Any]) -> None:
    """Display the agent's solution result."""
    print("\n" + "="*80)
    print("SCIENTIFIC AGENT SOLUTION SUMMARY")
    print("="*80)
    
    # Display problem
    print(f"\nPROBLEM:")
    print(f"{result.get('problem_statement')}")
    
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
    
    print(f"\nLIMITATIONS:")
    for limitation in summary.get("limitations", ["Not available"]):
        print(f"- {limitation}")
    
    print(f"\nPOTENTIAL IMPACT:")
    for impact in summary.get("potential_impact", ["Not available"]):
        print(f"- {impact}")
    
    print(f"\nFUTURE WORK:")
    for work in summary.get("future_work", ["Not available"]):
        print(f"- {work}")
    
    # Display execution details
    print(f"\nExecution time: {result.get('duration_seconds', 0):.2f} seconds")
    
    # Display path to results
    if "execution_results" in result:
        created_files = []
        for step_id, step_result in result["execution_results"].items():
            if isinstance(step_result, dict) and "created_files" in step_result:
                created_files.extend(step_result["created_files"])
            elif isinstance(step_result, dict) and "visualizations" in step_result:
                created_files.extend(step_result["visualizations"])
        
        if created_files:
            print(f"\nCreated files:")
            for file_path in created_files:
                print(f"- {file_path}")
    
    print("\n" + "="*80)

def main():
    """Main entry point."""
    args = parse_args()
    result = run_agent(args)
    display_result(result)

if __name__ == "__main__":
    main()
