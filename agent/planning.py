"""
Planning components for the scientific agent framework.
Handles creating solution plans, analyzing dependencies, and allocating resources.
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class Planner:
    """Planning capabilities for scientific problem-solving."""
    
    def __init__(self, llm):
        """
        Initialize the planner.
        
        Args:
            llm: LLM provider to use for planning tasks
        """
        self.llm = llm
    
    def create_plan(self, formalized_problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a multi-step plan to solve the problem.
        
        Args:
            formalized_problem: Structured representation of the problem
            
        Returns:
            Solution plan with ordered steps
        """
        logger.info("Creating solution plan")
        
        # First, create the steps
        steps = self.create_steps(formalized_problem)
        
        # Then, analyze dependencies
        dependencies = self.analyze_dependencies(steps)
        
        # Finally, allocate resources
        resources = self.allocate_resources(steps, dependencies)
        
        # Return the complete plan
        return {
            "steps": steps,
            "dependencies": dependencies,
            "resources": resources,
            "estimated_completion_time": self._estimate_completion_time(steps, dependencies)
        }
    
    def create_steps(self, formalized_problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create individual steps for the solution plan.
        
        Args:
            formalized_problem: Structured representation of the problem
            
        Returns:
            List of plan steps with details
        """
        logger.info("Creating solution steps")
        
        steps_prompt = f"""
        You are planning a scientific investigation to solve this problem:
        {json.dumps(formalized_problem, indent=2)}
        
        Please create a step-by-step plan to solve this scientific problem. 
        Each step should be a well-defined task that can be implemented and evaluated.
        For each step, specify:
        
        1. A descriptive name for the step
        2. A detailed description of what needs to be done
        3. Whether the step requires generating and executing code
        4. Expected outputs of the step
        5. How to evaluate whether the step was successful
        
        Include a diverse range of appropriate steps such as:
        - Data gathering or generation
        - Data preprocessing and cleaning
        - Analysis and modeling
        - Hypothesis testing
        - Visualization
        - Evaluation and validation
        
        Format your response as a JSON array with objects having the following structure:
        [
            {{
                "id": "unique_identifier",
                "name": "Step name",
                "description": "Detailed description of the step",
                "requires_code": true/false,
                "expected_outputs": ["Output 1", "Output 2", ...],
                "success_criteria": ["Criterion 1", "Criterion 2", ...]
            }},
            ...
        ]
        """
        
        try:
            steps_result = self.llm.generate_with_json_output(steps_prompt, {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "requires_code": {"type": "boolean"},
                        "expected_outputs": {"type": "array", "items": {"type": "string"}},
                        "success_criteria": {"type": "array", "items": {"type": "string"}}
                    }
                }
            })
            
            # Add IDs if they're missing
            for i, step in enumerate(steps_result):
                if "id" not in step:
                    step["id"] = f"step_{i+1}"
                
                # Add type field for better step organization
                if i == 0:
                    step["type"] = "problem_formulation"
                elif "data" in step["name"].lower() and ("gather" in step["name"].lower() or "collect" in step["name"].lower()):
                    step["type"] = "data_gathering"
                elif "data" in step["name"].lower() and ("process" in step["name"].lower() or "clean" in step["name"].lower()):
                    step["type"] = "data_processing" 
                elif "analy" in step["name"].lower() or "model" in step["name"].lower():
                    step["type"] = "analysis"
                elif "test" in step["name"].lower() or "experiment" in step["name"].lower():
                    step["type"] = "testing"
                elif "visuali" in step["name"].lower() or "plot" in step["name"].lower():
                    step["type"] = "visualization"
                elif "evaluat" in step["name"].lower() or "validat" in step["name"].lower():
                    step["type"] = "evaluation"
                else:
                    step["type"] = "other"
            
            logger.debug(f"Created {len(steps_result)} solution steps")
            return steps_result
            
        except Exception as e:
            logger.error(f"Error creating solution steps: {str(e)}")
            # Return minimal plan with basic steps
            return [
                {
                    "id": "step_1",
                    "name": "Problem Analysis",
                    "description": "Analyze the problem and identify key components",
                    "requires_code": False,
                    "expected_outputs": ["Problem analysis document"],
                    "success_criteria": ["Clear understanding of the problem"],
                    "type": "problem_formulation"
                },
                {
                    "id": "step_2",
                    "name": "Data Collection",
                    "description": "Collect relevant data for the problem",
                    "requires_code": True,
                    "expected_outputs": ["Dataset"],
                    "success_criteria": ["Sufficient data collected"],
                    "type": "data_gathering"
                },
                {
                    "id": "step_3",
                    "name": "Solution Implementation",
                    "description": "Implement a solution to the problem",
                    "requires_code": True,
                    "expected_outputs": ["Solution implementation"],
                    "success_criteria": ["Solution addresses the problem"],
                    "type": "analysis"
                },
                {
                    "id": "step_4",
                    "name": "Evaluation",
                    "description": "Evaluate the solution against success criteria",
                    "requires_code": True,
                    "expected_outputs": ["Evaluation results"],
                    "success_criteria": ["Evaluation completed successfully"],
                    "type": "evaluation"
                }
            ]
    
    def analyze_dependencies(self, steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Analyze dependencies between steps.
        
        Args:
            steps: List of plan steps
            
        Returns:
            Dictionary mapping step IDs to lists of dependency step IDs
        """
        logger.info("Analyzing step dependencies")
        
        if not steps:
            return {}
        
        # For simple sequential plans, just make each step depend on the previous
        if len(steps) <= 4:
            dependencies = {}
            for i in range(len(steps)):
                step_id = steps[i]["id"]
                if i == 0:
                    dependencies[step_id] = []
                else:
                    dependencies[step_id] = [steps[i-1]["id"]]
            return dependencies
        
        # For more complex plans, use the LLM to analyze dependencies
        steps_json = json.dumps(steps, indent=2)
        dependencies_prompt = f"""
        You are analyzing the dependencies between steps in a scientific investigation plan.
        
        PLAN STEPS:
        {steps_json}
        
        For each step, identify which other steps it depends on (i.e., which steps must be completed before this step can begin).
        
        Format your response as a JSON object with step IDs as keys and arrays of dependency step IDs as values. If a step has no dependencies, use an empty array.
        
        Example format:
        {{
            "step_1": [],
            "step_2": ["step_1"],
            "step_3": ["step_1", "step_2"],
            "step_4": ["step_3"]
        }}
        """
        
        try:
            dependencies = self.llm.generate_with_json_output(dependencies_prompt, {})
            
            # Validate dependencies to avoid cycles
            valid_deps = self._validate_dependencies(dependencies, steps)
            logger.debug(f"Analyzed step dependencies: {json.dumps(valid_deps, indent=2)}")
            return valid_deps
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {str(e)}")
            # Fall back to sequential dependencies
            dependencies = {}
            for i in range(len(steps)):
                step_id = steps[i]["id"]
                if i == 0:
                    dependencies[step_id] = []
                else:
                    dependencies[step_id] = [steps[i-1]["id"]]
            return dependencies
    
    def _validate_dependencies(self, dependencies: Dict[str, List[str]], steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Validate dependencies to ensure they are acyclic and refer to actual steps.
        
        Args:
            dependencies: Dictionary of step dependencies
            steps: List of plan steps
            
        Returns:
            Validated dependencies dictionary
        """
        valid_deps = {}
        step_ids = [step["id"] for step in steps]
        
        for step_id in step_ids:
            # Make sure each step has an entry
            if step_id not in dependencies:
                valid_deps[step_id] = []
                continue
            
            # Filter out invalid dependencies
            valid_step_deps = [dep for dep in dependencies[step_id] 
                              if dep in step_ids and dep != step_id]
            
            valid_deps[step_id] = valid_step_deps
        
        # Check for circular dependencies (using a simple approach)
        changed = True
        while changed:
            changed = False
            for step_id, deps in valid_deps.items():
                for dep in deps:
                    if step_id in valid_deps.get(dep, []):
                        # Circular dependency found - remove one direction
                        valid_deps[dep] = [d for d in valid_deps[dep] if d != step_id]
                        changed = True
        
        return valid_deps
    
    def allocate_resources(self, steps: List[Dict[str, Any]], dependencies: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Allocate resources to steps.
        
        Args:
            steps: List of plan steps
            dependencies: Dictionary of step dependencies
            
        Returns:
            Dictionary mapping step IDs to lists of required resources
        """
        logger.info("Allocating resources to steps")
        
        resources = {}
        
        # Analyze each step for required resources
        for step in steps:
            step_id = step["id"]
            
            # Default resources based on step type
            if step.get("requires_code", False):
                resources[step_id] = ["computation", "code_execution"]
            else:
                resources[step_id] = []
            
            # Add resources based on step name/description
            description = f"{step.get('name', '')} {step.get('description', '')}"
            
            if any(term in description.lower() for term in ["data", "dataset"]):
                resources[step_id].append("data_storage")
            
            if any(term in description.lower() for term in ["visual", "plot", "graph", "chart"]):
                resources[step_id].append("visualization")
            
            if any(term in description.lower() for term in ["model", "train", "machine learning", "deep learning"]):
                resources[step_id].extend(["computation", "model_training"])
            
            if any(term in description.lower() for term in ["experiment", "test", "simulation"]):
                resources[step_id].append("experiment_execution")
            
            # Remove duplicates
            resources[step_id] = list(set(resources[step_id]))
        
        return resources
    
    def _estimate_completion_time(self, steps: List[Dict[str, Any]], dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Estimate completion time for the plan.
        
        Args:
            steps: List of plan steps
            dependencies: Dictionary of step dependencies
            
        Returns:
            Estimated completion time information
        """
        # Simple time estimation based on step types
        time_estimates = {}
        total_time = 0
        
        for step in steps:
            step_id = step["id"]
            step_type = step.get("type", "other")
            
            # Base time in minutes for each step type
            if step_type == "problem_formulation":
                time = 10
            elif step_type == "data_gathering":
                time = 20
            elif step_type == "data_processing":
                time = 15
            elif step_type == "analysis":
                time = 30
            elif step_type == "testing":
                time = 25
            elif step_type == "visualization":
                time = 15
            elif step_type == "evaluation":
                time = 20
            else:
                time = 15
            
            # Adjust based on whether code is required
            if step.get("requires_code", False):
                time *= 1.5
            
            time_estimates[step_id] = time
            total_time += time
        
        return {
            "total_minutes": total_time,
            "step_estimates": time_estimates,
            "human_readable": f"Approximately {int(total_time/60)} hours and {total_time%60} minutes"
        }
    
    def adapt_plan(self, current_plan: Dict[str, Any], new_information: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update plan based on new information or failed steps.
        
        Args:
            current_plan: Current solution plan
            new_information: New information to incorporate
            
        Returns:
            Updated solution plan
        """
        logger.info("Adapting solution plan")
        
        # Extract current plan components
        current_steps = current_plan.get("steps", [])
        current_dependencies = current_plan.get("dependencies", {})
        
        adaptation_prompt = f"""
        You are adapting a scientific investigation plan based on new information.
        
        CURRENT PLAN:
        {json.dumps(current_plan, indent=2)}
        
        NEW INFORMATION:
        {json.dumps(new_information, indent=2)}
        
        Please update the plan to incorporate this new information. You can:
        1. Keep existing steps unchanged
        2. Modify existing steps
        3. Add new steps
        4. Remove existing steps
        
        Format your response as a JSON object with the following structure:
        {{
            "steps": [
                {{
                    "id": "step_id",
                    "name": "Step name",
                    "description": "Detailed description",
                    "requires_code": true/false,
                    "expected_outputs": ["Output 1", "Output 2", ...],
                    "success_criteria": ["Criterion 1", "Criterion 2", ...],
                    "type": "step_type"
                }},
                ...
            ],
            "modifications_explanation": "Explanation of why and how the plan was modified"
        }}
        """
        
        try:
            updated_plan = self.llm.generate_with_json_output(adaptation_prompt, {})
            
            # Extract updated steps
            updated_steps = updated_plan.get("steps", current_steps)
            
            # Re-analyze dependencies for the updated steps
            updated_dependencies = self.analyze_dependencies(updated_steps)
            
            # Re-allocate resources
            updated_resources = self.allocate_resources(updated_steps, updated_dependencies)
            
            return {
                "steps": updated_steps,
                "dependencies": updated_dependencies,
                "resources": updated_resources,
                "estimated_completion_time": self._estimate_completion_time(updated_steps, updated_dependencies),
                "modifications_explanation": updated_plan.get("modifications_explanation", "Plan updated based on new information")
            }
            
        except Exception as e:
            logger.error(f"Error adapting plan: {str(e)}")
            return current_plan