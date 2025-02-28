"""
Execution components for the scientific agent framework.
Handles executing solution plans, running code, and managing results.
"""

import logging
import json
import os
import subprocess
import tempfile
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

logger = logging.getLogger(__name__)

class Executor:
    """Execution capabilities for scientific problem-solving."""
    
    def __init__(self, llm, workspace_dir: str, timeout: int = 60):
        """
        Initialize the executor.
        
        Args:
            llm: LLM provider to use for code generation and analysis
            workspace_dir: Directory for storing execution artifacts
            timeout: Default timeout for code execution in seconds
        """
        self.llm = llm
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        
        # Create workspace if it doesn't exist
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Dictionary of registered tools
        self._tools = {}
        
        # Register built-in tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools for use in execution."""
        self._tools.update({
            "generate_code": self.generate_code,
            "execute_code": self.execute_code,
            "analyze_data": self.analyze_data,
            "visualize_data": self.visualize_data
        })
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get dictionary of available tools."""
        return self._tools
    
    def prepare_step(self, plan: Dict[str, Any], step_index: int, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a step for execution.
        
        Args:
            plan: The solution plan
            step_index: Index of the step to prepare
            previous_results: Results from previous steps
            
        Returns:
            Prepared step state
        """
        logger.info(f"Preparing step {step_index + 1}")
        
        # Get the step to execute
        steps = plan.get("steps", [])
        if not steps or step_index >= len(steps):
            raise ValueError(f"Invalid step index: {step_index}")
        
        step = steps[step_index]
        
        # Check dependencies
        dependencies = plan.get("dependencies", {}).get(step["id"], [])
        unsatisfied_deps = []
        
        for dep_id in dependencies:
            if dep_id not in previous_results:
                unsatisfied_deps.append(dep_id)
        
        if unsatisfied_deps:
            raise ValueError(f"Cannot execute step {step['id']} because dependencies are not satisfied: {unsatisfied_deps}")
        
        # Prepare step-specific resources
        prepared_state = {
            "step": step,
            "previous_results": previous_results,
            "dependencies": dependencies,
            "resources": plan.get("resources", {}).get(step["id"], []),
            "preparation_time": time.time()
        }
        
        return prepared_state
    
    def execute_step(self, step: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step from the solution plan.
        
        Args:
            step: The step to execute
            previous_results: Results from previous steps
            
        Returns:
            Execution results
        """
        logger.info(f"Executing step: {step.get('name', step.get('id', 'unknown'))}")
        
        step_type = step.get("type", "other")
        
        # Dispatch based on step type
        if step.get("requires_code", False):
            # Steps requiring code execution
            return self._execute_code_step(step, previous_results)
        elif step_type == "visualization":
            # Visualization steps
            return self._execute_visualization_step(step, previous_results)
        else:
            # General steps (analysis, evaluation, etc.)
            return self._execute_general_step(step, previous_results)
    
    def _execute_code_step(self, step: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step that requires code."""
        logger.info(f"Executing code step: {step.get('name')}")
        
        try:
            # Generate code based on step description and previous results
            code_result = self.generate_code(step, previous_results)
            
            # Execute the generated code
            execution_result = self.execute_code(code_result["code"], code_result["language"])
            
            # Analyze the execution result
            analysis = self._analyze_execution_result(step, execution_result)
            
            return {
                "status": "completed" if not execution_result.get("error") else "failed",
                "code": code_result["code"],
                "language": code_result["language"],
                "execution_result": execution_result,
                "analysis": analysis,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing code step: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _execute_visualization_step(self, step: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a visualization step."""
        logger.info(f"Executing visualization step: {step.get('name')}")
        
        try:
            # Generate visualization code
            viz_code = self.visualize_data(step, previous_results)
            
            # Execute the visualization code
            execution_result = self.execute_code(viz_code["code"], viz_code["language"])
            
            # Save any generated visualization files
            visualizations = []
            if "file_path" in viz_code and os.path.exists(viz_code["file_path"]):
                visualizations.append(viz_code["file_path"])
            
            return {
                "status": "completed" if not execution_result.get("error") else "failed",
                "code": viz_code["code"],
                "language": viz_code["language"],
                "visualizations": visualizations,
                "execution_result": execution_result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing visualization step: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _execute_general_step(self, step: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general step that doesn't fit other categories."""
        logger.info(f"Executing general step: {step.get('name')}")
        
        try:
            # For steps that don't require code, use the LLM to perform the task
            task_prompt = f"""
            You are executing a step in a scientific investigation.
            
            STEP DETAILS:
            {json.dumps(step, indent=2)}
            
            PREVIOUS RESULTS:
            {json.dumps(previous_results, indent=2)}
            
            Please perform this step based on the previous results. 
            Your task is: {step.get('description', 'Complete the task described in the step details')}
            
            Format your response as a JSON object with the following structure:
            {{
                "result_summary": "Brief summary of the result",
                "detailed_findings": "Detailed explanation of findings or results",
                "meets_success_criteria": true/false,
                "success_criteria_evaluation": ["Criterion 1: Evaluation", "Criterion 2: Evaluation", ...],
                "outputs": {{"output_name": "output_value", ...}}
            }}
            """
            
            result = self.llm.generate_with_json_output(task_prompt, {})
            
            return {
                "status": "completed",
                "result": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing general step: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def handle_error(self, plan: Dict[str, Any], step_index: int, error: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle execution errors.
        
        Args:
            plan: The solution plan
            step_index: Index of the failed step
            error: Error message
            previous_results: Results from previous steps
            
        Returns:
            Error handling result
        """
        logger.info(f"Handling error in step {step_index + 1}: {error}")
        
        # Get the failed step
        steps = plan.get("steps", [])
        if not steps or step_index >= len(steps):
            logger.error(f"Invalid step index: {step_index}")
            return {"error_resolved": False}
        
        step = steps[step_index]
        
        # Generate error handling strategy
        error_prompt = f"""
        You are debugging an error in a scientific workflow.
        
        STEP THAT FAILED:
        {json.dumps(step, indent=2)}
        
        ERROR:
        {error}
        
        PREVIOUS RESULTS:
        {json.dumps(previous_results, indent=2)}
        
        Please analyze this error and suggest a fix. Consider:
        1. What might have caused the error
        2. How to modify the step to avoid the error
        3. Whether the step needs to be broken down into smaller steps
        4. Whether to retry with different parameters
        
        Format your response as a JSON object with the following structure:
        {{
            "error_analysis": "Detailed analysis of the error",
            "fix_strategy": "Strategy to fix the error",
            "retry_with_modifications": true/false,
            "modified_step": {{...}},  // Only include if retry_with_modifications is true
            "skip_step": true/false    // Whether to skip this step entirely
        }}
        """
        
        try:
            error_handling = self.llm.generate_with_json_output(error_prompt, {})
            
            if error_handling.get("skip_step", False):
                logger.info(f"Skipping failed step: {step.get('name')}")
                return {
                    "error_resolved": True,
                    "resolution_strategy": "skip_step",
                    "error_analysis": error_handling.get("error_analysis", "")
                }
            elif error_handling.get("retry_with_modifications", False):
                logger.info(f"Retrying step with modifications: {step.get('name')}")
                modified_step = error_handling.get("modified_step", step)
                
                # Update the step in the plan
                steps[step_index] = modified_step
                
                return {
                    "error_resolved": True,
                    "resolution_strategy": "retry_with_modifications",
                    "error_analysis": error_handling.get("error_analysis", ""),
                    "modified_step": modified_step
                }
            else:
                logger.info(f"No resolution for step: {step.get('name')}")
                return {
                    "error_resolved": False,
                    "resolution_strategy": "no_resolution",
                    "error_analysis": error_handling.get("error_analysis", "")
                }
                
        except Exception as e:
            logger.error(f"Error in error handling: {str(e)}")
            return {"error_resolved": False}
    
    def check_completion(self, plan: Dict[str, Any], current_step_index: int, results: Dict[str, Any]) -> Tuple[bool, int]:
        """
        Check if execution is complete and determine next step.
        
        Args:
            plan: The solution plan
            current_step_index: Index of the current step
            results: Results from executed steps
            
        Returns:
            Tuple of (is_complete, next_step_index)
        """
        steps = plan.get("steps", [])
        
        # If no steps or current step is past the end, we're done
        if not steps or current_step_index >= len(steps) - 1:
            return True, current_step_index
        
        # Move to the next step
        return False, current_step_index + 1
    
    def generate_code(self, step: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code to accomplish a specific task.
        
        Args:
            step: Step requiring code
            previous_results: Results from previous steps
            
        Returns:
            Generated code information
        """
        logger.info(f"Generating code for step: {step.get('name')}")
        
        # Determine language based on step
        language = "python"  # Default to Python
        
        # Extract relevant previous results
        relevant_results = {}
        dependencies = step.get("dependencies", [])
        for dep_id in dependencies:
            if dep_id in previous_results:
                relevant_results[dep_id] = previous_results[dep_id]
        
        # Construct prompt for code generation
        code_prompt = f"""
        You are generating code for a scientific investigation.
        
        STEP DETAILS:
        {json.dumps(step, indent=2)}
        
        RELEVANT PREVIOUS RESULTS:
        {json.dumps(relevant_results, indent=2)}
        
        Please generate {language} code to accomplish this step. The code should:
        1. Be well-documented with comments
        2. Include appropriate error handling
        3. Be efficient and follow best practices
        4. Read any necessary input data from previous steps
        5. Save any output data or visualizations to files
        6. Print informative messages about what it's doing
        
        If the step involves data analysis, include:
        - Data loading (from files if mentioned in previous results)
        - Data preprocessing and cleaning
        - Analysis methods appropriate for the task
        - Saving results for use in later steps
        
        If the step involves visualization, include:
        - Creating appropriate plots/visualizations
        - Adding proper labels, titles, and legends
        - Saving the visualizations to files
        
        Return the code only, without any additional explanations.
        """
        
        try:
            code = self.llm.generate(code_prompt)
            
            # Extract code if wrapped in markdown code blocks
            if "```" in code:
                code_parts = code.split("```")
                if len(code_parts) >= 3:
                    # Extract the language if specified
                    lang_line = code_parts[1].strip().split("\n")[0].strip()
                    if lang_line in ["python", "py"]:
                        language = "python"
                    
                    # Extract the actual code
                    if "\n" in code_parts[1]:
                        code = "\n".join(code_parts[1].strip().split("\n")[1:])
                    else:
                        code = code_parts[2].strip()
            
            return {
                "code": code,
                "language": language,
                "step_id": step.get("id"),
                "generation_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            # Return minimal code that prints the error
            return {
                "code": f'print("Error generating code: {str(e)}")',
                "language": language,
                "step_id": step.get("id"),
                "generation_time": time.time(),
                "error": str(e)
            }
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code in a controlled environment.
        
        Args:
            code: The code to execute
            language: Programming language of the code
            
        Returns:
            Execution results
        """
        logger.info(f"Executing {language} code")
        
        if language.lower() not in ["python", "py"]:
            logger.error(f"Unsupported language: {language}")
            return {
                "output": "",
                "error": f"Unsupported language: {language}",
                "execution_time": 0
            }
        
        # Create a temporary directory for execution
        execution_dir = tempfile.mkdtemp(dir=self.workspace_dir)
        
        # Write code to a file
        code_file = os.path.join(execution_dir, "code.py")
        with open(code_file, "w") as f:
            f.write(code)
        
        # Execute the code
        start_time = time.time()
        process = None
        
        try:
            # Run the code with timeout
            process = subprocess.Popen(
                ["python", code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=execution_dir
            )
            
            stdout, stderr = process.communicate(timeout=self.timeout)
            
            execution_time = time.time() - start_time
            
            # Get the list of files created during execution
            created_files = []
            for root, _, files in os.walk(execution_dir):
                for file in files:
                    if file != "code.py":
                        file_path = os.path.join(root, file)
                        created_files.append(file_path)
            
            return {
                "output": stdout,
                "error": stderr if stderr else None,
                "execution_time": execution_time,
                "exit_code": process.returncode,
                "created_files": created_files
            }
            
        except subprocess.TimeoutExpired:
            if process:
                process.kill()
                stdout, stderr = process.communicate()
            
            logger.error(f"Code execution timed out after {self.timeout} seconds")
            return {
                "output": "",
                "error": f"Execution timed out after {self.timeout} seconds",
                "execution_time": self.timeout
            }
            
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return {
                "output": "",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _analyze_execution_result(self, step: Dict[str, Any], execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the results of code execution.
        
        Args:
            step: The step that was executed
            execution_result: Results from code execution
            
        Returns:
            Analysis of the execution results
        """
        # Check for errors
        if execution_result.get("error"):
            return {
                "success": False,
                "error_analysis": f"Execution failed with error: {execution_result['error']}",
                "meets_success_criteria": False
            }
        
        # Check success criteria
        success_criteria = step.get("success_criteria", [])
        criteria_evaluation = []
        
        # For each criterion, check if it's met based on the output
        for criterion in success_criteria:
            # This is a simplified check - in a real system, you'd want more sophisticated criteria matching
            if any(word.lower() in execution_result.get("output", "").lower() for word in criterion.split()):
                criteria_evaluation.append({"criterion": criterion, "met": True})
            else:
                criteria_evaluation.append({"criterion": criterion, "met": False})
        
        # Check if all criteria are met
        all_criteria_met = all(eval["met"] for eval in criteria_evaluation)
        
        return {
            "success": all_criteria_met,
            "criteria_evaluation": criteria_evaluation,
            "meets_success_criteria": all_criteria_met,
            "output_analysis": "Output analysis would be here in a full implementation"
        }
    
    def analyze_data(self, data: Any, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze data of various types.
        
        Args:
            data: The data to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing data with type: {analysis_type}")
        
        # In a real implementation, this would dispatch to different analysis methods
        # based on data type and analysis type
        
        analysis_prompt = f"""
        You are analyzing data for a scientific investigation.
        
        DATA:
        {json.dumps(data)[:1000]}... [truncated]
        
        ANALYSIS TYPE:
        {analysis_type}
        
        Please analyze this data and provide insights. Include:
        1. Summary statistics
        2. Key patterns or trends
        3. Anomalies or outliers
        4. Suggestions for further analysis
        
        Format your response as a JSON object with appropriate analysis results.
        """
        
        try:
            analysis = self.llm.generate_with_json_output(analysis_prompt, {})
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return {
                "error": str(e),
                "analysis_time": time.time()
            }
    
    def visualize_data(self, step: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code for data visualization.
        
        Args:
            step: Step requiring visualization
            previous_results: Results from previous steps
            
        Returns:
            Visualization information
        """
        logger.info(f"Generating visualization for step: {step.get('name')}")
        
        # Extract relevant previous results
        relevant_results = {}
        dependencies = step.get("dependencies", [])
        for dep_id in dependencies:
            if dep_id in previous_results:
                relevant_results[dep_id] = previous_results[dep_id]
        
        # Determine visualization type
        viz_type = "general"
        step_desc = step.get("description", "").lower()
        
        if "scatter" in step_desc:
            viz_type = "scatter"
        elif "line" in step_desc:
            viz_type = "line"
        elif "bar" in step_desc:
            viz_type = "bar"
        elif "histogram" in step_desc:
            viz_type = "histogram"
        elif "heatmap" in step_desc:
            viz_type = "heatmap"
        
        # Generate visualization code
        viz_prompt = f"""
        You are generating code for data visualization in a scientific investigation.
        
        STEP DETAILS:
        {json.dumps(step, indent=2)}
        
        RELEVANT PREVIOUS RESULTS:
        {json.dumps(relevant_results, indent=2)}
        
        VISUALIZATION TYPE:
        {viz_type}
        
        Please generate Python code to create a high-quality visualization. The code should:
        1. Load any necessary data from previous results
        2. Process the data as needed for visualization
        3. Create a {viz_type} plot (or appropriate visualization if "general")
        4. Add proper labels, title, legend, and other enhancents
        5. Use a visually appealing style (e.g., with matplotlib, seaborn, or plotly)
        6. Save the visualization to a file (use timestamp in filename)
        7. Display the visualization if in an interactive environment
        
        Return only the Python code without additional explanations.
        """
        
        try:
            code = self.llm.generate(viz_prompt)
            
            # Extract code if wrapped in markdown code blocks
            if "```" in code:
                code_parts = code.split("```")
                if len(code_parts) >= 3:
                    # Extract the actual code
                    if "\n" in code_parts[1]:
                        code = "\n".join(code_parts[1].strip().split("\n")[1:])
                    else:
                        code = code_parts[2].strip()
            
            # Create a timestamp for the file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Generate a filename for the visualization
            viz_filename = f"visualization_{viz_type}_{timestamp}.png"
            file_path = os.path.join(self.workspace_dir, viz_filename)
            
            # Modify the code to save to our specific path
            if "plt.savefig" not in code:
                code += f"\n\nimport matplotlib.pyplot as plt\nplt.savefig('{file_path}')\n"
            
            return {
                "code": code,
                "language": "python",
                "visualization_type": viz_type,
                "file_path": file_path,
                "step_id": step.get("id"),
                "generation_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization code: {str(e)}")
            return {
                "code": f'import matplotlib.pyplot as plt\nplt.figure()\nplt.text(0.5, 0.5, "Error generating visualization: {str(e)}", ha="center", va="center")\nplt.savefig("error_visualization.png")',
                "language": "python",
                "visualization_type": "error",
                "step_id": step.get("id"),
                "generation_time": time.time(),
                "error": str(e)
            }
