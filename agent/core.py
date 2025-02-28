"""
Core agent implementation for the scientific problem-solving framework.
"""

import logging
import time
import uuid
import os
import yaml
from typing import Dict, Any, List, Optional, Union, Tuple

# Local imports
from .memory import Memory
from .reasoning import Reasoner
from .planning import Planner
from .execution import Executor
from ..llm.ollama_provider import OllamaProvider
from ..llm.langchain_provider import LangChainProvider
from ..llm.langgraph_workflows import ScientificWorkflows, ProblemState, PlanningState, ExecutionState

logger = logging.getLogger(__name__)

class ScientificAgent:
    """Agent for solving scientific problems."""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 llm_provider: str = "ollama",
                 model_name: str = "llama3",
                 workspace_dir: Optional[str] = None):
        """
        Initialize the scientific agent.
        
        Args:
            config_path: Path to configuration file
            llm_provider: Type of LLM provider ("ollama" or "langchain")
            model_name: Name of the LLM model to use
            workspace_dir: Directory for storing work artifacts
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up workspace
        self.workspace_dir = workspace_dir or os.path.join(os.getcwd(), "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Initialize LLM
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Initialize components
        self.memory = Memory()
        self.reasoner = Reasoner(self.llm)
        self.planner = Planner(self.llm)
        self.executor = Executor(self.llm, self.workspace_dir)
        
        # Initialize workflows
        self.workflows = self._initialize_workflows()
        
        # Generate a unique ID for this agent instance
        self.agent_id = str(uuid.uuid4())
        logger.info(f"Scientific agent initialized with ID: {self.agent_id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "llm": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "system_prompt": "You are a scientific problem-solving assistant that helps with research and experimentation."
            },
            "tools": {
                "enable_code_execution": True,
                "execution_timeout": 60,
                "max_iterations": 10
            },
            "logging": {
                "level": "INFO",
                "save_history": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                # Merge configs, with user config taking precedence
                config = default_config.copy()
                for key, value in user_config.items():
                    if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
                
                return config
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {str(e)}")
                logger.warning("Using default configuration")
                return default_config
        else:
            return default_config
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration."""
        llm_config = self.config.get("llm", {})
        
        if self.llm_provider.lower() == "ollama":
            return OllamaProvider(
                model_name=self.model_name,
                temperature=llm_config.get("temperature", 0.1),
                max_tokens=llm_config.get("max_tokens", 4096),
                system_prompt=llm_config.get("system_prompt")
            )
        elif self.llm_provider.lower() == "langchain":
            return LangChainProvider(
                model_name=self.model_name,
                provider="ollama",  # Using Ollama as the backend
                temperature=llm_config.get("temperature", 0.1),
                max_tokens=llm_config.get("max_tokens", 4096),
                system_prompt=llm_config.get("system_prompt")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _initialize_workflows(self):
        """Initialize LangGraph workflows."""
        return ScientificWorkflows(tools=self.executor.get_tools())
    
    def solve(self, problem_statement: str) -> Dict[str, Any]:
        """
        Solve a scientific problem end-to-end.
        
        Args:
            problem_statement: Natural language description of the problem
            
        Returns:
            Dictionary containing solution details and results
        """
        start_time = time.time()
        logger.info(f"Starting to solve problem: {problem_statement}")
        
        try:
            # Record the problem
            problem_id = str(uuid.uuid4())
            self.memory.add("problem_id", problem_id)
            self.memory.add("problem_statement", problem_statement)
            self.memory.add("start_time", start_time)
            
            # Phase 1: Problem analysis
            analysis_result = self._analyze_problem(problem_statement)
            
            # Phase 2: Solution planning
            plan_result = self._create_solution_plan(analysis_result)
            
            # Phase 3: Execute solution plan
            execution_result = self._execute_solution(plan_result)
            
            # Phase 4: Evaluate and summarize results
            summary = self._summarize_solution(
                problem_statement,
                analysis_result,
                plan_result,
                execution_result
            )
            
            # Save everything to memory
            self.memory.add("analysis_result", analysis_result)
            self.memory.add("plan_result", plan_result)
            self.memory.add("execution_result", execution_result)
            self.memory.add("summary", summary)
            self.memory.add("end_time", time.time())
            
            # Return the final result
            return {
                "problem_id": problem_id,
                "problem_statement": problem_statement,
                "formalized_problem": analysis_result.get("formalized_problem"),
                "solution_plan": plan_result.get("plan"),
                "execution_results": execution_result.get("results"),
                "summary": summary,
                "duration_seconds": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}", exc_info=True)
            self.memory.add("error", str(e))
            self.memory.add("end_time", time.time())
            
            return {
                "problem_statement": problem_statement,
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    def _analyze_problem(self, problem_statement: str) -> Dict[str, Any]:
        """Analyze and formalize the problem."""
        logger.info("Analyzing problem")
        
        # Create initial state for problem analysis workflow
        initial_state = ProblemState(problem_statement=problem_statement)
        
        # Run the problem analysis workflow
        problem_analysis_workflow = self.workflows.create_problem_analysis_workflow(
            formalize_problem_fn=self.reasoner.formalize_problem,
            identify_components_fn=self.reasoner.identify_components
        )
        
        final_state = problem_analysis_workflow.invoke(initial_state)
        
        return {
            "formalized_problem": final_state.formalized_problem,
            "variables": final_state.variables,
            "constraints": final_state.constraints,
            "domains": final_state.domains,
            "approaches": final_state.approaches
        }
    
    def _create_solution_plan(self, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a solution plan based on problem analysis."""
        logger.info("Creating solution plan")
        
        # Create initial state for planning workflow
        initial_state = PlanningState(formalized_problem=problem_analysis["formalized_problem"])
        
        # Run the planning workflow
        planning_workflow = self.workflows.create_planning_workflow(
            create_steps_fn=self.planner.create_steps,
            analyze_dependencies_fn=self.planner.analyze_dependencies,
            allocate_resources_fn=self.planner.allocate_resources
        )
        
        final_state = planning_workflow.invoke(initial_state)
        
        return {
            "plan": {
                "steps": final_state.plan_steps,
                "dependencies": final_state.dependencies,
                "resources": final_state.resources_needed
            }
        }
    
    def _execute_solution(self, plan_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the solution plan."""
        logger.info("Executing solution plan")
        
        # Create initial state for execution workflow
        initial_state = ExecutionState(plan=plan_result["plan"])
        
        # Run the execution workflow
        execution_workflow = self.workflows.create_execution_workflow(
            prepare_step_fn=self.executor.prepare_step,
            execute_step_fn=self.executor.execute_step,
            handle_error_fn=self.executor.handle_error,
            check_completion_fn=self.executor.check_completion
        )
        
        final_state = execution_workflow.invoke(initial_state)
        
        return {
            "status": final_state.status,
            "results": final_state.results,
            "error": final_state.error
        }
    
    def _summarize_solution(self, 
                           problem_statement: str,
                           analysis_result: Dict[str, Any],
                           plan_result: Dict[str, Any],
                           execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive summary of the solution."""
        logger.info("Summarizing solution")
        
        summary = self.reasoner.generate_summary(
            problem_statement=problem_statement,
            formalized_problem=analysis_result.get("formalized_problem", {}),
            solution_plan=plan_result.get("plan", {}),
            execution_results=execution_result.get("results", {}),
            execution_status=execution_result.get("status", "unknown")
        )
        
        return summary
