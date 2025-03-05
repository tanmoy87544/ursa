"""
LangGraph-based workflows for scientific problem-solving.
Implements the core flow patterns using LangGraph for orchestration.
"""

import logging
from typing import Dict, Any, List, Callable, TypedDict, Annotated, Literal, Union, Optional

# LangGraph imports
from langgraph.graph import StateGraph, END

# Try to import MemorySaver, but don't fail if it doesn't exist
try:
    from langgraph.checkpoint import MemorySaver
except ImportError:
    # Create a simple fallback if MemorySaver doesn't exist
    class MemorySaver:
        """Fallback MemorySaver implementation."""
        def __init__(self):
            self.memories = {}
        
        def get(self, key):
            return self.memories.get(key)
        
        def put(self, key, value):
            self.memories[key] = value

# Use try/except for the ToolExecutor import
try:
    # Try the original import path
    from langgraph.prebuilt import ToolExecutor
except ImportError:
    # Define a fallback ToolExecutor class if it doesn't exist
    class ToolExecutor:
        """Fallback ToolExecutor implementation."""
        def __init__(self, tools=None):
            self.tools = tools or {}
        
        def invoke(self, tool_name, tool_input):
            """Execute a tool by name."""
            if tool_name not in self.tools:
                raise ValueError(f"Tool {tool_name} not found")
            return self.tools[tool_name](**tool_input)

# Pydantic for state typing
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Define state types
class ProblemState(BaseModel):
    """State for problem analysis workflow"""
    problem_statement: str = Field(description="Original problem statement")
    formalized_problem: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured problem representation")
    variables: List[str] = Field(
        default_factory=list, description="Key variables identified in the problem")
    constraints: List[str] = Field(
        default_factory=list, description="Constraints identified in the problem")
    domains: List[str] = Field(
        default_factory=list, description="Scientific domains relevant to the problem")
    approaches: List[str] = Field(
        default_factory=list, description="Potential scientific approaches")
    phase: Literal["initial", "formalizing", "completed"] = Field(
        default="initial", description="Current phase of problem analysis")

class PlanningState(BaseModel):
    """State for solution planning workflow"""
    formalized_problem: Dict[str, Any] = Field(description="Formalized problem representation")
    plan_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Ordered steps in the solution plan")
    current_step_index: int = Field(default=0, description="Index of current step being developed")
    dependencies: Dict[str, List[str]] = Field(
        default_factory=dict, description="Dependencies between steps")
    resources_needed: Dict[str, List[str]] = Field(
        default_factory=dict, description="Resources needed for each step")
    phase: Literal["initial", "step_planning", "dependency_analysis", "resource_allocation", "completed"] = Field(
        default="initial", description="Current phase of planning")

"""
Update to the ExecutionState class to include debugging information.
"""

class ExecutionState(BaseModel):
    """State for plan execution workflow"""
    plan: Dict[str, Any] = Field(description="The solution plan to execute")
    current_step_index: int = Field(default=0, description="Index of step being executed")
    results: Dict[str, Any] = Field(
        default_factory=dict, description="Results from executed steps")
    status: Literal["pending", "in_progress", "failed", "completed"] = Field(
        default="pending", description="Overall execution status")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    phase: Literal["setup", "execution", "error_handling", "completed"] = Field(
        default="setup", description="Current phase of execution")
    debug_attempts: int = Field(default=0, description="Number of debug attempts for current step")
    debug_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of debugging attempts")

class ScientificWorkflows:
    """Factory for creating scientific workflow graphs using LangGraph."""
    
    def __init__(self, tools: Dict[str, Any] = None):
        """
        Initialize the workflow factory.
        
        Args:
            tools: Dictionary of tools available for workflows
        """
        self.tools = tools or {}
        self.tool_executor = ToolExecutor(self.tools) if self.tools else None
    
    def create_problem_analysis_workflow(self, 
                                         formalize_problem_fn: Callable,
                                         identify_components_fn: Callable) -> StateGraph:
        """
        Create a workflow for analyzing and formalizing scientific problems.
        
        Args:
            formalize_problem_fn: Function to formalize the problem
            identify_components_fn: Function to identify problem components
            
        Returns:
            A configured StateGraph for problem analysis
        """
        # Define the nodes in our graph
        def formalize_problem(state: ProblemState) -> ProblemState:
            """Convert raw problem statement to formalized representation"""
            logger.info("Formalizing problem")
            problem_statement = state.problem_statement
            formalized = formalize_problem_fn(problem_statement)
            return state.model_copy(update={"formalized_problem": formalized, "phase": "formalizing"})
        
        def identify_components(state: ProblemState) -> ProblemState:
            """Identify variables, constraints, domains, and approaches"""
            logger.info("Identifying problem components")
            formalized = state.formalized_problem
            components = identify_components_fn(formalized)
            
            return state.model_copy(update={
                "variables": components.get("variables", []),
                "constraints": components.get("constraints", []),
                "domains": components.get("domains", []),
                "approaches": components.get("approaches", []),
                "phase": "completed"
            })
        
        # Create the graph
        workflow = StateGraph(ProblemState)
        
        # Add nodes
        workflow.add_node("formalize_problem", formalize_problem)
        workflow.add_node("identify_components", identify_components)
        
        # Add edges
        workflow.add_edge("formalize_problem", "identify_components")
        workflow.add_edge("identify_components", END)
        
        # Set the entry point
        workflow.set_entry_point("formalize_problem")
        
        # Compile the graph
        return workflow.compile()
    
    def create_planning_workflow(self,
                                create_steps_fn: Callable,
                                analyze_dependencies_fn: Callable,
                                allocate_resources_fn: Callable) -> StateGraph:
        """
        Create a workflow for planning the solution.
        
        Args:
            create_steps_fn: Function to create plan steps
            analyze_dependencies_fn: Function to analyze dependencies
            allocate_resources_fn: Function to allocate resources
            
        Returns:
            A configured StateGraph for solution planning
        """
        # Define the nodes
        def create_steps(state: PlanningState) -> PlanningState:
            """Create the initial solution steps"""
            logger.info("Creating solution steps")
            steps = create_steps_fn(state.formalized_problem)
            return state.model_copy(update={"plan_steps": steps, "phase": "step_planning"})
        
        def analyze_dependencies(state: PlanningState) -> PlanningState:
            """Analyze dependencies between steps"""
            logger.info("Analyzing step dependencies")
            dependencies = analyze_dependencies_fn(state.plan_steps)
            return state.model_copy(update={"dependencies": dependencies, "phase": "dependency_analysis"})
        
        def allocate_resources(state: PlanningState) -> PlanningState:
            """Allocate resources to steps"""
            logger.info("Allocating resources to steps")
            resources = allocate_resources_fn(state.plan_steps, state.dependencies)
            return state.model_copy(update={"resources_needed": resources, "phase": "resource_allocation"})
        
        def finalize_plan(state: PlanningState) -> PlanningState:
            """Finalize the plan"""
            logger.info("Finalizing plan")
            return state.model_copy(update={"phase": "completed"})
        
        # Create the graph
        workflow = StateGraph(PlanningState)
        
        # Add nodes
        workflow.add_node("create_steps", create_steps)
        workflow.add_node("analyze_dependencies", analyze_dependencies)
        workflow.add_node("allocate_resources", allocate_resources)
        workflow.add_node("finalize_plan", finalize_plan)
        
        # Add edges
        workflow.add_edge("create_steps", "analyze_dependencies")
        workflow.add_edge("analyze_dependencies", "allocate_resources")
        workflow.add_edge("allocate_resources", "finalize_plan")
        workflow.add_edge("finalize_plan", END)
        
        # Set the entry point
        workflow.set_entry_point("create_steps")
        
        # Compile the graph
        return workflow.compile()
    
    def create_execution_workflow(self,
                                 prepare_step_fn: Callable,
                                 execute_step_fn: Callable,
                                 handle_error_fn: Callable,
                                 check_completion_fn: Callable) -> StateGraph:
        """
        Create a workflow for executing the solution plan.
        
        Args:
            prepare_step_fn: Function to prepare a step for execution
            execute_step_fn: Function to execute a step
            handle_error_fn: Function to handle execution errors
            check_completion_fn: Function to check if execution is complete
            
        Returns:
            A configured StateGraph for plan execution
        """
        # Define the nodes
        def prepare_step(state: ExecutionState) -> ExecutionState:
            """Prepare the current step for execution"""
            logger.info(f"Preparing step {state.current_step_index + 1}/{len(state.plan['steps'])}")
            prepared_state = prepare_step_fn(state.plan, state.current_step_index, state.results)
            return state.model_copy(update={
                "status": "in_progress",
                "phase": "execution"
            })
        
        def execute_step(state: ExecutionState) -> ExecutionState:
            """Execute the current step"""
            try:
                logger.info(f"Executing step {state.current_step_index + 1}/{len(state.plan['steps'])}")
                current_step = state.plan["steps"][state.current_step_index]
                step_result = execute_step_fn(current_step, state.results)
                
                # Update results with this step's output
                updated_results = state.results.copy()
                updated_results[current_step["id"]] = step_result
                
                return state.model_copy(update={
                    "results": updated_results
                })
            except Exception as e:
                logger.error(f"Error executing step: {str(e)}")
                return state.model_copy(update={
                    "status": "failed",
                    "error": str(e),
                    "phase": "error_handling"
                })
        
        def handle_error(state: ExecutionState) -> ExecutionState:
            """Handle execution errors"""
            logger.info("Handling execution error")
            updated_state = handle_error_fn(
                state.plan, 
                state.current_step_index, 
                state.error, 
                state.results
            )
            
            # If error was resolved, move back to execution
            if updated_state.get("error_resolved", False):
                return state.model_copy(update={
                    "status": "in_progress",
                    "error": None,
                    "phase": "execution"
                })
            else:
                # Error couldn't be resolved, mark as failed
                return state.model_copy(update={
                    "status": "failed",
                    "phase": "completed"
                })
        
        def check_completion(state: ExecutionState) -> ExecutionState: # -> Dict[str, Any]:
            """Check if execution is complete and advance step if not"""
            logger.info("Checking completion status")
            is_complete, next_step = check_completion_fn(
                state.plan, 
                state.current_step_index, 
                state.results
            )
            
            if is_complete:
                logger.info("Execution completed successfully")
                return state.model_copy(update={"status": "completed","phase": "completed"})
            else:
                logger.info(f"Moving to next step: {next_step}")
                return state.model_copy(update={"current_step_index": next_step, "phase": "setup"})
        
        # Define how to route between nodes
        def route_from_check_completion(state: ExecutionState) -> str:
            """Route based on execution status"""
            if state.phase == "completed":
                return "END"
            else:
                return "prepare_step"
        
        def route_from_execute(state: ExecutionState) -> str:
            """Route based on execution status"""
            if state.status == "failed":
                return "handle_error"
            else:
                return "check_completion"
        
        def route_from_handle_error(state: ExecutionState) -> str:
            """Route based on error handling result"""
            if state.phase == "execution":  # Error was resolved
                return "execute_step"
            else:  # Error couldn't be resolved
                return "END"
        
        # Create the graph
        workflow = StateGraph(ExecutionState)
        
        # Add nodes
        workflow.add_node("prepare_step", prepare_step)
        workflow.add_node("execute_step", execute_step)
        workflow.add_node("handle_error", handle_error)
        workflow.add_node("check_completion", check_completion)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "execute_step",
            route_from_execute,
            {
                "handle_error": "handle_error",
                "check_completion": "check_completion"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            route_from_handle_error,
            {
                "execute_step": "execute_step",
                "check_completion": "check_completion"
            }
        )
        
        workflow.add_conditional_edges(
            "check_completion",
            route_from_check_completion,
            {
                "prepare_step": "prepare_step",
                "END": END
            }
        )
        
        # Add standard edges
        workflow.add_edge("prepare_step", "execute_step")
        
        # Set the entry point
        workflow.set_entry_point("prepare_step")
        
        # Compile the graph
        return workflow.compile()
