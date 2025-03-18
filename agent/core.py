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
from .memory                      import Memory
from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI

from ..llm.langgraph_workflows import ScientificWorkflows

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
        # # Load configuration
        # self.config = self._load_config(config_path)
        
        # Set up workspace
        self.workspace_dir = workspace_dir or os.path.join(os.getcwd(), "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Initialize LLM
        self.llm_provider = llm_provider
        self.model_name   = model_name
        self.llm          = self._initialize_llm()
        
        # Initialize workflows
        self.memory       = Memory()
        self.workflows    = self._initialize_workflows()
        
        # Generate a unique ID for this agent instance
        self.agent_id = str(uuid.uuid4())
        logger.info(f"Scientific agent initialized with ID: {self.agent_id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "llm": {
                "temperature": 0.6,
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
            model = ChatOllama(
                model       = self.model_name, # "llama3.1:8b",
                max_tokens  = 10000,
                timeout     = None,
                max_retries = 2
                )
        elif self.llm_provider.lower() == "openai":
            model = ChatOpenAI(
                model       = self.model_name,
                max_tokens  = 10000,
                timeout     = None,
                max_retries = 2
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
            
            analysis_result = self.workflow(problem_statement)
                        
            # Save everything to memory
            self.memory.add("analysis_result", analysis_result)
            self.memory.add("end_time", time.time())
       
            self.memory.memory_file = "memory.json"
            self.memory._save_memory()
            
            # Return the final result
            return {
                "problem_id": problem_id,
                "problem_statement": problem_statement,
                "analysis_result": analysis_result.get("formalized_problem"),
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
