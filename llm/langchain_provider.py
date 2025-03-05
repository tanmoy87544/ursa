"""
LangChain LLM Provider for scientific agent framework.
Provides access to LLMs through LangChain's unified interface.
"""

import logging
from typing import Dict, Any, List, Optional, Union

# LangChain imports
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_community.llms.ollama import Ollama
from langchain.chains import LLMChain
from langchain.schema import OutputParserException
# from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

class LangChainProvider:
    """Provides access to LLMs through the LangChain library."""
    
    def __init__(self, 
                 model_name: str = "llama3",
                 provider: str = "ollama",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.6,
                 max_tokens: int = 4096,
                 system_prompt: Optional[str] = None):
        """
        Initialize the LangChain provider.
        
        Args:
            model_name: Name of the model
            provider: LLM provider type (currently only 'ollama' supported)
            base_url: API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
        """
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize and return the appropriate LangChain LLM."""
        if self.provider.lower() == "ollama":
            model_kwargs = {}
            if self.system_prompt:
                model_kwargs["system"] = self.system_prompt
                
            return Ollama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.base_url,
                max_tokens=self.max_tokens,
                model_kwargs=model_kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using LangChain.
        
        Args:
            prompt: The input prompt for text generation
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Generated text as a string
        """
        try:
            # Override defaults if needed
            if kwargs:
                temp_llm = self._initialize_llm()
                if "temperature" in kwargs:
                    temp_llm.temperature = kwargs["temperature"]
                if "max_tokens" in kwargs:
                    temp_llm.max_tokens = kwargs["max_tokens"]
                if "system_prompt" in kwargs:
                    if hasattr(temp_llm, "model_kwargs"):
                        temp_llm.model_kwargs["system"] = kwargs["system_prompt"]
                result = temp_llm.invoke(prompt)
            else:
                result = self.llm.invoke(prompt)
                
            return result
        except Exception as e:
            logger.error(f"Error generating text with LangChain: {str(e)}")
            raise
    
    def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate text and parse it as JSON using LangChain's output parsers.
        
        Args:
            prompt: The input prompt
            json_schema: JSON schema for the expected output
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON as a dictionary
        """
        try:
            # Create a parser based on the schema
            parser = JsonOutputParser()
            
            # Create the prompt template
            prompt_template = PromptTemplate(
                template="""
                {prompt}
                
                Return your response in JSON format according to the following schema:
                {schema}
                
                Response (JSON format only):
                """,
                input_variables=["prompt"],
                partial_variables={"schema": json_schema}
            )
            
            # Create and run the chain
            chain = prompt_template | self.llm | parser
            
            # Execute the chain
            result = chain.invoke({"prompt": prompt})
            return result
            
        except OutputParserException as e:
            logger.error(f"Error parsing LLM output as JSON: {str(e)}")
            # Try a fallback approach
            raw_response = self.generate(prompt)
            try:
                import json
                import re
                
                # Extract JSON if it's in a code block
                json_match = re.search(r"```json(.*?)```", raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = raw_response
                
                # Try to parse it
                parsed = json.loads(json_str)
                return parsed
            except Exception:
                raise ValueError("Failed to parse LLM output as JSON")
        except Exception as e:
            logger.error(f"Error in JSON generation: {str(e)}")
            raise
    
    def create_pydantic_chain(self, prompt_template: str, output_class: BaseModel) -> LLMChain:
        """
        Create a LangChain chain that outputs a specific Pydantic model.
        
        Args:
            prompt_template: Template for the prompt
            output_class: Pydantic model class to use for output parsing
            
        Returns:
            An initialized LLMChain
        """
        parser = PydanticOutputParser(pydantic_object=output_class)
        
        prompt = PromptTemplate(
            template=prompt_template + "\n{format_instructions}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt, output_parser=parser)
