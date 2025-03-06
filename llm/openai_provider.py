"""
OpenAI LLM Provider for scientific agent framework.
Provides a uniform interface for interactions with OpenAI-served LLMs.
"""

import re
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Union

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class OpenAIProvider:
    """Provides access to LLMs through the OpenAI API."""
    
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 temperature: float = 0.6,
                 max_tokens: int = 4096,
                 system_prompt: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_name: Name of the model (e.g., "llama3", "mistral")
            base_url: OpenAI API base URL
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt to set model behavior
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        #### Verify connection
        #### self._verify_connection()
    
    #### def _verify_connection(self) -> None:
    ####     """Verify that we can connect to the OpenAI service."""
    ####     try:
    ####         response = requests.get(f"{self.base_url}/api/version")
    ####         if response.status_code == 200:
    ####             version_info = response.json()
    ####             logger.info(f"Connected to Ollama version: {version_info.get('version', 'unknown')}")
    ####         else:
    ####             logger.warning(f"Connected to Ollama but received unexpected status: {response.status_code}")
    ####     except Exception as e:
    ####         logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
    ####         raise ConnectionError(f"Cannot connect to Ollama service: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using the OpenAI API.
        
        Args:
            prompt: The input prompt for text generation
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Generated text as a string
        """
        # Merge default parameters with any overrides
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False
        }
        
        # Add system prompt if provided
        if self.system_prompt or kwargs.get("system_prompt"):
            prompt = kwargs.get("system_prompt", self.system_prompt) + prompt
        
        try:
            requests = ChatOpenAI(
                           model=self.model_name,
                           temperature=kwargs.get("temperature", self.temperature),
                           max_tokens=kwargs.get("max_tokens", self.max_tokens),
                           timeout=None,
                           max_retries=2)

            response = requests.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
    
    def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text and parse it as JSON.
        
        Args:
            prompt: The input prompt
            json_schema: Optional JSON schema description to include in the prompt
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON as a dictionary
        """
        # Add JSON formatting instructions to the prompt
        if json_schema:
            schema_str = json.dumps(json_schema, indent=2)
            formatted_prompt = f"{prompt}\n\nPlease format your response as a valid JSON object matching this schema:\n{schema_str}\n\nResponse (JSON format only):"
        else:
            formatted_prompt = f"{prompt}\n\nPlease format your response as a valid JSON object.\n\nResponse (JSON format only):"
        
        # Make multiple attempts to get valid JSON (LLMs sometimes struggle with this)
        max_attempts = kwargs.get("max_json_attempts", 3)
        
        for attempt in range(max_attempts):
            try:
                result = self.generate(formatted_prompt, **kwargs)
                
                # Try to extract JSON if it's surrounded by markdown code blocks or other text
                parsed_result = self._extract_json(result)
                return parsed_result
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON (attempt {attempt+1}/{max_attempts})")
                
                if attempt == max_attempts - 1:
                    logger.error(f"All attempts to parse JSON failed. Last response: {result}")
                    raise ValueError("Failed to get valid JSON response from LLM")
        
        # Should never reach here due to the exception in the loop
        return {}

    def _extract_json(self, text: str):
        """
        Extract a JSON object or array from text that might contain markdown or other content.
        
        The function attempts three strategies:
          1. Extract JSON from a markdown code block labeled as JSON.
          2. Extract JSON from any markdown code block.
          3. Use bracket matching to extract a JSON substring starting with '{' or '['.
        
        Returns:
            A Python object parsed from the JSON string (dict or list).
        
        Raises:
            ValueError: If no valid JSON is found.
        """
        # Approach 1: Look for a markdown code block specifically labeled as JSON.
        labeled_block = re.search(r'```json\s*([\[{].*?[\]}])\s*```', text, re.DOTALL)
        if labeled_block:
            json_str = labeled_block.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Fall back to the next approach if parsing fails.
                pass
    
        # Approach 2: Look for any code block delimited by triple backticks.
        generic_block = re.search(r'```(.*?)```', text, re.DOTALL)
        if generic_block:
            json_str = generic_block.group(1).strip()
            if json_str.startswith('{') or json_str.startswith('['):
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    
        # Approach 3: Attempt to extract JSON using bracket matching.
        # Find the first occurrence of either '{' or '['.
        first_obj = text.find('{')
        first_arr = text.find('[')
        if first_obj == -1 and first_arr == -1:
            raise ValueError("No JSON object or array found in the text.")
    
        # Determine which bracket comes first.
        if first_obj == -1:
            start = first_arr
            open_bracket = '['
            close_bracket = ']'
        elif first_arr == -1:
            start = first_obj
            open_bracket = '{'
            close_bracket = '}'
        else:
            if first_obj < first_arr:
                start = first_obj
                open_bracket = '{'
                close_bracket = '}'
            else:
                start = first_arr
                open_bracket = '['
                close_bracket = ']'
        
        # Bracket matching: find the matching closing bracket.
        depth = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == open_bracket:
                depth += 1
            elif text[i] == close_bracket:
                depth -= 1
                if depth == 0:
                    end = i
                    break
    
        if end is None:
            raise ValueError("Could not find matching closing bracket for JSON content.")
    
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError("Extracted content is not valid JSON.") from e


