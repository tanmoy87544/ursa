"""
Ollama LLM Provider for scientific agent framework.
Provides a uniform interface for interactions with Ollama-served LLMs.
"""

import json
import logging
import requests
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class OllamaProvider:
    """Provides access to LLMs through the Ollama API."""
    
    def __init__(self, 
                 model_name: str = "llama3",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1,
                 max_tokens: int = 4096,
                 system_prompt: Optional[str] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            model_name: Name of the model (e.g., "llama3", "mistral")
            base_url: Ollama API base URL
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt to set model behavior
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.api_url = f"{self.base_url}/api/generate"
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self) -> None:
        """Verify that we can connect to the Ollama service."""
        try:
            response = requests.get(f"{self.base_url}/api/version")
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"Connected to Ollama version: {version_info.get('version', 'unknown')}")
            else:
                logger.warning(f"Connected to Ollama but received unexpected status: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            raise ConnectionError(f"Cannot connect to Ollama service: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using the Ollama API.
        
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
            params["system"] = kwargs.get("system_prompt", self.system_prompt)
        
        try:
            response = requests.post(self.api_url, json=params)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {str(e)}")
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
                json_str = self._extract_json(result)
                parsed_result = json.loads(json_str)
                return parsed_result
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON (attempt {attempt+1}/{max_attempts})")
                
                if attempt == max_attempts - 1:
                    logger.error(f"All attempts to parse JSON failed. Last response: {result}")
                    raise ValueError("Failed to get valid JSON response from LLM")
        
        # Should never reach here due to the exception in the loop
        return {}
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or other content."""
        # Try to extract from code blocks first
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0]
                return json_part.strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        
        # If no code blocks, return the original text
        return text.strip()
