"""
LLM Interface Module
Supports multiple LLM providers with deterministic mode and robust output parsing.
"""

import os
import re
import json
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class LLMResponse:
    """Represents an LLM response with metadata."""
    raw_response: str
    extracted_answer: Any
    provider: str
    model: str
    prompt: str
    timestamp: str
    latency: float
    tokens_used: Optional[int] = None
    

class LLMInterface:
    """Interface for multiple LLM providers with robust error handling."""
    
    def __init__(self, provider: str = "openai", model: str = None,
                 temperature: float = 0.0, max_tokens: int = 2000,
                 timeout: int = 60, max_retries: int = 3,
                 retry_delay: int = 2, log_dir: str = "logs"):
        """
        Initialize LLM interface.
        
        Args:
            provider: "openai" or "anthropic"
            model: Specific model to use
            temperature: Temperature for generation (0 = deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            log_dir: Directory for logging responses
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_dir = log_dir
        
        # Set up logging
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"LLMInterface-{provider}")
        
        # Initialize provider
        if self.provider == "openai":
            if openai is None:
                raise ImportError("openai package not installed")
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4"
        elif self.provider == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic package not installed")
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-3-opus-20240229"
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.logger.info(f"Initialized {self.provider} with model {self.model}")
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Query the LLM with retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            LLMResponse object
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                if self.provider == "openai":
                    response = self._query_openai(prompt, system_prompt)
                elif self.provider == "anthropic":
                    response = self._query_anthropic(prompt, system_prompt)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                
                latency = time.time() - start_time
                
                # Extract answer from response
                extracted = self._extract_answer(response["text"])
                
                llm_response = LLMResponse(
                    raw_response=response["text"],
                    extracted_answer=extracted,
                    provider=self.provider,
                    model=self.model,
                    prompt=prompt,
                    timestamp=datetime.now().isoformat(),
                    latency=latency,
                    tokens_used=response.get("tokens")
                )
                
                # Log the response
                self._log_response(llm_response)
                
                return llm_response
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def _query_openai(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        """Query OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            seed=42  # For determinism
        )
        
        return {
            "text": response.choices[0].message.content,
            "tokens": response.usage.total_tokens if response.usage else None
        }
    
    def _query_anthropic(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        """Query Anthropic API."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        
        return {
            "text": response.content[0].text,
            "tokens": response.usage.input_tokens + response.usage.output_tokens if response.usage else None
        }
    
    def _extract_answer(self, response: str) -> Any:
        """
        Extract the answer from LLM response using multiple strategies.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Extracted answer (can be int, list, string)
        """
        # Clean up the response
        response = response.strip()
        
        # Strategy 1: Look for explicit answer markers
        answer_patterns = [
            r"(?:final answer|answer|result)(?:\s*is)?:\s*([^\n]+)",
            r"(?:the answer is|equals?)\s*([^\n]+)",
            r"^answer:\s*([^\n]+)",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                response = match.group(1).strip()
                break
        
        # Strategy 2: Try to parse as a number
        number_match = re.search(r'-?\d+(?:\.\d+)?', response)
        if number_match:
            num_str = number_match.group()
            try:
                # Try integer first
                if '.' not in num_str:
                    return int(num_str)
                else:
                    return float(num_str)
            except ValueError:
                pass
        
        # Strategy 3: Try to parse as a list of numbers
        list_match = re.search(r'\[([\d,\s]+)\]', response)
        if list_match:
            try:
                numbers = [int(x.strip()) for x in list_match.group(1).split(',')]
                return numbers
            except ValueError:
                pass
        
        # Strategy 4: Comma-separated numbers
        if re.match(r'^[\d,\s]+$', response):
            try:
                numbers = [int(x.strip()) for x in response.split(',') if x.strip()]
                return numbers
            except ValueError:
                pass
        
        # Strategy 5: Look for quoted strings or parenthesized answers
        quote_match = re.search(r'["\']([^"\']+)["\']', response)
        if quote_match:
            return quote_match.group(1)
        
        paren_match = re.search(r'\(([^)]+)\)', response)
        if paren_match:
            return paren_match.group(0)  # Include parentheses
        
        # Strategy 6: Look for winning/losing for game theory
        if re.search(r'\b(winning|losing)\b', response, re.IGNORECASE):
            match = re.search(r'\b(winning|losing)\b', response, re.IGNORECASE)
            return match.group(1).lower()
        
        # Strategy 7: Extract last line if multi-line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # Try to extract number from last line
            num_match = re.search(r'-?\d+(?:\.\d+)?', last_line)
            if num_match:
                num_str = num_match.group()
                try:
                    return int(num_str) if '.' not in num_str else float(num_str)
                except ValueError:
                    pass
            return last_line
        
        # Fallback: return cleaned response
        return response
    
    def _log_response(self, response: LLMResponse):
        """Log the LLM response to file."""
        log_file = os.path.join(
            self.log_dir,
            f"{self.provider}_{self.model}_responses.jsonl"
        )
        
        log_entry = {
            "timestamp": response.timestamp,
            "provider": response.provider,
            "model": response.model,
            "prompt": response.prompt,
            "raw_response": response.raw_response,
            "extracted_answer": str(response.extracted_answer),
            "latency": response.latency,
            "tokens_used": response.tokens_used
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def batch_query(self, prompts: List[str], 
                   system_prompt: Optional[str] = None) -> List[LLMResponse]:
        """
        Query multiple prompts in batch.
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt for all queries
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for prompt in prompts:
            response = self.query(prompt, system_prompt)
            responses.append(response)
        return responses


class LLMFactory:
    """Factory for creating LLM interfaces."""
    
    @staticmethod
    def create(provider: str, model: str = None, **kwargs) -> LLMInterface:
        """
        Create an LLM interface.
        
        Args:
            provider: Provider name
            model: Model name
            **kwargs: Additional arguments for LLMInterface
            
        Returns:
            LLMInterface instance
        """
        return LLMInterface(provider=provider, model=model, **kwargs)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> LLMInterface:
        """Create LLM interface from configuration dictionary."""
        return LLMInterface(
            provider=config.get("provider", "openai"),
            model=config.get("model"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 2000),
            timeout=config.get("timeout", 60),
            max_retries=config.get("max_retries", 3),
            retry_delay=config.get("retry_delay", 2),
            log_dir=config.get("log_dir", "logs")
        )


# if __name__ == "__main__":
#     # Test LLM interface
#     import logging
#     logging.basicConfig(level=logging.INFO)
    
#     print("=== Testing LLM Interface ===\n")
    
#     # Test with a simple math problem
#     test_prompt = "What is 2 + 2? Provide only the numerical answer."
    
#     # Test OpenAI (if available)
#     if os.getenv("OPENAI_API_KEY"):
#         print("Testing OpenAI...")
#         try:
#             llm = LLMInterface(provider="openai", model="gpt-3.5-turbo", temperature=0.0)
#             response = llm.query(test_prompt)
#             print(f"Raw Response: {response.raw_response}")
#             print(f"Extracted Answer: {response.extracted_answer}")
#             print(f"Latency: {response.latency:.2f}s")
#             print(f"Tokens: {response.tokens_used}\n")
#         except Exception as e:
#             print(f"OpenAI test failed: {e}\n")
#     else:
#         print("OPENAI_API_KEY not set, skipping OpenAI test\n")
    
#     # Test Anthropic (if available)
#     if os.getenv("ANTHROPIC_API_KEY"):
#         print("Testing Anthropic...")
#         try:
#             llm = LLMInterface(provider="anthropic", temperature=0.0)
#             response = llm.query(test_prompt)
#             print(f"Raw Response: {response.raw_response}")
#             print(f"Extracted Answer: {response.extracted_answer}")
#             print(f"Latency: {response.latency:.2f}s")
#             print(f"Tokens: {response.tokens_used}\n")
#         except Exception as e:
#             print(f"Anthropic test failed: {e}\n")
#     else:
#         print("ANTHROPIC_API_KEY not set, skipping Anthropic test\n")
    
#     # Test extraction
#     print("Testing answer extraction...")
#     test_responses = [
#         "The answer is 42",
#         "After calculating, we get: 123",
#         "Final answer: [1, 2, 3, 4, 5]",
#         "The result is winning",
#         "2, 3, 5, 7, 11",
#         "(4, 4)",
#     ]
    
#     llm = LLMInterface(provider="openai")  # Just for extraction testing
#     for test_resp in test_responses:
#         extracted = llm._extract_answer(test_resp)
#         print(f"Input: '{test_resp}' -> Extracted: {extracted} (type: {type(extracted).__name__})")
