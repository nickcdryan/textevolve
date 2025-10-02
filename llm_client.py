#!/usr/bin/env python3
"""
llm_client.py - Abstract base class and implementations for LLM clients

This module provides a clean abstraction for LLM API calls, allowing easy switching
between different LLM providers (Gemini, Claude, OpenAI, etc.) and models.

Usage:
    # Create a client
    client = LLMClientFactory.create_orchestrator_client()
    
    # Generate response
    response = client.generate(
        prompt="What is 2+2?",
        system_instruction="You are a helpful math tutor"
    )
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model: str, **config):
        """
        Initialize the LLM client
        
        Args:
            model: Model identifier (e.g., "gemini-2.5-flash")
            **config: Additional configuration parameters
        """
        self.model = model
        self.config = config
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                system_instruction: Optional[str] = None,
                **kwargs) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name being used"""
        pass


class GeminiClient(LLMClient):
    """Gemini LLM client implementation"""
    
    def __init__(self, model: str = "gemini-2.5-flash", **config):
        """
        Initialize Gemini client
        
        Args:
            model: Gemini model name (default: gemini-2.5-flash)
            **config: Additional configuration (api_key, etc.)
        """
        super().__init__(model, **config)
        
        # Initialize the Gemini client
        try:
            from google import genai
            from google.genai import types
            
            # Store imports for later use
            self.genai = genai
            self.types = types
            
            # Get API key from config or environment
            api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in config or environment")
            
            self.client = genai.Client(api_key=api_key)
            
        except ImportError as e:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-generativeai"
            ) from e
    
    def generate(self, 
                prompt: str, 
                system_instruction: Optional[str] = None,
                thinking_budget: int = 0,
                **kwargs) -> str:
        """
        Generate response from Gemini
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            thinking_budget: Thinking budget (default: 0 to disable thinking)
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            Generated text response
        """
        try:
            # Build configuration
            config = None
            if system_instruction or thinking_budget is not None:
                config_params = {}
                
                if system_instruction:
                    config_params["system_instruction"] = system_instruction
                
                # Add thinking config if specified
                config_params["thinking_config"] = self.types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
                
                config = self.types.GenerateContentConfig(**config_params)
            
            # Call the API
            response = self.client.models.generate_content(
                model=self.model,
                config=config,
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            print(error_msg)
            return f"Error: {str(e)}"
    
    def get_model_name(self) -> str:
        """Return the Gemini model name"""
        return self.model


class ClaudeClient(LLMClient):
    """Claude LLM client implementation (placeholder for future implementation)"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **config):
        super().__init__(model, **config)
        raise NotImplementedError("Claude client not yet implemented")
    
    def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError("Claude client not yet implemented")
    
    def get_model_name(self) -> str:
        return self.model


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation (placeholder for future implementation)"""
    
    def __init__(self, model: str = "gpt-4", **config):
        super().__init__(model, **config)
        raise NotImplementedError("OpenAI client not yet implemented")
    
    def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError("OpenAI client not yet implemented")
    
    def get_model_name(self) -> str:
        return self.model


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    # Default models for each role
    DEFAULT_ORCHESTRATOR_MODEL = "gemini-2.5-flash"
    DEFAULT_INFERENCE_MODEL = "gemini-2.5-flash"
    
    # Provider registry
    PROVIDERS = {
        "gemini": GeminiClient,
        "claude": ClaudeClient,
        "openai": OpenAIClient,
    }
    
    @classmethod
    def create_client(cls, 
                     provider: str = "gemini",
                     model: Optional[str] = None,
                     **config) -> LLMClient:
        """
        Create an LLM client
        
        Args:
            provider: Provider name (gemini, claude, openai)
            model: Model identifier (optional, uses provider default if not specified)
            **config: Additional configuration parameters
            
        Returns:
            LLMClient instance
        """
        provider = provider.lower()
        
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available providers: {list(cls.PROVIDERS.keys())}"
            )
        
        client_class = cls.PROVIDERS[provider]
        
        # Create client with or without model specified
        if model:
            return client_class(model=model, **config)
        else:
            return client_class(**config)
    
    @classmethod
    def create_orchestrator_client(cls, 
                                   model: Optional[str] = None,
                                   provider: Optional[str] = None,
                                   **config) -> LLMClient:
        """
        Create an orchestrator LLM client (for meta-level operations)
        
        Args:
            model: Model identifier (overrides env var and default)
            provider: Provider name (overrides env var, defaults to gemini)
            **config: Additional configuration parameters
            
        Returns:
            LLMClient instance for orchestrator use
        """
        # Priority: argument > environment variable > default
        model = model or os.environ.get("ORCHESTRATOR_LLM_MODEL") or cls.DEFAULT_ORCHESTRATOR_MODEL
        provider = provider or os.environ.get("ORCHESTRATOR_LLM_PROVIDER") or "gemini"
        
        return cls.create_client(provider=provider, model=model, **config)
    
    @classmethod
    def create_inference_client(cls,
                               model: Optional[str] = None,
                               provider: Optional[str] = None,
                               **config) -> LLMClient:
        """
        Create an inference LLM client (for script execution)
        
        Args:
            model: Model identifier (overrides env var and default)
            provider: Provider name (overrides env var, defaults to gemini)
            **config: Additional configuration parameters
            
        Returns:
            LLMClient instance for inference use
        """
        # Priority: argument > environment variable > default
        model = model or os.environ.get("INFERENCE_LLM_MODEL") or cls.DEFAULT_INFERENCE_MODEL
        provider = provider or os.environ.get("INFERENCE_LLM_PROVIDER") or "gemini"
        
        return cls.create_client(provider=provider, model=model, **config)


# Convenience function for backward compatibility
def create_gemini_client(model: str = "gemini-2.5-flash", **config) -> GeminiClient:
    """
    Create a Gemini client (convenience function)
    
    Args:
        model: Gemini model name
        **config: Additional configuration
        
    Returns:
        GeminiClient instance
    """
    return GeminiClient(model=model, **config)

