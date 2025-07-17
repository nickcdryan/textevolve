from abc import ABC, abstractmethod

from clients import GeminiClient, LLMClient, OpenAIClient


class LLMProvider(ABC):

    @abstractmethod
    def build_client(self, model_name: str) -> LLMClient:
        return GeminiClient(model_name)



class GeminiProvider(LLMProvider):
    def build_client(self, model_name: str) -> GeminiClient:
        return GeminiClient(model_name)

class OpenAIProvider(LLMProvider):
    def build_client(self, model_name: str) -> OpenAIClient:
        return OpenAIClient(model_name)



class ProviderFactory:
    @staticmethod
    def get_provider(provider_type: str) -> LLMProvider:
        if provider_type == "gemini":
            return GeminiProvider()
        elif provider_type == "openai":
            return OpenAIProvider()
        elif provider_type == "huggingface":
            # TODO(jam): add in separate PR
            raise NotImplementedError("HF not yet supported")
        else:
            raise ValueError(f"Invalid provider type: {provider_type}")

    @staticmethod
    def get_client(provider_type: str, model_name: str) -> LLMClient:
        provider = ProviderFactory.get_provider(provider_type)
        return provider.build_client(model_name)