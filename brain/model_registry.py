from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelCapabilities:
    """Explicit capability flags for a model/provider."""
    supports_text: bool = True
    supports_structured_json: bool = False
    supports_streaming: bool = False
    supports_cancellation: bool = False
    supports_tools: bool = False
    supports_system_prompt: bool = True
    context_tokens: int = 8192
    output_tokens: int = 4096
    modality: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class ModelHealth:
    """Current availability and health status."""
    is_available: bool = True
    current_health_status: str = "healthy"


@dataclass
class ModelMetadata:
    """Canonical model descriptor."""
    provider: str
    model_id: str
    capabilities: ModelCapabilities
    health: ModelHealth = field(default_factory=ModelHealth)


class ModelRegistry:
    """
    A canonical registry for tracking model and provider capabilities.
    Enables MoERouter to answer: "Can this model satisfy this request?"
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelMetadata] = {}

    def register(self, metadata: ModelMetadata) -> None:
        """Register a model's metadata."""
        self._models[metadata.provider] = metadata

    def get_metadata(self, provider_name: str) -> Optional[ModelMetadata]:
        """Retrieve metadata for a given provider."""
        return self._models.get(provider_name)

    def find_eligible_providers(
        self,
        require_text: bool = False,
        require_structured_json: bool = False,
        require_streaming: bool = False,
        require_cancellation: bool = False,
        require_tools: bool = False,
    ) -> list[str]:
        """
        Returns a list of provider names that satisfy the required capability matrix.
        """
        eligible = []
        for provider, metadata in self._models.items():
            if not metadata.health.is_available:
                continue

            cap = metadata.capabilities
            if require_text and not cap.supports_text:
                continue
            if require_structured_json and not cap.supports_structured_json:
                continue
            if require_streaming and not cap.supports_streaming:
                continue
            if require_cancellation and not cap.supports_cancellation:
                continue
            if require_tools and not cap.supports_tools:
                continue
            
            eligible.append(provider)
        
        return eligible

# Global registry instance
registry = ModelRegistry()
