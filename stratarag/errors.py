"""Exception hierarchy for stratarag."""


class StrataRAGError(Exception):
    """Base class for all stratarag errors."""


class ConfigurationError(StrataRAGError):
    """Raised when a component is configured incorrectly."""


class MissingDependencyError(StrataRAGError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(self, package: str, extra: str, feature: str):
        self.package = package
        self.extra = extra
        self.feature = feature
        super().__init__(
            f"{feature} requires the optional package '{package}'. "
            f"Install it with: pip install stratarag[{extra}]"
        )


class StoreError(StrataRAGError):
    """Raised on vector store failures."""


class ToolError(StrataRAGError):
    """Raised when a tool fails or is misused."""


class GenerationError(StrataRAGError):
    """Raised when the LLM provider fails."""
