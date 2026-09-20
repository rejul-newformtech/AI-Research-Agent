"""Re-export from app.schema.structured_output for backwards compatibility."""

from app.schema.structured_output import (
    CitationModel,
    HyDEPassageModel,
    MultiQueryExpansionModel,
    ResearchSynthesisModel,
    UserProfileContext,
)

__all__ = [
    "CitationModel",
    "HyDEPassageModel",
    "MultiQueryExpansionModel",
    "ResearchSynthesisModel",
    "UserProfileContext",
]
