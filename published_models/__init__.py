from published_models.base import BasePublishedModel

# Implementations. NOTE: add new implementations here and in factory.py.
from published_models.model_implementations.google_research_vit.google_research_vit import GoogleResearchViT
from published_models.model_implementations.facebook_deit.facebook_deit import FacebookDeiT
from published_models.model_implementations.swin.swin import Swin

from published_models.factory import get