"""Template asset management for Lacuna.

This module provides template registry and loading with TemplateFlow integration.
"""

from lacuna.assets.templates.loader import is_template_cached, load_template
from lacuna.assets.templates.registry import TemplateMetadata, list_templates

__all__ = [
    "TemplateMetadata",
    "list_templates",
    "load_template",
    "is_template_cached",
]
