"""Text utility functions for LLM response processing."""

import logging

logger = logging.getLogger(__name__)


def clean_markdown_json_response(text: str) -> str:
    """Remove markdown code block formatting from JSON responses.
    
    LLM models sometimes wrap JSON responses in markdown code blocks.
    This function removes those blocks while preserving the actual JSON content.
    
    Args:
        text: Raw text response that may contain markdown code blocks
        
    Returns:
        str: Cleaned text with markdown blocks removed
        
    Examples:
        >>> clean_markdown_json_response('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> clean_markdown_json_response('```\\n{"key": "value"}\\n```')
        '{"key": "value"}'
    """
    if not text:
        return text
    
    cleaned = text.strip()
    
    # Remove opening markdown blocks
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    
    # Remove closing markdown blocks
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    
    return cleaned
