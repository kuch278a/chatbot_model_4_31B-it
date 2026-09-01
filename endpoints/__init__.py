"""
Endpoints package proxy to src.endpoints.
Provides backward compatibility for root-level endpoint imports.
"""

from src.endpoints.talk import talk_bp

__all__ = ["talk_bp"]

