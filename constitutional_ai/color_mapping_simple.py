"""
Simple Color Mapping System for Constitutional Genomes
Simplified version that works with current trait system.
"""

import math
from typing import Dict, Any, Tuple

def traits_to_simple_color(traits: Dict[str, Any]) -> str:
    """
    Convert traits to a simple hex color using hash-based approach.
    
    Args:
        traits: Dictionary of trait names to values
        
    Returns:
        Hex color string (e.g., "#FF5733")
    """
    # Convert traits to a stable string representation
    trait_str = str(sorted(traits.items()))
    
    # Use hash to generate RGB values
    hash_val = abs(hash(trait_str))
    
    # Extract RGB components from hash
    r = (hash_val >> 16) & 0xFF
    g = (hash_val >> 8) & 0xFF  
    b = hash_val & 0xFF
    
    # Ensure colors are not too dark (minimum brightness)
    r = max(r, 64)
    g = max(g, 64)
    b = max(b, 64)
    
    return f"#{r:02X}{g:02X}{b:02X}"

def get_simple_color_description(hex_color: str) -> str:
    """Get a simple description of the color."""
    # Parse RGB from hex
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    
    # Determine dominant color
    if r > g and r > b:
        return "Red-dominant"
    elif g > r and g > b:
        return "Green-dominant"
    elif b > r and b > g:
        return "Blue-dominant"
    else:
        return "Balanced"

def traits_to_hsv_simple(traits: Dict[str, Any]) -> Tuple[float, float, float]:
    """Convert traits to HSV using simple approach."""
    hex_color = traits_to_simple_color(traits)
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    
    # Convert RGB to HSV
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val
    
    # Hue calculation
    if delta == 0:
        h = 0
    elif max_val == r:
        h = 60 * (((g - b) / delta) % 6)
    elif max_val == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    
    # Saturation calculation
    s = 0 if max_val == 0 else delta / max_val
    
    # Value calculation
    v = max_val
    
    return (h, s, v)