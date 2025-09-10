"""
Simple Color Mapping System for Constitutional Genomes
Simplified version that works with current trait system.
"""

from typing import Dict, Any, Tuple


def traits_to_simple_color(traits: Dict[str, Any]) -> str:
    """
    Convert traits to a simple hex color based on actual trait values.
    Colors reflect the agent's personality and capabilities.

    Args:
        traits: Dictionary of trait names to values

    Returns:
        Hex color string (e.g., "#FF5733")
    """
    # Map traits to RGB components based on their psychological meaning

    # Red component: Leadership, Autonomy, Risk-taking
    red_traits = ["Leadership", "Autonomy", "RiskTolerance", "InnovationDrive"]
    red_values = [traits.get(trait, 0) for trait in red_traits if trait in traits]
    if red_values:
        # Normalize to 0-1 (assuming most traits are 0-5 range)
        red_normalized = sum(min(1.0, val / 5.0) for val in red_values) / len(
            red_values
        )
        r = int(64 + (red_normalized * 191))  # Range: 64-255
    else:
        r = 128

    # Green component: Empathy, Cooperation, Emotional Intelligence
    green_traits = [
        "Empathy",
        "Cooperation",
        "EmotionalIntelligence",
        "Trustworthiness",
    ]
    green_values = [traits.get(trait, 0) for trait in green_traits if trait in traits]
    if green_values:
        green_normalized = sum(min(1.0, val / 5.0) for val in green_values) / len(
            green_values
        )
        g = int(64 + (green_normalized * 191))  # Range: 64-255
    else:
        g = 128

    # Blue component: Analytical thinking, Stability, Pattern Recognition
    blue_traits = [
        "CriticalThinking",
        "Stability",
        "PatternRecognition",
        "CausalReasoning",
    ]
    blue_values = [traits.get(trait, 0) for trait in blue_traits if trait in traits]
    if blue_values:
        blue_normalized = sum(min(1.0, val / 5.0) for val in blue_values) / len(
            blue_values
        )
        b = int(64 + (blue_normalized * 191))  # Range: 64-255
    else:
        b = 128

    # Ensure RGB values are within valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f"#{r:02X}{g:02X}{b:02X}"


def deterministic_hash(s: str) -> int:
    """
    Deterministic hash function that produces consistent results
    across sessions.

    Args:
        s: String to hash

    Returns:
        64-bit hash value
    """
    # DJB2 hash algorithm - deterministic and fast
    hash_val = 5381
    for char in s:
        # hash_val * 33 + ord(char)
        hash_val = ((hash_val << 5) + hash_val) + ord(char)
        hash_val &= 0xFFFFFFFFFFFFFFFF  # Keep it to 64 bits

    return hash_val


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
