"""
Color Mapping System for Constitutional Genomes
Derives HSV/RGB colors deterministically from trait value combinations.

This module converts an AI agent's complete trait profile into a unique color
signature that visually represents their personality and capabilities.
"""

import math
from typing import Dict, Any, Tuple


def normalize_trait_value(trait_name: str, trait_value: Any) -> float:
    """
    Normalize a trait value to 0.0-1.0 range for color mapping.

    Args:
        trait_name: Name of the trait
        trait_value: Current trait value

    Returns:
        Normalized value between 0.0 and 1.0
    """
    from .traits import COMPLETE_TRAIT_DEFINITIONS

    if trait_name not in COMPLETE_TRAIT_DEFINITIONS:
        return 0.5  # Default middle value

    trait_def = COMPLETE_TRAIT_DEFINITIONS[trait_name]

    if trait_def.allele_type == "numeric":
        min_val, max_val = trait_def.domain
        return max(0.0, min(1.0, (trait_value - min_val) / (max_val - min_val)))

    elif trait_def.allele_type == "categorical":
        # Use monotone ordering index
        if trait_value in trait_def.monotone_order:
            ordering_idx = trait_def.monotone_order.index(trait_value)
            max_idx = len(trait_def.monotone_order) - 1
            return ordering_idx / max_idx if max_idx > 0 else 0.0
        return 0.5

    elif trait_def.allele_type == "boolean":
        return 1.0 if trait_value else 0.0

    return 0.5


def traits_to_hue(traits: Dict[str, Any]) -> float:
    """
    Map trait combination to hue (0-360 degrees).

    Different trait combinations map to different color regions:
    - Cognitive traits (Perception, Memory, Expertise) -> Blue/Cyan region
    - Learning traits (LearningRate, Transfer) -> Green region
    - Behavioral traits (Attention, Risk, Innovation) -> Yellow/Orange region
    - Performance traits (Speed, Stability, Efficiency) -> Red/Magenta region
    - Social traits (Social, Communication) -> Violet region
    - Advanced traits (MetaLearning, Curiosity) -> Full spectrum blend
    """

    # Normalize all trait values
    normalized = {}
    for trait_name, trait_value in traits.items():
        normalized[trait_name] = normalize_trait_value(trait_name, trait_value)

    # Define trait category weights and hue ranges
    cognitive_traits = ["Perception", "WorkingMemory", "Expertise"]
    learning_traits = ["LearningRate", "TransferLearning"]
    behavioral_traits = ["AttentionSpan", "RiskTolerance", "InnovationDrive"]
    performance_traits = ["ProcessingSpeed", "Stability", "EnergyEfficiency"]
    social_traits = ["SocialDrive", "CommunicationStyle"]
    advanced_traits = ["MetaLearning", "Curiosity"]

    # Calculate category strengths
    cognitive_strength = sum(normalized.get(t, 0) for t in cognitive_traits) / len(
        cognitive_traits
    )
    learning_strength = sum(normalized.get(t, 0) for t in learning_traits) / len(
        learning_traits
    )
    behavioral_strength = sum(normalized.get(t, 0) for t in behavioral_traits) / len(
        behavioral_traits
    )
    performance_strength = sum(normalized.get(t, 0) for t in performance_traits) / len(
        performance_traits
    )
    social_strength = sum(normalized.get(t, 0) for t in social_traits) / len(
        social_traits
    )
    advanced_strength = sum(normalized.get(t, 0) for t in advanced_traits) / len(
        advanced_traits
    )

    # Map to hue regions (in degrees)
    # Cognitive: 180-240 (Blue-Cyan)
    # Learning: 120-180 (Green)
    # Behavioral: 60-120 (Yellow-Orange)
    # Performance: 300-360 + 0-60 (Red-Magenta)
    # Social: 240-300 (Violet)
    # Advanced: Modifies the blend

    hue_components = [
        cognitive_strength * 210,  # Blue-cyan center
        learning_strength * 150,  # Green center
        behavioral_strength * 90,  # Yellow-orange center
        performance_strength * 330,  # Red-magenta center (wraps around)
        social_strength * 270,  # Violet center
    ]

    # Weight by category strengths
    weights = [
        cognitive_strength,
        learning_strength,
        behavioral_strength,
        performance_strength,
        social_strength,
    ]

    if sum(weights) > 0:
        # Weighted average of hue components
        weighted_hue = sum(h * w for h, w in zip(hue_components, weights)) / sum(
            weights
        )

        # Advanced traits create "shimmer" - slight hue shifts
        shimmer = (
            advanced_strength * 30 * math.sin(sum(normalized.values()) * 2 * math.pi)
        )

        final_hue = (weighted_hue + shimmer) % 360
        return final_hue

    return 180.0  # Default blue if no traits


def traits_to_saturation(traits: Dict[str, Any]) -> float:
    """
    Map trait combination to saturation (0.0-1.0).

    Higher overall trait variance = more saturated colors
    More balanced traits = less saturated (more neutral)
    """
    normalized = {}
    for trait_name, trait_value in traits.items():
        normalized[trait_name] = normalize_trait_value(trait_name, trait_value)

    if not normalized:
        return 0.5

    # Calculate variance in trait values
    values = list(normalized.values())
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)

    # Map variance to saturation (higher variance = more saturated)
    # Variance range is roughly 0-0.25 for normalized values
    saturation = min(1.0, variance * 4.0)  # Scale to 0-1 range

    # Ensure minimum saturation for visibility
    return max(0.2, saturation)


def traits_to_value_brightness(traits: Dict[str, Any]) -> float:
    """
    Map trait combination to value/brightness (0.0-1.0).

    Higher overall trait levels = brighter colors
    Lower overall trait levels = darker colors
    """
    normalized = {}
    for trait_name, trait_value in traits.items():
        normalized[trait_name] = normalize_trait_value(trait_name, trait_value)

    if not normalized:
        return 0.5

    # Average of all normalized trait values
    brightness = sum(normalized.values()) / len(normalized.values())

    # Ensure reasonable brightness range (not too dark or too bright)
    return max(0.3, min(0.95, brightness))


def traits_to_hsv(traits: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Convert trait combination to HSV color values.

    Args:
        traits: Dictionary of trait names to values

    Returns:
        Tuple of (hue, saturation, value) where:
        - hue: 0-360 degrees
        - saturation: 0.0-1.0
        - value: 0.0-1.0
    """
    hue = traits_to_hue(traits)
    saturation = traits_to_saturation(traits)
    value = traits_to_value_brightness(traits)

    return (hue, saturation, value)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """
    Convert HSV to RGB values.

    Args:
        h: Hue (0-360)
        s: Saturation (0.0-1.0)
        v: Value/brightness (0.0-1.0)

    Returns:
        Tuple of (red, green, blue) values (0-255)
    """
    # Handle None values with defaults
    if h is None:
        h = 0.0
    if s is None:
        s = 0.5
    if v is None:
        v = 0.5

    h = h % 360  # Ensure hue is in valid range
    c = v * s  # Chroma
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r_prime, g_prime, b_prime = c, x, 0
    elif 60 <= h < 120:
        r_prime, g_prime, b_prime = x, c, 0
    elif 120 <= h < 180:
        r_prime, g_prime, b_prime = 0, c, x
    elif 180 <= h < 240:
        r_prime, g_prime, b_prime = 0, x, c
    elif 240 <= h < 300:
        r_prime, g_prime, b_prime = x, 0, c
    else:  # 300 <= h < 360
        r_prime, g_prime, b_prime = c, 0, x

    # Convert to 0-255 range
    r = int((r_prime + m) * 255)
    g = int((g_prime + m) * 255)
    b = int((b_prime + m) * 255)

    return (r, g, b)


def traits_to_rgb(traits: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Convert trait combination directly to RGB values.

    Args:
        traits: Dictionary of trait names to values

    Returns:
        Tuple of (red, green, blue) values (0-255)
    """
    h, s, v = traits_to_hsv(traits)
    return hsv_to_rgb(h, s, v)


def traits_to_hex_color(traits: Dict[str, Any]) -> str:
    """
    Convert trait combination to hex color string.

    Args:
        traits: Dictionary of trait names to values

    Returns:
        Hex color string (e.g., "#FF5733")
    """
    r, g, b = traits_to_rgb(traits)
    return f"#{r:02X}{g:02X}{b:02X}"


def get_color_description(traits: Dict[str, Any]) -> str:
    """
    Get a human-readable description of the color derived from traits.

    Args:
        traits: Dictionary of trait names to values

    Returns:
        Descriptive color name (e.g., "Deep Ocean Blue", "Vibrant Sunset Orange")
    """
    h, s, v = traits_to_hsv(traits)

    # Determine base color name from hue
    if 0 <= h < 30 or 330 <= h < 360:
        base_color = "Red"
    elif 30 <= h < 60:
        base_color = "Orange"
    elif 60 <= h < 90:
        base_color = "Yellow"
    elif 90 <= h < 150:
        base_color = "Green"
    elif 150 <= h < 210:
        base_color = "Blue"
    elif 210 <= h < 270:
        base_color = "Cyan"
    elif 270 <= h < 330:
        base_color = "Violet"
    else:
        base_color = "Magenta"

    # Add intensity descriptors
    if v < 0.4:
        intensity = "Deep"
    elif v > 0.8:
        intensity = "Bright"
    else:
        intensity = "Rich"

    # Add saturation descriptors
    if s < 0.3:
        saturation_desc = "Muted"
    elif s > 0.8:
        saturation_desc = "Vibrant"
    else:
        saturation_desc = intensity
        intensity = ""

    # Combine descriptors
    parts = [part for part in [saturation_desc, intensity, base_color] if part]
    return " ".join(parts)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample traits
    sample_traits = {
        "Perception": 7.5,
        "WorkingMemory": "High",
        "Expertise": 2.0,
        "LearningRate": 0.1,
        "TransferLearning": 1.5,
        "AttentionSpan": 5.0,
        "RiskTolerance": 1.0,
        "InnovationDrive": 2.5,
        "ProcessingSpeed": 2.0,
        "Stability": 2.5,
        "EnergyEfficiency": 1.8,
        "SocialDrive": 1.5,
        "CommunicationStyle": "Balanced",
        "MetaLearning": 1.2,
        "Curiosity": 2.0,
    }

    hsv = traits_to_hsv(sample_traits)
    rgb = traits_to_rgb(sample_traits)
    hex_color = traits_to_hex_color(sample_traits)
    description = get_color_description(sample_traits)

    print(f"HSV: {hsv}")
    print(f"RGB: {rgb}")
    print(f"Hex: {hex_color}")
    print(f"Description: {description}")
