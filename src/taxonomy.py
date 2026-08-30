from __future__ import annotations


INATURALIST_DOMAIN_KINGDOMS = {
    "animals": "Animalia",
    "plants": "Plantae",
    "mushrooms": "Fungi",
}
INATURALIST_DOMAIN_CLASS_COUNTS = {
    "animals": 5388,
    "plants": 4271,
    "mushrooms": 341,
}


def inaturalist_kingdom(class_name: str) -> str | None:
    parts = class_name.split("_")
    if len(parts) >= 3 and parts[0].isdigit():
        return parts[1]
    return None


def filter_inaturalist_probabilities(
    probabilities: dict[str, float], domain: str
) -> tuple[dict[str, float], float]:
    """Filter 10k-class probabilities by kingdom and normalize within it."""

    try:
        kingdom = INATURALIST_DOMAIN_KINGDOMS[domain]
    except KeyError as error:
        raise ValueError(f"Unsupported iNaturalist domain: {domain}") from error

    filtered = {
        class_name: probability
        for class_name, probability in probabilities.items()
        if inaturalist_kingdom(class_name) == kingdom
    }
    kingdom_probability = float(sum(filtered.values()))
    if not filtered or kingdom_probability <= 0:
        raise ValueError(f"No positive probabilities found for kingdom {kingdom}")
    normalized = {
        class_name: float(probability / kingdom_probability)
        for class_name, probability in filtered.items()
    }
    return normalized, kingdom_probability
