from __future__ import annotations


INATURALIST_DOMAIN_KINGDOMS = {
    "animals": "Animalia",
    "plants": "Plantae",
    "mushrooms": "Fungi",
}
INATURALIST_DOMAIN_TAXA = {
    "animals": (1, "Animalia"),
    "insects": (3, "Insecta"),
    "plants": (1, "Plantae"),
    "mushrooms": (1, "Fungi"),
}
INATURALIST_DOMAIN_CLASS_COUNTS = {
    "animals": 5388,
    "insects": 2526,
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
    """Filter 10k-class probabilities by the selected taxon and normalize them."""

    try:
        taxon_index, taxon_name = INATURALIST_DOMAIN_TAXA[domain]
    except KeyError as error:
        raise ValueError(f"Unsupported iNaturalist domain: {domain}") from error

    filtered = {
        class_name: probability
        for class_name, probability in probabilities.items()
        if (parts := class_name.split("_"))[0].isdigit()
        and len(parts) > taxon_index
        and parts[taxon_index] == taxon_name
    }
    taxon_probability = float(sum(filtered.values()))
    if not filtered or taxon_probability <= 0:
        raise ValueError(f"No positive probabilities found for taxon {taxon_name}")
    normalized = {
        class_name: float(probability / taxon_probability)
        for class_name, probability in filtered.items()
    }
    return normalized, taxon_probability
