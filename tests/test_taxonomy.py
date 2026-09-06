import pytest

from src.taxonomy import (
    filter_inaturalist_probabilities,
    format_inaturalist_taxonomy,
    inaturalist_kingdom,
)


PROBABILITIES = {
    "00001_Animalia_Chordata_Mammalia_Test_Test_Canis_lupus": 0.30,
    "00002_Animalia_Chordata_Aves_Test_Test_Corvus_corax": 0.20,
    "00003_Plantae_Tracheophyta_Test_Test_Test_Quercus_robur": 0.40,
    "00004_Fungi_Ascomycota_Test_Test_Test_Amanita_muscaria": 0.10,
}


def test_kingdom_is_read_from_class_name():
    assert inaturalist_kingdom(next(iter(PROBABILITIES))) == "Animalia"


def test_animal_filter_returns_mass_and_normalized_probabilities():
    filtered, mass = filter_inaturalist_probabilities(PROBABILITIES, "animals")
    assert mass == pytest.approx(0.5)
    assert sum(filtered.values()) == pytest.approx(1.0)
    assert set(filtered) == set(list(PROBABILITIES)[:2])


def test_insect_filter_uses_insecta_taxonomic_class():
    insect = "00005_Animalia_Arthropoda_Insecta_Test_Test_Apis_mellifera"
    probabilities = {**PROBABILITIES, insect: 0.25}

    filtered, mass = filter_inaturalist_probabilities(probabilities, "insects")

    assert mass == pytest.approx(0.25)
    assert filtered == {insect: pytest.approx(1.0)}


def test_taxonomy_path_is_translated_for_ukrainian_interface():
    taxa = ["Animalia", "Chordata", "Mammalia", "Carnivora", "Canidae"]

    assert format_inaturalist_taxonomy(taxa, "УКР") == (
        "Царство: Тварини · Тип: Хордові · Клас: Ссавці · "
        "Ряд: Хижі · Родина: Псові"
    )
    assert format_inaturalist_taxonomy(taxa, "EN") == " › ".join(taxa)


def test_unknown_domain_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        filter_inaturalist_probabilities(PROBABILITIES, "histology")
