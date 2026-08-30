import pytest

from src.taxonomy import filter_inaturalist_probabilities, inaturalist_kingdom


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


def test_unknown_domain_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        filter_inaturalist_probabilities(PROBABILITIES, "histology")
