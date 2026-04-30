import pytest

from lacuna.assets.naming import cache_dir_name


def test_basic_entity_formatting():
    assert (
        cache_dir_name("sntf-cache", atlas="hansen2022", connectome="dTOR985")
        == "sntf-cache/atlas-hansen2022_connectome-dTOR985"
    )


def test_entities_are_sorted_for_determinism():
    a = cache_dir_name("ace-cache", connectome="GSP1000", atlas="hansen2022")
    b = cache_dir_name("ace-cache", atlas="hansen2022", connectome="GSP1000")
    assert a == b
    assert a == "ace-cache/atlas-hansen2022_connectome-GSP1000"


def test_value_must_not_contain_underscore_or_dash():
    with pytest.raises(ValueError, match="entity value"):
        cache_dir_name("sntf-cache", atlas="hansen_2022")
