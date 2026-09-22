from __future__ import annotations

import pytest

from field_normalizer import (
    FIELD_BORROWER_NAME,
    FIELD_LOAN_NUMBER,
    FIELD_PROPERTY_ADDRESS,
    SUPPORTED_FIELDS,
    canonical_key,
    canonicalize_address,
    canonicalize_loan_number,
    canonicalize_person_name,
    cluster_keys,
    cluster_similarity_threshold,
    normalize_text,
    representative_value,
    similarity,
    values_match,
)

PAGE_ONE_ADDRESS = "604N C restview Hill Dr, Unit 1144 Las Vegas, NV 89139"
SPLIT_ADDRESS = "604 N Crest View Hill Dr Unit 1144, Las Vegas, NV 89139"
CLEAN_ADDRESS = "604 N Crestview Hill Dr Unit 1144, Las Vegas, NV 89139"
ANNOTATED_ADDRESS = "604 N Crestview Hill Dr Unit 1144, Las Vegas, NV 89139 [Property Address]"
MAILING_ADDRESS = "4124 Silvercrest Avenue, Las Vegas, NV 89129"
EM_DASH_NAME = "Alya Renard—Van Mercer"
HYPHEN_NAME = "Alya Renard-Van Mercer"
FOOTER_NAME = "Fannie Mae/Freddie Mac UNIFORM INSTRUMENT"


def test_normalize_text_handles_diacritics_and_dashes() -> None:
    assert normalize_text("Alya Renard—Van Merçer") == "alya renard van mercer"
    assert normalize_text("LOAN #: 20-414-784\n") == "loan 20 414 784"


def test_canonicalize_address_strips_brackets_and_labels() -> None:
    expected = canonicalize_address(CLEAN_ADDRESS)
    assert canonicalize_address(ANNOTATED_ADDRESS) == expected
    assert canonicalize_address("Property Address: 123 Main St") == "123 main st"
    assert canonicalize_address("123 Main St Property Address") == "123 main st"
    assert canonicalize_address("Address: 123 Main St") == "123 main st"


def test_canonicalize_address_maps_designators_and_unit_markers() -> None:
    assert (
        canonicalize_address("123 Main Street, Apartment # 4B, North Las Vegas")
        == "123 main st apt unit 4b n las vegas"
    )
    assert canonicalize_address("123 Main St Unit # 45") == "123 main st unit 45"
    assert canonicalize_address("123 Main St Unit #45") == "123 main st unit 45"


def test_canonicalize_address_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="Address must be a non-empty string"):
        canonicalize_address("   ")


def test_canonicalize_loan_number_strips_formatting() -> None:
    assert canonicalize_loan_number("20-414-784") == "20414784"
    assert canonicalize_loan_number("20414784") == "20414784"
    assert canonicalize_loan_number("AB-12 34") == "ab1234"
    assert canonicalize_loan_number("N/A") == "na"


def test_canonicalize_person_name_is_order_and_dash_insensitive() -> None:
    assert canonicalize_person_name(EM_DASH_NAME) == canonicalize_person_name(
        HYPHEN_NAME
    )
    assert canonicalize_person_name(
        "Renard-Van Mercer, Alya"
    ) == canonicalize_person_name(HYPHEN_NAME)


def test_cluster_keys_merge_split_words_but_not_other_addresses() -> None:
    clean_primary, clean_compact = cluster_keys(FIELD_PROPERTY_ADDRESS, CLEAN_ADDRESS)
    assert clean_primary == canonicalize_address(CLEAN_ADDRESS)
    assert clean_compact == clean_primary.replace(" ", "")
    assert set(cluster_keys(FIELD_PROPERTY_ADDRESS, SPLIT_ADDRESS)) & {
        clean_primary,
        clean_compact,
    }
    assert not set(cluster_keys(FIELD_PROPERTY_ADDRESS, MAILING_ADDRESS)) & {
        clean_primary,
        clean_compact,
    }
    assert canonical_key(FIELD_PROPERTY_ADDRESS, ANNOTATED_ADDRESS) == clean_primary


def test_cluster_keys_are_primary_only_for_non_address_fields() -> None:
    assert cluster_keys(FIELD_LOAN_NUMBER, "20-414-784") == ("20414784",)
    assert len(cluster_keys(FIELD_BORROWER_NAME, HYPHEN_NAME)) == 1


def test_values_match_accepts_areal_variants_and_rejects_mailing_address() -> None:
    variants = [PAGE_ONE_ADDRESS, SPLIT_ADDRESS, CLEAN_ADDRESS, ANNOTATED_ADDRESS]
    for left in variants:
        for right in variants:
            assert values_match(FIELD_PROPERTY_ADDRESS, left, right)
    for variant in variants:
        assert not values_match(FIELD_PROPERTY_ADDRESS, variant, MAILING_ADDRESS)
        assert not values_match(FIELD_PROPERTY_ADDRESS, MAILING_ADDRESS, variant)


def test_values_match_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported field"):
        values_match("unknown_field", "a", "b")
    with pytest.raises(ValueError, match="Unsupported field"):
        cluster_similarity_threshold("unknown_field")


def test_similarity_is_bounded_and_handles_empty_canonical_keys() -> None:
    assert similarity(FIELD_PROPERTY_ADDRESS, CLEAN_ADDRESS, CLEAN_ADDRESS) == 1.0
    assert (
        similarity(FIELD_PROPERTY_ADDRESS, CLEAN_ADDRESS, MAILING_ADDRESS)
        < cluster_similarity_threshold(FIELD_PROPERTY_ADDRESS)
    )
    assert canonicalize_address("Address") == ""
    assert similarity(FIELD_PROPERTY_ADDRESS, "Address", CLEAN_ADDRESS) == 0.0
    assert not values_match(FIELD_PROPERTY_ADDRESS, "Address", CLEAN_ADDRESS)


def test_representative_value_prefers_supported_clean_surface() -> None:
    variants = [
        (PAGE_ONE_ADDRESS, [1]),
        (SPLIT_ADDRESS, [3]),
        (CLEAN_ADDRESS, [4]),
        (ANNOTATED_ADDRESS, [7]),
    ]
    assert representative_value(FIELD_PROPERTY_ADDRESS, variants) == CLEAN_ADDRESS


def test_representative_value_prefers_ascii_punctuation_in_names() -> None:
    variants = [
        (EM_DASH_NAME, [1]),
        (HYPHEN_NAME, [3, 6]),
        (FOOTER_NAME, [7, 8]),
    ]
    assert representative_value(FIELD_BORROWER_NAME, variants) == HYPHEN_NAME


def test_representative_value_rejects_invalid_variants() -> None:
    with pytest.raises(ValueError, match="variants must not be empty"):
        representative_value(FIELD_PROPERTY_ADDRESS, [])
    with pytest.raises(ValueError, match="at least one page"):
        representative_value(FIELD_PROPERTY_ADDRESS, [(CLEAN_ADDRESS, [])])
    with pytest.raises(ValueError, match="positive integers"):
        representative_value(FIELD_PROPERTY_ADDRESS, [(CLEAN_ADDRESS, [0])])
    with pytest.raises(ValueError, match="positive integers"):
        representative_value(FIELD_PROPERTY_ADDRESS, [(CLEAN_ADDRESS, [True])])
    with pytest.raises(ValueError, match="Variant value must be a non-empty string"):
        representative_value(FIELD_PROPERTY_ADDRESS, [("  ", [1])])


def test_canonical_key_rejects_unknown_field_and_empty_value() -> None:
    with pytest.raises(ValueError, match="Unsupported field"):
        canonical_key("unknown_field", "value")
    with pytest.raises(ValueError, match="Loan number must be a non-empty string"):
        canonical_key(FIELD_LOAN_NUMBER, "")


def test_supported_fields_are_pinned() -> None:
    assert SUPPORTED_FIELDS == (
        "borrower_name",
        "property_address",
        "loan_number",
        "page_number",
    )
