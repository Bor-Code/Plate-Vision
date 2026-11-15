from plate_vision.data.profiles import (
    AzerbaijanCountryProfile,
    CountryProfileRegistry,
    GermanyCountryProfile,
    TurkeyCountryProfile,
    UnitedKingdomCountryProfile,
)
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor


def test_multi_country_post_processor_auto_detects_jurisdiction() -> None:
    registry = CountryProfileRegistry()
    registry.register(TurkeyCountryProfile())
    registry.register(GermanyCountryProfile())
    registry.register(UnitedKingdomCountryProfile())
    registry.register(AzerbaijanCountryProfile())

    processor = MultiCountryPostProcessor(
        country_codes=("TR", "GB", "DE", "AZ"),
        registry=registry,
    )
    assert processor.country_codes == ("TR", "GB", "DE", "AZ")

    tr_res = processor.resolve("34 ABC 123")
    assert tr_res.country_code == "TR"
    assert tr_res.corrected_text == "34ABC123"
    assert tr_res.is_valid_format is True
    assert tr_res.was_corrected is False

    gb_res = processor.resolve("BD51 SMR")
    assert gb_res.country_code == "GB"
    assert gb_res.corrected_text == "BD51SMR"
    assert gb_res.is_valid_format is True

    gb_corrected = processor.resolve("8D5I 5MR")
    assert gb_corrected.country_code == "GB"
    assert gb_corrected.corrected_text == "BD51SMR"
    assert gb_corrected.was_corrected is True

    de_res = processor.resolve("MUC AB 987")
    assert de_res.country_code == "DE"
    assert de_res.corrected_text == "MUCAB987"
    assert de_res.is_valid_format is True

    az_res = processor.resolve("90 ZZ 555")
    assert az_res.country_code == "AZ"
    assert az_res.corrected_text == "90ZZ555"
    assert az_res.is_valid_format is True

    fallback = processor.resolve("???")
    assert fallback.country_code == "TR"
    assert fallback.is_valid_format is False
