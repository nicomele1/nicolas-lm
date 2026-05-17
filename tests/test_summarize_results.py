from scripts.summarize_results import effect_rows, format_number


def test_format_number_rounds_small_values() -> None:
    assert format_number("2.0744338178634645") == "2.0744"


def test_format_number_uses_two_decimals_for_large_values() -> None:
    assert format_number("107.31605288253243") == "107.32"


def test_effect_rows_compares_medium_and_high() -> None:
    rows = [
        {
            "model_name": "transformer",
            "corpus_name": "medium",
            "test_loss_mean": "2.0",
            "test_ppl": "10.0",
            "generalization_gap": "0.1",
        },
        {
            "model_name": "transformer",
            "corpus_name": "high",
            "test_loss_mean": "2.5",
            "test_ppl": "15.0",
            "generalization_gap": "0.3",
        },
    ]

    effects = effect_rows(rows)

    assert effects == [
        {
            "model_name": "transformer",
            "delta_test_loss_high_minus_medium": "0.5000",
            "relative_ppl_change_high_vs_medium": "0.5000",
            "delta_gap_high_minus_medium": "0.2000",
        }
    ]
