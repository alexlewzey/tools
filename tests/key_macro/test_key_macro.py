from src.key_macro.callables import ENCODINGS


def test_no_duplicate_encodings() -> None:
    codes = [macro.encoding for macro in ENCODINGS]
    assert len(codes) == len(set(codes))
