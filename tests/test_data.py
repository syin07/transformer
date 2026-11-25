import pytest
from data import encode, decode

@pytest.mark.parametrize(
    "string, token_ids",
    [("123+456=579\n", [1, 2, 3, 10, 4, 5, 6, 11, 5, 7, 9, 12])]
)
def test_data_one_direction_and_round_trip(string, token_ids):
    test_encode = encode(string)

    assert test_encode == token_ids

    test_decode = decode(token_ids)

    assert test_decode == string

    assert encode(decode(token_ids)) == token_ids
    assert decode(encode(string)) == string