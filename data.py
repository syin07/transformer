"""
Vocab:
0-9
+ (10)
= (11)
<eos> (12)
"""
ENCODE_TABLE = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "=": 11,
    "\n": 12,
}

DECODE_TABLE = { value: key for key, value in ENCODE_TABLE.items() }

VOCAB_SIZE = len(ENCODE_TABLE)

def encode(string: str):
    """Turns a string into a list of token_ids"""
    res = []
    for character in string:
        res.append(ENCODE_TABLE[character])
    return res

def decode(token_ids: list):
    """Turn a list of token_ids into a string"""
    return "".join([DECODE_TABLE[id] for id in token_ids])


def generate_problems(num_digits, num_problems):
    """Generates num_problems addition problems up to num_digits"""
    num_test = min(10000, num_problems // 10)
    num_validation = min(2000, num_probles // 10)
    num_training = num_problems - num_test - num_validation

    # sample evenly across each num of digits
    test, validation, training = [], [], []
    seen = set()

    # TODO: finish this function




    







