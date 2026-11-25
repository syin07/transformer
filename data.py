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
    res = []
    for character in string:
        res.append(ENCODE_TABLE[character])
    return res

def decode(token_ids: list):
    return "".join([DECODE_TABLE[id] for id in token_ids])




