import regex

# with whitespace
TOKENIZATION_STRING_WS = r'[\p{L}]+|\d+|[ \t\f]+|[\n\r\v]+|[^\p{L}\d\s]'
TOKENIZATION_STRING = r'[\p{L}]+|\d+|[\t\f]+|[\n\r\v]+|[^\p{L}\d\s]'
filter_character = ['', '\t', '\v', '\r', '\f']
alphanumeric = '[A-zÀ-ÿ0-9]'


def sentences2tokens(text: str) -> list:
    """
    Tokenizes text into tokens, use list[x].span() to get span of token, list[x].group() for token-string. Does not tokenize whitespace.
    :param str text: The text to be tokenized
    :return: List of tokens
    :rtype: List[str]
    """
    tokenizer = regex.compile(TOKENIZATION_STRING)
    matches = [match for match in tokenizer.finditer(text)]
    return matches


def sentences2tokensWithWS(text) -> list:
    """
    Tokenizes text into tokens, use list[x].span() to get span of token, list[x].group() for token-string. Does tokenize whitespace.
    :param str text: The text to be tokenized
    :return: List of tokens
    :rtype: List[str]
    """
    tokenizer = regex.compile(TOKENIZATION_STRING_WS)
    matches = [match for match in tokenizer.finditer(text)]
    return matches
