import tiktoken


def tokenization_test():
    text = "Chapter 1: Building Rapport and Capturing"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    print(tokens)

if __name__ == '__main__':
    tokenization_test()