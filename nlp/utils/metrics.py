# TODO: BLEU SCORE 코드 작성 (https://donghwa-kim.github.io/BLEU.html)

from typing import List, Optional


def bleu_score(
    reference: str, output: str, n: Optional[int] = 4, use_BP: bool = True
) -> float:
    # pre-processing sentences
    reference = reference.strip()
    output = output.strip()

    # make ngram precision (https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/)
    # TODO: tokenizing 방식을 바꾼다면?? (현재는 띄어쓰기 기준임.)
    # TODO: token보다 큰 n_gra수가 들어오면?
    # TODO: BP 추가
    refer_tokens = [token for token in reference.split(" ") if token != ""]
    output_tokens = [token for token in output.split(" ") if token != ""]
    precision = 1

    # TODO: 1. predict > target, 2. target > preidict 고민
    gram_num = min(refer_tokens, output_tokens, n)

    if use_BP:
        bp = min(1, len(output_tokens) / len(refer_tokens))
    else:
        bp = 1

    for num in range(1, gram_num + 1):
        output_ngrams = list(zip(*[output_tokens[i:] for i in range(num)]))
        refer_ngrams = list(zip(*[refer_tokens[i:] for i in range(num)]))
        matched = len([token for token in output_ngrams if refer_ngrams.count(token)])
        precision *= matched / len(output_ngrams)
        print(f"{matched}/{len(output_ngrams)}")

    return (
        min(1, (len(output_tokens) / len(refer_tokens)))
        * (precision ** (1 / gram_num))
        * bp
    )


# def make_ngram(reference: str, output: str, gram_num: int):
# if __name__ == "__main__":
# test
#     output = "안녕하세요 저는 김예신입니까?"
#     refer = "안녕하세요 저는 김예신입니다."
# output = '빛이 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다'
# refer = '빛이 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 가능성이 훨씬 높았다'

# print(bleu_score(refer, output, 3))
# print(min(1, 14 / 14) * ((10 / 14) * (5 / 13) * (2 / 12) * (1 / 11)) ** (1 / 4))
# print(
#     min(1, 14 / 14) * ((10 / 14) * (5 / 13) * (2 / 12) * (1 / 11)) ** (1 / 4)
#     == bleu_score(refer, output, 4)
# )
