# TODO: BLEU SCORE 코드 작성 (https://donghwa-kim.github.io/BLEU.html)

from typing import List


def bleu_score(reference: str, output: str, gram_num: int) -> float:
    # pre-processing sentences
    reference = reference.strip()
    output = output.strip()
    
    
    # make ngram precision
    refer_tokens = [token for token in reference.split(" ") if token != ""]
    output_tokens = [token for token in output.split(" ") if token != ""]
    precision = 1
    for num in range(1, gram_num+1):
        output_ngrams = sorted(list(zip(*[output_tokens[i:] for i in range(num)])))
        refer_ngrams = sorted(list(zip(*[refer_tokens[i:] for i in range(num)])))
        matched = len([token for token in output_ngrams if refer_ngrams.count(token)])
        precision *= matched/len(output_ngrams)
        # print(f"{matched}/{len(output_ngrams)}")

    return min(1, (len(output_tokens)/len(refer_tokens))) * precision * (1/gram_num)


# def make_ngram(reference: str, output: str, gram_num: int):
if __name__ == '__main__':


    output = '빛이 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다'
    refer = '빛이 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 가능성이 훨씬 높았다'

    print(bleu_score(refer, output, 4))
    print(min(1, 14/14) * ((10/14) * (5/13) * (2/12) * (1/11)) * (1/4))
    print(min(1, 14/14) * ((10/14) * (5/13) * (2/12) * (1/11)) * (1/4) == bleu_score(refer, output, 4))