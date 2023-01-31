import os
import os.path as osp
import shutil

import sentencepiece as spm
from torch.utils.data import dataset


def exist_file(path: str) -> bool:
    if osp.exists(path):
        return True
    return False


def create_or_load_tokenizer(
    file_path: str,
    save_path: str,
    language: str,
    vocab_size: int,
    tokenizer_type: str = "unigram",
    bos_id: int = 0,
    eos_id: int = 1,
    unk_id: int = 2,
    pad_id: int = 3,
) -> spm.SentencePieceProcessor:
    corpus_prefix = f"{language}_corpus_{vocab_size}"

    if tokenizer_type.strip().lower() not in ["unigram", "bpe", "char", "word"]:
        raise ValueError(
            "param 'tokenizer_type' must be one of ['unigram', 'bpe', 'char', 'word']"
        )

    if not osp.isdir(save_path):
        os.makedirs(save_path)

    model_path = osp.join(save_path, corpus_prefix + ".model")
    vocab_path = osp.join(save_path, corpus_prefix + ".vocab")

    if not exist_file(model_path) and not exist_file(
        vocab_path
    ):  # model과 vocab은 언제나 쌍으로 있어야 함.
        model_train_cmd = f"--input={file_path}, --model_prefix={corpus_prefix} --model_type={tokenizer_type} --vocab_size={vocab_size} --bos_id={bos_id} --eos_id={eos_id} --unk_id={unk_id} --pad_id={pad_id}"
        spm.SentencePieceTrainer.Train(model_train_cmd)

        shutil.move(corpus_prefix + ".model", model_path)
        shutil.move(corpus_prefix + ".vocab", vocab_path)

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
