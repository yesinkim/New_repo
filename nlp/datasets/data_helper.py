import os
import os.path as osp
import shutil
from abc import abstractmethod
from typing import List

import sentencepiece as spm
import torch
from torch import Tensor
from torch.utils.data import Dataset


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


class AbstractDataset(Dataset):
    def __init__(
        self, x_path: str, src_vocab: spm.SentencePieceProcessor, max_sequence_size: int
    ) -> None:
        self.src_data = open(x_path, "r", encoding="utf-8").readlines()
        self.src_vocab = src_vocab
        self.max_sequence_size = max_sequence_size
        self.pad = self.src_vocab["<pad>"]

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        return super().__getitem__(index)

    @abstractmethod
    def encoder_input2tensor(self, sentence: str) -> Tensor:
        pass

    def padding(self, idx_list: List[int]) -> List[int]:
        pass


class TrainDataset(AbstractDataset):
    def __init__(
        self,
        x_path: str,
        src_vocab: spm.SentencePieceProcessor,
        y_path: str,
        trg_vocab: spm.SentencePieceProcessor,
        max_sequence_size,
    ):
        self.trg_data = open(y_path, "r", encoding="utf-8").readlines()
        self.trg_vocab = trg_vocab
        self.bos = self.trg_vocab["<s>"]
        self.eos = self.trg_vocab["</s>"]

    def __len__(self) -> int:
        if len(self.src_data) != len(self.trg_data):
            raise IndexError("Not equal X data, Y data line size")
        return len(self.src_data)

    def __getitem__(self, index):
        encoder_input = self.encoder_input2tensor(self.src_data[index])
        decoder_input = self.decoder_input2tensor(self.trg_data[index])
        decoder_output = decoder_input[1:] + torch.tensor(self.eos)

        return encoder_input, self.padding(decoder_input), self.padding(decoder_output)

    def encoder_input2tensor(self, sentence: str) -> Tensor:
        idx_list = self.src_vocab.EncodeAsIds(sentence)
        idx_list = self.padding(idx_list)

        return idx_list

    def decoder_input2tensor(self, sentence: str) -> Tensor:
        idx_list = self.trg_vocab.EncodeAsIds(sentence)
        idx_list.insert(0, self.bos)

        return torch.tensor(idx_list)

    def decoder_output2tensor(self, sentence: str) -> Tensor:
        idx_list = self.trg_vocab.EncodeAsIds(sentence)
        idx_list.append(self.eos)
        idx_list = self.padding(idx_list)
        # TODO: 마지막에 꼭 eos이 나와야 하는지, 다음단어가 나오는 게 좋을지 고려해보아야 함.
        if len(idx_list) > self.max_sequence_size:
            idx_list = idx_list[self.max_sequence_size]

        return torch.tensor(idx_list)

    def padding(self, idx_list: List[int]) -> List[int]:
        if len(idx_list) < self.max_sequence_size:
            idx_list = idx_list + [self.pad] * (self.max_sequence_size - len(idx_list))
        else:
            idx_list = idx_list[: self.max_sequence_size]

        return idx_list
