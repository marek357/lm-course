import torch

from ttlm.tokenizer.base import Tokenizer
from tqdm import tqdm


class BPETokenizer(Tokenizer):
    def __init__(self, dataset, num_merges: int = 10, max_vocab_size: int = 10000) -> None:
        super().__init__()
        self.bpe_ranks = {}
        self.vocab = [chr(c) for c in range(256)]
        self.inverse_vocab = {k: v for k, v in enumerate(self.vocab)}
        self.num_merges = num_merges
        self.max_vocab_size = max_vocab_size

        self.train(dataset.data)
        # print(self.vocab)

    """Abstract base class for tokenizers."""

    @property
    def bos_token_id(self) -> int:
        """Beginning of sentence token id."""
        return len(self.vocab) + 1

    @property
    def bos_token(self) -> str:
        """Beginning of sentence token string."""
        return '<bos>'

    @property
    def eos_token_id(self) -> int:
        """End of sentence token id."""
        return len(self.vocab) + 2

    @property
    def eos_token(self) -> str:
        """End of sentence token string."""
        return '</eos>'

    @property
    def pad_token_id(self) -> int:
        """Padding token id."""
        return len(self.vocab) + 3

    @property
    def pad_token(self) -> str:
        """Padding token string."""
        return '<pad>'

    @property
    def unk_token_id(self) -> int:
        """Unknown token id."""
        return len(self.vocab) + 4

    @property
    def unk_token(self) -> str:
        """Unknown token string."""
        return "<unk>"

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    def _merge_vocab(self, pair: tuple[str, str]) -> None:
        """Merges the most frequent pair in the vocabulary."""
        new_token = pair[0] + pair[1]
        # print('Adding new token to vocab: "', new_token, '"')
        self.vocab.append(new_token)
        self.inverse_vocab[new_token] = len(self.vocab) - 1

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        """Encodes a list of tokens to their corresponding ids."""
        assert all(
            token in self.vocab for token in tokens), "Some tokens are not in the vocabulary."
        return [self.inverse_vocab.get(token, self.unk_token_id) for token in tokens]

    def train(self, texts: list[str]) -> None:
        """Trains the tokenizer on a list of texts."""
        for _ in tqdm(range(self.num_merges)):
            frequencies = {}
            breakdowns = []
            for text in texts[:100]:
                # break down text into tokens
                end_idx = 0
                og_text = text
                while end_idx < len(og_text):
                    for token in sorted(self.vocab, key=len, reverse=True):
                        if text.startswith(token) and end_idx + len(token) <= len(og_text):
                            # print("Matching token:", token, "in text:", text)
                            breakdowns.append(token)
                            frequencies[token] = frequencies.get(token, 0) + 1
                            end_idx += len(token)
                            text = og_text[end_idx:]
                            break

            # find most frequent pair
            pair_frequencies = {}
            for i in range(len(breakdowns) - 1):
                pair = (breakdowns[i], breakdowns[i + 1])
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1

            # Find the most frequent pair
            best_pair = max(pair_frequencies, key=pair_frequencies.get)

            # Merge the pair
            self._merge_vocab(best_pair)

            # Stop if we reach the max vocab size
            if len(self.vocab) >= self.max_vocab_size:
                break

    def encode(
        self, strings: list[str], bos: bool = True, eos: bool = True
    ) -> list[torch.LongTensor]:
        """Encodes a batch of strings."""
        encoded = []
        for text in strings[:500]:
            # break down text into tokens
            end_idx = 0
            og_text = text
            breakdowns = []
            while end_idx < len(og_text):
                for token in sorted(self.vocab, key=len, reverse=True):
                    if text.startswith(token) and end_idx + len(token) <= len(og_text):
                        # print("Matching token:", token, "in text:", text)
                        breakdowns.append(token)
                        end_idx += len(token)
                        text = og_text[end_idx:]
                        break

            encoded.append(breakdowns)

        return [torch.LongTensor(self.encode_tokens(breakdown)) for breakdown in encoded]

    def decode(
        self, tokens: list[list[int]], special_tokens: bool = False
    ) -> list[str]:
        """Decodes a batch of tokens.
        When special_tokens=True, special tokens should be decoded as:
        - BOS token: <BOS>
        - EOS token: <EOS>
        - UNK token: <UNK>
        - PAD token: <PAD>
        """
        decoded = []
        for token_list in tokens:
            string_tokens = [
                self.vocab[token_id] if token_id < len(
                    self.vocab) else self.unk_token
                for token_id in token_list
            ]
            decoded.append("".join(string_tokens))
        return decoded
