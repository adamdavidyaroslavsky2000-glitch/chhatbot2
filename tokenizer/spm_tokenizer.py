import sentencepiece as spm


class SPMTokenizer:
    """Wrapper around a SentencePiece model for easy encoding and decoding.

    This class exposes special token IDs (PAD/BOS/EOS/UNK) and provides
    helper methods for converting between text and token IDs.  It also
    exposes the vocabulary size so that downstream modules can
    automatically adjust to the correct size.
    """

    def __init__(self, model_path: str = "tokenizer/spm.model") -> None:
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        # Save special tokens
        self.PAD = self.sp.pad_id()
        self.BOS = self.sp.bos_id()
        self.EOS = self.sp.eos_id()
        self.UNK = self.sp.unk_id()
        self.VOCAB_SIZE = self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """Encode a string into a list of token IDs.

        Args:
            text: Text to encode.
            add_bos: If True, prepend the BOS token.
            add_eos: If True, append the EOS token.

        Returns:
            List of integer token IDs.
        """
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into a string."""
        return self.sp.decode(ids)