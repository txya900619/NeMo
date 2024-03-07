# -*- coding: utf-8 -*-
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List

from datasets import load_dataset

from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import (
    chinese_text_preprocessing,
)
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    BaseTokenizer,
)
from nemo.utils import logging


def ipa_string_to_phoneme(ipa: str, tone_prefix=None, phoneme_sep=None) -> List[str]:
    """Converts an IPA string to a list of phonemes and tones.
    Args:
        ipa: IPA string, e.g. "a_24 t͡ɕ i_31"
        tone_prefix: Prefix for tone symbols, e.g. "#" for "#24" and "#31"
        phoneme_sep: if not None, replace this special character with space before splitting
    Examples: "a_24 t͡ɕ i_31" -> ["a", "#24", " ", "t", "͡", "ɕ", " ", "i", "#31"] (if tone_prefix is "#")
    """
    if tone_prefix is None:
        tone_prefix = ""

    if phoneme_sep is not None:
        ipa = ipa.replace(phoneme_sep, " ")

    phonemes = []

    ipa_list = re.split(r"(?<![， ])(?=[， ])|(?<=[， ])(?![， ])", ipa)

    for phoneme_with_tone in ipa_list:
        if phoneme_with_tone in [" ", "，"]:
            phonemes.append(phoneme_with_tone)
            continue
        split_phoneme_and_tone = phoneme_with_tone.split("_")

        if len(split_phoneme_and_tone) == 2:
            phoneme, tone = split_phoneme_and_tone
            # phonemes.append(phoneme)
            phonemes.extend(phoneme)  # one phoneme
            phonemes.append(f"{tone_prefix}{tone}")
        else:
            # phonemes.append(split_phoneme_and_tone[0])
            phonemes.extend(split_phoneme_and_tone[0])  # one phoneme

        # one phoneme

    # Remove trailing space
    if phonemes[-1] == " ":
        phonemes.pop()

    return phonemes


class HakkaPhonemesTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    ZH_PUNCT_LIST = list("，。？！；：、‘’“”（）【】「」《》") + list(PUNCT_LIST)

    def __init__(
        self,
        dataset_name=None,
        config_name=None,
        split_name=None,
        punct=True,
        non_default_punct_list=None,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at=None,
        pad_with_space=False,
        text_preprocessing_func=chinese_text_preprocessing,
        tone_prefix: str = "#",
        save_path=None,
        tokens_path=None,
    ):
        """Hakka phoneme-based tokenizer.
        Note: This tokenizer for now covers Hakka phonemes/tones.
        Args:
            dataset_name: Name of the hugginface dataset for which you want to generate the phoneme list and tone list.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones.
             Note that lower() function shouldn't be applied here, in case the text contains phonemes (it will be handled by g2p).
            tone_prefix: Prefix for tone symbols, e.g. "#" for "#24" and "#31"
            save_path: Path to save the token list.
        """
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space
        self.datasets_num_proc = 8

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        self.tone_prefix = tone_prefix

        if tokens_path is not None:
            with open(tokens_path, "r", encoding="utf-8") as f:
                tokens = f.read().splitlines()
        elif dataset_name is not None and config_name is not None:
            tokens.extend(self.generate_tokens(dataset_name, config_name, split_name))

            self.text_preprocessing_func = text_preprocessing_func

            if apostrophe:
                tokens.append("'")  # Apostrophe

            if punct:
                if non_default_punct_list is not None:
                    self.PUNCT_LIST = non_default_punct_list
                else:
                    self.PUNCT_LIST = list(self.ZH_PUNCT_LIST)
                tokens.extend(self.PUNCT_LIST)

            if save_path is not None:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(tokens))
        else:
            raise ValueError("Either tokens_path or dataset_name and split_name must be provided.")

        super().__init__(tokens, sep=sep, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space

    def generate_tokens(self, dataset_name, config_name, split_name):
        """Generate the phoneme list and tone list for the given dataset."""
        # Load the dataset
        dataset = load_dataset(dataset_name, config_name, split=split_name)
        
        # Only keep the ipa column
        dataset = dataset.select_columns("ipa")

        # tag_regex = re.compile(r"<\S*>")
        dataset = dataset.filter(lambda example: example["ipa"] is not None and "<OOV>" not in example["ipa"], num_proc=self.datasets_num_proc)

        # Examples: "a_24 t͡ɕ i_31" -> ["a", "#24", " ", "t", "͡", "ɕ", " ", "i", "#31"]] (if tone_prefix is "#")
        dataset = dataset.map(
            lambda example: {
                "ipa": ipa_string_to_phoneme(example["ipa"], tone_prefix=self.tone_prefix, phoneme_sep="-"),
            },
            num_proc=self.datasets_num_proc,
        )


        tokens = set()
        for example in dataset:
            ipa = example["ipa"]

            tokens.update(ipa)

        if " " in tokens:
            tokens.remove(" ")
        if "，" in tokens:
            tokens.remove("，")
                    
        return sorted(tokens)


    def encode(self, text: str) -> List[int]:
        """See base class for more information."""
        # text = self.text_preprocessing_func(text) # ipa don't need this (?)
        phonemes = ipa_string_to_phoneme(text, tone_prefix=self.tone_prefix, phoneme_sep="-")

        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)
        for p in phonemes:  # noqa
            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)
            # Add next phoneme or tone or punctuation or apostrophe.
            elif p in tokens:
                ps.append(p)
            # Warn about unknown char/phoneme
            elif p != space:
                message = f"Text: [{' '.join(phonemes)}] contains unknown char/phoneme: [{p}]."
                logging.warning(message)

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]
