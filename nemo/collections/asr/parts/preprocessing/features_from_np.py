# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter
import math
import random
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.utils import logging

class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(
        self,
        samples,
        sample_rate,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db=None,
    ):
        audio = AudioSegment(
            samples[min(offset, samples.size):max(offset + duration*sample_rate, samples.size)],
            sample_rate=sample_rate,
            target_sr=self.sample_rate,
            trim=trim,
            trim_ref=trim_ref,
            trim_top_db=trim_top_db,
            trim_frame_length=trim_frame_length,
            trim_hop_length=trim_hop_length,
            orig_sr=orig_sr,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
        )
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa)