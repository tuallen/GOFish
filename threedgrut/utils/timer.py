# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright Amazon.com, Inc. or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Callable, Final, TypeVar

import torch

log: Final = logging.getLogger(__name__)


# global variable that can be imported in other scripts for usage

C = TypeVar("C", bound=Callable)


# Timer utilities based on NVIDIA Warp's ScopedTimer (https://github.com/NVIDIA/warp, warp/utils.py).


class CudaTimer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        if self.enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._recording = False
            self._end = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.enabled:
            assert not self._recording, "CudaTimer has already started."
            self._start.record()
            self._recording = True

    def end(self):
        if self.enabled:
            assert self._recording, "CudaTimer has not started."
            self._end.record()
            self._recording = False

    def timing(self) -> float:
        if self.enabled:
            assert not self._recording, "CudaTimer has not ended."
            self._end.synchronize()
            return self._start.elapsed_time(self._end)
        return 0.0
