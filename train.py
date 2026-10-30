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

"""Train one GOFish scene.

    python train.py --config-name apps/fiord path=<scene> out_dir=<runs> experiment_name=<name>
"""

import random

import hydra
import numpy as np
from omegaconf import DictConfig

from threedgrut.utils.logger import logger


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    seed_everything(int(conf.seed_initialization))
    logger.info("Compiling native code..")
    from threedgrut.trainer import Trainer3DGRUT

    trainer = Trainer3DGRUT(conf)
    trainer.run_training()


if __name__ == "__main__":
    main()
