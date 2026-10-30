# Tanks and Temples toolbox (https://github.com/isl-org/TanksAndTemples), as packaged by GOF.
# The MIT License (MIT), Copyright (c) 2017 Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, Vladlen Koltun:
# third_party/licenses/TANKS_AND_TEMPLES_LICENSE.md

import os


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
