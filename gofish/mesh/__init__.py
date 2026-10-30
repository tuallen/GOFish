# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Full Field-of-View Mesh Extraction (matched camera-min opacity field + Marching Tetrahedra)."""

from .extract import extract_mesh

__all__ = ["extract_mesh"]
