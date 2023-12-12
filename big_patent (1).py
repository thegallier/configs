# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""BigPatent Dataset."""
import gzip
import json
import os

import datasets


_HOMEPAGE = "https://evasharma.github.io/bigpatent/"

_CITATION = """
@misc{sharma2019bigpatent,
    title={BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization},
    author={Eva Sharma and Chen Li and Lu Wang},
    year={2019},
    eprint={1906.03741},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
BIGPATENT, consisting of 1.3 million records of U.S. patent documents
along with human written abstractive summaries.
Each US patent application is filed under a Cooperative Patent Classification
(CPC) code. There are nine such classification categories:
A (Human Necessities), B (Performing Operations; Transporting),
C (Chemistry; Metallurgy), D (Textiles; Paper), E (Fixed Constructions),
F (Mechanical Engineering; Lightning; Heating; Weapons; Blasting),
G (Physics), H (Electricity), and
Y (General tagging of new or cross-sectional technology)
There are two features:
  - description: detailed description of patent.
  - abstract: Patent abastract.
"""

_LICENSE = "Creative Commons Attribution 4.0 International"

_SPLIT_NAMES = {datasets.Split.TRAIN: "train", datasets.Split.VALIDATION: "val", datasets.Split.TEST: "test"}
_URL = "data/{version}/{split_name}.zip"

_DOCUMENT = "description"
_SUMMARY = "abstract"

_CPC_DESCRIPTION = {
    "a": "Human Necessities",
    "b": "Performing Operations; Transporting",
    "c": "Chemistry; Metallurgy",
    "d": "Textiles; Paper",
    "e": "Fixed Constructions",
    "f": "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
    "g": "Physics",
    "h": "Electricity",
    "y": "General tagging of new or cross-sectional technology",
}

# Available versions:
# 1.0.0 lower cased tokenized words.
# 2.0.0 cased raw strings.
# 2.1.2 cased raw strings (fixed).

_VERSION = "2.1.2"


class BigPatentConfig(datasets.BuilderConfig):
    """BuilderConfig for BigPatent."""

    def __init__(self, codes="all", version=_VERSION, **kwargs):
        """BuilderConfig for BigPatent.
        Args:
            codes (str or list, default 'all'): CPC codes. Either 'all' or a combination
                of {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'y'}.
            **kwargs: keyword arguments forwarded to super.
        """
        if isinstance(codes, str):
            codes = [codes]
        name = "+".join(codes)
        if name == "all":
            codes = list(_CPC_DESCRIPTION)
        if version != _VERSION:
            name = f"{name}-{version}"
        super().__init__(name=name, version=version, **kwargs)
        self.codes = codes


class BigPatent(datasets.GeneratorBasedBuilder):
    """BigPatent datasets."""

    BUILDER_CONFIG_CLASS = BigPatentConfig
    BUILDER_CONFIGS = [
        BigPatentConfig(
            codes="all",
            description="Patents under all categories.",
        ),
    ] + [
        BigPatentConfig(
            codes=k,
            description=f"Patents under Cooperative Patent Classification (CPC) {k}: {v}",
        )
        for k, v in sorted(_CPC_DESCRIPTION.items())
    ]
    DEFAULT_CONFIG_NAME = "all"
    VERSION = _VERSION

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({_DOCUMENT: datasets.Value("string"), _SUMMARY: datasets.Value("string")}),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls = {
            split: _URL.format(version=self.config.version, split_name=split_name)
            for split, split_name in _SPLIT_NAMES.items()
        }
        dl_paths = dl_manager.download_and_extract(urls)
        paths = {
            split: [
                dl_manager.iter_files(os.path.join(dl_paths[split], split_name, code)) for code in self.config.codes
            ]
            for split, split_name in _SPLIT_NAMES.items()
        }
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"paths": paths[split]},
            )
            for split in _SPLIT_NAMES
        ]

    def _generate_examples(self, paths=None):
        """Yields examples."""
        for paths_per_code in paths:
            for path in paths_per_code:
                with open(path, "rb") as fin:
                    fin = gzip.GzipFile(fileobj=fin)
                    for row in fin:
                        json_obj = json.loads(row)
                        yield json_obj["publication_number"], {
                            _DOCUMENT: json_obj[_DOCUMENT],
                            _SUMMARY: json_obj[_SUMMARY],
                        }
