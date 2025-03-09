# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import torch._dynamo.config
from iree.turbine.dynamo.backends.base import backend_generator

torch._dynamo.config.dynamic_shapes = (
    False  # TODO: https://github.com/nod-ai/SHARK-ModelDev/issues/93
)


class MNISTDataLoader:
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.mnist_trainset = datasets.MNIST(
            root="../data", train=True, download=True, transform=transform
        )
        self.mnist_testset = datasets.MNIST(
            root="../data", train=False, download=True, transform=transform
        )

    def get_train_loader(self):
        return DataLoader(
            dataset=self.mnist_trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def get_test_loader(self):
        return DataLoader(
            dataset=self.mnist_testset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(28, 28, bias=True)
        self.layer1 = nn.Linear(28, 14, bias=True)
        self.layer2 = nn.Linear(14, 7, bias=True)
        self.layer3 = nn.Linear(7, 7, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        x = torch.sigmoid(x)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        return x


def infer_iteration(model, images):
    outputs = model(images)
    return outputs


def infer(backend):
    # Example Parameters
    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "num_epochs": 10,
    }

    custom_data_loader = MNISTDataLoader(config["batch_size"])
    test_loader = custom_data_loader.get_test_loader()

    model = MLP()
    test_opt = torch.compile(infer_iteration, backend=backend)

    for i, (images, labels) in enumerate(test_loader):
        test_opt(model, images)


class ModelTests(unittest.TestCase):
    def testMNISTEagerDefaultCpu(self):
        infer("turbine_cpu")

    def testMNISTEagerCustomBackend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test_MNIST.mlir"
            options = {
                "target_backends": "llvm-cpu",
                "flags": [
                    "--iree-llvmcpu-target-cpu=host",
                    "--iree-input-demote-f64-to-f32=false",
                ],
                "driver": "local-sync",
                "save_mlir": path,
            }
            infer(backend_generator(**options))
            mlir_str = path.read_text()
            self.assertIn("func.func", mlir_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
