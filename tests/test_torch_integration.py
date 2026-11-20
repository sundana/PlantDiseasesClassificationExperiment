"""Integration tests that need torch/torchvision.

These are skipped automatically when torch is not installed. The pretrained
mapping test downloads ImageNet weights, so it only runs when explicitly
requested:

    RUN_PRETRAINED_TESTS=1 python -m unittest tests.test_torch_integration -v
"""

import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)


@unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed")
class TestFactory(unittest.TestCase):
    def test_scratch_forward_shapes(self):
        import torch
        from src.models.factory import build_model

        for name in [
            "mobilenetv3_large",
            "mobilenetv3_small",
            "mobilenetv3_large_none",
            "proposed_large_32",
        ]:
            with self.subTest(model=name):
                model, info = build_model(name, num_classes=5)
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(1, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (1, 5))
                self.assertEqual(info["init"], "scratch")

    @unittest.skipUnless(
        os.environ.get("RUN_PRETRAINED_TESTS") == "1",
        "set RUN_PRETRAINED_TESTS=1 to download ImageNet weights",
    )
    def test_pretrained_backbone_mapping(self):
        import torch
        from src.models.factory import build_model

        for name in ["mobilenetv3_large", "mobilenetv3_small", "mobilenetv3_large_none"]:
            with self.subTest(model=name):
                model, info = build_model(name, num_classes=29, pretrained=True)
                self.assertEqual(info["init"], "imagenet_backbone")
                self.assertGreater(info["mapped_tensors"], 100)
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(1, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (1, 29))


@unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed")
class TestExcludeClasses(unittest.TestCase):
    def test_exclude_before_split(self):
        from PIL import Image
        from torchvision import transforms

        from src.data_setup import create_dataloader

        with tempfile.TemporaryDirectory() as tmp:
            for class_name in ["alpha", "beta", "gamma"]:
                class_dir = os.path.join(tmp, class_name)
                os.makedirs(class_dir)
                for i in range(10):
                    Image.new("RGB", (32, 32), (i * 10, 50, 100)).save(
                        os.path.join(class_dir, f"{i}.png")
                    )

            loader, _, _, class_names = create_dataloader(
                data_path=tmp,
                train_transform=transforms.ToTensor(),
                test_transform=transforms.ToTensor(),
                batch_size=4,
                num_workers=0,
                stratified=False,
                exclude_classes=["gamma"],
            )
            self.assertEqual(class_names, ["alpha", "beta"])
            images, labels = next(iter(loader))
            self.assertLess(int(labels.max()), 2)
            self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
