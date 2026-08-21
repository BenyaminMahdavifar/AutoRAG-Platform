"""
Automated Tests for AutoRAG PyTorch Installation Policy and Embedding Device Resolution.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

from autorag.torch_policy import (
    PYTORCH_CUDA_VARIANT, PYTORCH_INDEX_URL, PYTORCH_PACKAGE,
    resolve_embedding_device, CUDAUnavailableError, inspect_installed_torch, get_nvidia_driver_info
)


class TestTorchPolicy(unittest.TestCase):

    def test_canonical_constants(self):
        self.assertEqual(PYTORCH_CUDA_VARIANT, "cu124")
        self.assertEqual(PYTORCH_INDEX_URL, "https://download.pytorch.org/whl/cu124")
        self.assertEqual(PYTORCH_PACKAGE, "torch")

    def test_resolve_embedding_device_cpu(self):
        # CPU setting must always return 'cpu'
        self.assertEqual(resolve_embedding_device("cpu"), "cpu")
        self.assertEqual(resolve_embedding_device("CPU"), "cpu")
        self.assertEqual(resolve_embedding_device(" cpu "), "cpu")

    def test_resolve_embedding_device_cuda_when_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            self.assertEqual(resolve_embedding_device("cuda"), "cuda")

    def test_resolve_embedding_device_cuda_when_cuda_unavailable_raises_error(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.__version__ = "2.5.1+cu124"
        mock_torch.version.cuda = "12.4"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            with self.assertRaises(CUDAUnavailableError) as ctx:
                resolve_embedding_device("cuda")
            self.assertEqual(ctx.exception.code, "CUDA_RUNTIME_UNAVAILABLE")
            err_dict = ctx.exception.to_dict("2.5.1+cu124", "12.4")
            self.assertEqual(err_dict["code"], "CUDA_RUNTIME_UNAVAILABLE")
            self.assertIn("recommendations", err_dict)

    def test_resolve_embedding_device_auto_fallback_to_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Auto must fall back gracefully to cpu without raising error
            self.assertEqual(resolve_embedding_device("auto"), "cpu")
            self.assertEqual(resolve_embedding_device(""), "cpu")
            self.assertEqual(resolve_embedding_device(None), "cpu")

    def test_resolve_embedding_device_auto_selects_cuda_if_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            self.assertEqual(resolve_embedding_device("auto"), "cuda")

    def test_invalid_device_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_embedding_device("tpu")

    def test_inspect_installed_torch_cuda_build(self):
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.5.1+cu124"
        mock_torch.version.cuda = "12.4"
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            info = inspect_installed_torch()
            self.assertTrue(info["installed"])
            self.assertTrue(info["isCudaBuild"])
            self.assertFalse(info["isCpuBuild"])
            self.assertEqual(info["healthCode"], "TORCH_CUDA_BUILD_READY")

    def test_inspect_installed_torch_cpu_build_detected_as_mismatch(self):
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.5.1+cpu"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            info = inspect_installed_torch()
            self.assertTrue(info["installed"])
            self.assertFalse(info["isCudaBuild"])
            self.assertTrue(info["isCpuBuild"])
            self.assertEqual(info["healthCode"], "TORCH_CPU_BUILD_DETECTED")


if __name__ == "__main__":
    unittest.main()
