# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
from typing import TYPE_CHECKING, Optional
import re
from pathlib import Path
from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_auto_gptq_available, is_gptqmodel_available, is_optimum_available, is_torch_available, logging
from ..utils.quantization_config import GPTQConfig, QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class GptqHfQuantizer(HfQuantizer):
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` or `gptqmodel` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """

    requires_calibration = False
    required_packages = ["optimum", "auto_gptq", "gptqmodel"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        from optimum.gptq import GPTQQuantizer

        self.optimum_quantizer = GPTQQuantizer.from_dict(self.quantization_config.to_dict_optimum())

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        if is_auto_gptq_available() and is_gptqmodel_available():
            logger.warning("Detected gptqmodel and auto-gptq, will use gptqmodel")

        gptq_supports_cpu = (
            is_auto_gptq_available()
            and version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
        ) or is_gptqmodel_available()
        if not gptq_supports_cpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantize model.")
        elif not (is_auto_gptq_available() or is_gptqmodel_available()):
            raise ImportError(
                "Loading a GPTQ quantized model requires gptqmodel (`pip install gptqmodel`) or auto-gptq (`pip install auto-gptq`) library. "
            )
        elif is_auto_gptq_available() and version.parse(importlib.metadata.version("auto_gptq")) < version.parse(
            "0.4.2"
        ):
            raise ImportError(
                "You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq` or use gptqmodel by `pip install gptqmodel>=1.4.3`."
            )
        elif is_gptqmodel_available() and (
            version.parse(importlib.metadata.version("gptqmodel")) < version.parse("1.4.3")
            or version.parse(importlib.metadata.version("optimum")) < version.parse("1.23.99")
        ):
            raise ImportError("The gptqmodel version should be >= 1.4.3, optimum version should >= 1.24.0")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
            logger.info("Loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually.")
        elif torch_dtype != torch.float16:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.")
        return torch_dtype

    def update_device_map(self, device_map):
        if device_map is None:
            device_map = {"": torch.device("cpu")}
        # Only with auto-gptq do not support CPU, we should move the model to cuda if available.
        if not is_gptqmodel_available() and device_map in ("cpu", {"": torch.device("cpu")}):
            device_map = {"": 0}
        return device_map

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")

        if self.pre_quantized:
            # compat: latest optimum has gptqmodel refactor
            if version.parse(importlib.metadata.version("optimum")) <= version.parse("1.23.99"):
                model = self.optimum_quantizer.convert_model(model)
            else:
                model = self.optimum_quantizer.convert_model(model, **kwargs)

    @staticmethod
    def derive_adapter_path_from_residual(
        model_name_or_path: str,
        base_dir: Optional[str] = None,
        adapter_prefix: str = "daniel_adapter",
    ) -> str:
        """
        Given a residual model path like:
        train_results_group_exp/HuggingFaceTB_SmolLM2-1.7B_residual_base_r128_fp16
        return the corresponding adapter path:
        train_results_group_exp/quantized_residuals_r128/daniel_adapter_r128_HuggingFaceTB_SmolLM2-1.7B

        If model_name_or_path is absolute, the result will be absolute.
        If it's relative, the result will be relative to its parent dir.
        You can override the root with base_dir.
        """
        residual_path = Path(model_name_or_path)
        # Parent dir to place the quantized_residuals_r{r} folder next to
        root = Path(base_dir) if base_dir is not None else residual_path.parent

        name = residual_path.name
        m = re.match(r"(?P<model_clean>.+)_residual_base_r(?P<rank>\d+)_fp16$", name)
        if not m:
            raise ValueError(
                f"Cannot parse residual model name: {name}. "
                "Expected pattern '*_residual_base_r<rank>_fp16'."
            )

        model_clean = m.group("model_clean")
        rank = m.group("rank")

        adapter_dir = root / f"quantized_residuals_r{rank}" / f"{adapter_prefix}_r{rank}_{model_clean}"
        return str(adapter_dir)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            model = self.optimum_quantizer.post_init_model(model)
        else:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path
            adapter_path = None
            if "_residual_base_r" in model.name_or_path:
                # derive next to the residual by default; pass base_dir if you need to override the root
                adapter_path = self.derive_adapter_path_from_residual(model.name_or_path)
            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer, adapter_path)
            model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return True

    def is_serializable(self, safe_serialization=None):
        return True
