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
        Robustly derive the LoRA adapter directory for various residual layouts.

        Supported inputs (basename of model_name_or_path):
          - "<model>_residual_base_r<rank>_fp16"
          - "w_res_<model>_r<rank>_daniel_<bits>bit_gs<g>_<calib>"
          - Iteration folders: "quantized_iter_<t>_..." or "residual_iter_<t>_fp16"
            located inside a folder named: "quantized_residuals_r<rank>".

        Returns:
          ".../quantized_residuals_r<rank>/<adapter_prefix>_r<rank>_<model>"
        """
        residual_path = Path(model_name_or_path)
        root = Path(base_dir) if base_dir is not None else residual_path.parent
        name = residual_path.name

        # Case 1: FP residual base
        m = re.match(r"(?P<model_clean>.+)_residual_base_r(?P<rank>\d+)_fp16$", name)
        if m:
            model_clean = m.group("model_clean")
            rank = m.group("rank")
            adapter_dir = root / f"quantized_residuals_r{rank}" / f"{adapter_prefix}_r{rank}_{model_clean}"
            return str(adapter_dir)

        # Case 2: Quantized residual artifact "w_res_<model>_r<rank>_..."
        m = re.match(r"w_res_(?P<model_clean>.+)_r(?P<rank>\d+)_", name)
        if m:
            model_clean = m.group("model_clean")
            rank = m.group("rank")
            # If already inside quantized_residuals_r<rank>, use that as root
            if re.match(rf"quantized_residuals_r{rank}$", root.name):
                adapter_dir = root / f"{adapter_prefix}_r{rank}_{model_clean}"
            else:
                adapter_dir = root / f"quantized_residuals_r{rank}" / f"{adapter_prefix}_r{rank}_{model_clean}"
            return str(adapter_dir)

        # Case 3: Inside a quantized_residuals_r<rank> directory (iterations etc.)
        parent = residual_path.parent
        m_parent = re.match(r"quantized_residuals_r(?P<rank>\d+)$", parent.name)
        if m_parent:
            rank = m_parent.group("rank")
            # Prefer the latest adapter_iter_<t> numerically
            candidates = [
                d for d in parent.iterdir()
                if d.is_dir() and d.name.startswith(f"{adapter_prefix}_r{rank}_")
            ]
            if candidates:
                # Split into adapter_iter_t and base adapter
                iter_adapters = []
                base_adapters = []
                for d in candidates:
                    m_iter = re.match(rf"{adapter_prefix}_r{rank}_.+?/??$", d.name)
                    # Detect explicit iteration folders like 'adapter_iter_<t>' as separate naming too
                    m_alt = re.match(r"adapter_iter_(?P<t>\d+)$", d.name)
                    if m_alt:
                        try:
                            iter_adapters.append((int(m_alt.group("t")), d))
                        except Exception:
                            pass
                    else:
                        base_adapters.append(d)
                if iter_adapters:
                    iter_adapters.sort(key=lambda x: x[0])
                    return str(iter_adapters[-1][1])
                # If we didn't find explicit 'adapter_iter_*', try to infer latest by mtime
                try:
                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                    return str(latest)
                except Exception:
                    return str(sorted(candidates)[-1])
            # Try to infer model_clean from a sibling w_res_* directory
            model_clean = None
            for d in parent.iterdir():
                if not d.is_dir():
                    continue
                mm = re.match(r"w_res_(?P<model_clean>.+)_r(?P<r>\d+)_", d.name)
                if mm and mm.group("r") == rank:
                    model_clean = mm.group("model_clean")
                    break
            if model_clean:
                pref = parent / f"{adapter_prefix}_r{rank}_{model_clean}"
                if pref.exists():
                    return str(pref)
            # Fallback: first matching adapter if available
            candidates = sorted(
                d for d in parent.iterdir()
                if d.is_dir() and d.name.startswith(f"{adapter_prefix}_r{rank}_")
            )
            if candidates:
                return str(candidates[0])

        # Case 4: Look one or two levels up for a quantized_residuals_r<rank> folder
        for anc in [residual_path.parent, residual_path.parent.parent if residual_path.parent else None]:
            if anc is None:
                continue
            m_anc = re.match(r"quantized_residuals_r(?P<rank>\d+)$", anc.name)
            if m_anc:
                rank = m_anc.group("rank")
                candidates = sorted(
                    d for d in anc.iterdir()
                    if d.is_dir() and d.name.startswith(f"{adapter_prefix}_r{rank}_")
                )
                if candidates:
                    return str(candidates[0])

        raise ValueError(
            f"Cannot derive adapter path from '{model_name_or_path}'. "
            "Expected names like '*_residual_base_r<rank>_fp16' or 'w_res_<model>_r<rank>_...' "
            "or be located inside a 'quantized_residuals_r<rank>/' directory."
        )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            model = self.optimum_quantizer.post_init_model(model)
        else:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path
            adapter_path = None
            name_path = Path(model.name_or_path)
            name = name_path.name
            parent = name_path.parent
            # Derive adapter path for multiple residual naming schemes
            if (
                "_residual_base_r" in name
                or name.startswith("w_res_")
                or name.startswith("quantized_iter_")
                or name.startswith("residual_iter_")
                or re.match(r"quantized_residuals_r\d+", parent.name) is not None
            ):
                try:
                    adapter_path = self.derive_adapter_path_from_residual(model.name_or_path)
                except Exception as e:
                    logger.warning(f"[GPTQ] Could not derive adapter path for '{model.name_or_path}': {e}")
            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer, adapter_path)
            model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return True

    def is_serializable(self, safe_serialization=None):
        return True
