from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import onnxruntime as ort


class U2NetSegmenter:
    """Reusable ONNX adapter which discovers tensor names and shapes at runtime."""

    def __init__(self, model_path: str | Path, config: dict[str, Any], session_factory: Callable = ort.InferenceSession):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"U2Net model not found: {self.model_path}")
        self.session = session_factory(str(self.model_path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ValueError(f"Incompatible U2Net model: expected one input and at least one output, got {len(inputs)} and {len(outputs)}")
        self.input = inputs[0]
        self.outputs = outputs
        input_type = getattr(self.input,"type","tensor(float)")
        if input_type != "tensor(float)":
            raise ValueError(f"Incompatible U2Net input type: {input_type}")
        shape = list(self.input.shape)
        if len(shape) != 4:
            raise ValueError(f"Incompatible U2Net input rank: {shape}")
        channels = shape[1] if isinstance(shape[1], int) else 3
        if channels != 3:
            raise ValueError(f"Incompatible U2Net channel count: {shape}")
        output_shape=list(getattr(self.outputs[0],"shape",[]))
        if output_shape and len(output_shape)!=4:
            raise ValueError(f"Incompatible U2Net output rank: {output_shape}")
        self.height = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 320
        self.width = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 320
        self.mean = np.asarray(config.get("input_mean", [0.485, 0.456, 0.406]), np.float32).reshape(1,1,3)
        self.std = np.asarray(config.get("input_std", [0.229, 0.224, 0.225]), np.float32).reshape(1,1,3)

    @classmethod
    def from_config(cls, config: dict[str, Any], session_factory: Callable = ort.InferenceSession):
        env_var = config.get("model_env_var", "OPENSCAN_U2NET_MODEL")
        path = os.environ.get(env_var)
        if not path:
            raise ValueError(f"U2Net model path is not set. Define {env_var} in .env or the environment.")
        return cls(path, config, session_factory)

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("U2Net expects a non-empty BGR image with three channels")
        target_shape = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = np.transpose(normalized, (2,0,1))[None]
        values = self.session.run([self.outputs[0].name], {self.input.name: tensor})
        if not values:
            raise ValueError("U2Net inference returned no outputs")
        prediction = np.asarray(values[0]).squeeze()
        if prediction.ndim != 2:
            raise ValueError(f"Incompatible U2Net output shape: {np.asarray(values[0]).shape}")
        prediction = prediction.astype(np.float32)
        finite = np.isfinite(prediction)
        if not finite.any():
            raise ValueError("U2Net output contains no finite values")
        prediction = np.nan_to_num(prediction)
        lo, hi = float(prediction.min()), float(prediction.max())
        if lo < 0.0 or hi > 1.0:
            prediction = 1.0 / (1.0 + np.exp(-np.clip(prediction, -30, 30)))
        prediction = np.clip(prediction, 0.0, 1.0)
        return cv2.resize(prediction, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
