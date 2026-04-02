"""
Model metadata provider for Seldon's /metadata endpoint.
"""

from datetime import datetime


class ModelMetadata:
    """Returns model info for Seldon's metadata API."""

    def __init__(
        self,
        name: str = "default",
        version: str = "v1",
        model_type: str = "image",
    ):
        self.name = name
        self.version = version
        self.model_type = model_type

    def metadata(self) -> dict:
        input_schema = self._get_input_schema()
        output_schema = self._get_output_schema()

        return {
            "name": self.name,
            "versions": [self.version],
            "platform": "pytorch",
            "inputs": input_schema,
            "outputs": output_schema,
            "custom": {
                "model_type": self.model_type,
                "framework": "pytorch",
                "updated_at": datetime.utcnow().isoformat(),
            },
        }

    def _get_input_schema(self) -> list[dict]:
        if self.model_type == "image":
            return [{
                "name": "image",
                "datatype": "FP32",
                "shape": [-1, 3, 224, 224],
            }]
        elif self.model_type == "text":
            return [{
                "name": "text",
                "datatype": "BYTES",
                "shape": [-1],
            }]
        return [{"name": "input", "datatype": "FP32", "shape": [-1]}]

    def _get_output_schema(self) -> list[dict]:
        if self.model_type == "image":
            return [{
                "name": "predictions",
                "datatype": "FP32",
                "shape": [-1, 1000],
            }]
        elif self.model_type == "text":
            return [{
                "name": "predictions",
                "datatype": "FP32",
                "shape": [-1, 2],
            }]
        return [{"name": "output", "datatype": "FP32", "shape": [-1]}]
