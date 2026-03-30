import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


class TensorRTQwenRunner:
    def __init__(
        self,
        engine_dir: str = "~/edgellm_work/engines/qwen3-vl-2b",
        multimodal_engine_dir: str = "~/edgellm_work/visual_engines/qwen3-vl-2b",
        llm_inference_bin: str = "~/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
        plugin_path: str = "~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so",
        work_dir: str = "~/edgellm_work/runtime",
        system_prompt: str = "당신은 건설 현장 안전 도우미입니다.",
    ):
        self.engine_dir = os.path.expanduser(engine_dir)
        self.multimodal_engine_dir = os.path.expanduser(multimodal_engine_dir)
        self.llm_inference_bin = os.path.expanduser(llm_inference_bin)
        self.plugin_path = os.path.expanduser(plugin_path)
        self.work_dir = Path(os.path.expanduser(work_dir))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = system_prompt

        self._validate_paths()

    def _validate_paths(self) -> None:
        required_paths = {
            "llm_inference_bin": self.llm_inference_bin,
            "engine_dir": self.engine_dir,
            "multimodal_engine_dir": self.multimodal_engine_dir,
            "plugin_path": self.plugin_path,
        }

        missing = [
            f"{name}={path}"
            for name, path in required_paths.items()
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "TensorRT Qwen 실행에 필요한 경로가 없습니다:\n" + "\n".join(missing)
            )

    def _build_input_payload(
        self,
        image_path: str,
        user_text: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        return {
            "batch_size": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": int(max_new_tokens),
            "requests": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "text",
                                    "text": user_text,
                                },
                            ],
                        },
                    ]
                }
            ],
        }

    def _looks_like_path(self, text: str) -> bool:
        if not text:
            return False

        text = text.strip()
        lowered = text.lower()

        if lowered.endswith((".json", ".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            return True
        if lowered.startswith("/home/") or lowered.startswith("/tmp/"):
            return True
        if "/" in text and any(
            lowered.endswith(ext)
            for ext in (".json", ".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            return True

        return False

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            text = data.strip()
            if not text or self._looks_like_path(text):
                return ""
            return text

        if isinstance(data, dict):
            priority_keys = [
                "text",
                "content",
                "output_text",
                "generated_text",
                "response",
                "message",
                "answer",
            ]

            for key in priority_keys:
                if key in data:
                    text = self._extract_text(data[key])
                    if text:
                        return text

            # dict 전체를 무차별 탐색하면 input.json 경로 같은 문자열을
            # 잘못 반환할 수 있으므로 제한적으로만 탐색
            nested_keys = [
                "result",
                "results",
                "outputs",
                "choices",
                "data",
            ]
            for key in nested_keys:
                if key in data:
                    text = self._extract_text(data[key])
                    if text:
                        return text

            return ""

        if isinstance(data, list):
            parts = []
            for item in data:
                text = self._extract_text(item)
                if text:
                    parts.append(text)

            # 너무 많은 조각이 붙는 걸 막기 위해 유효한 텍스트만 합침
            merged = "\n".join(parts).strip()
            return merged

        return ""

    def _clean_output(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("Assistant:", "").replace("User:", "").strip()

        if self._looks_like_path(text):
            return ""

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if self._looks_like_path(line):
                continue
            lines.append(line)

        cleaned = " ".join(lines).strip()
        return cleaned

    def infer(self, image_input, user_text: str, max_new_tokens: int = 64) -> str:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = self.plugin_path

        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmpdir:
            tmpdir_path = Path(tmpdir)

            image_path = tmpdir_path / "frame.jpg"
            input_path = tmpdir_path / "input.json"
            output_path = tmpdir_path / "output.json"

            image.save(image_path, quality=95)

            payload = self._build_input_payload(
                image_path=str(image_path),
                user_text=user_text,
                max_new_tokens=max_new_tokens,
            )
            input_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            cmd = [
                self.llm_inference_bin,
                "--engineDir", self.engine_dir,
                "--multimodalEngineDir", self.multimodal_engine_dir,
                "--inputFile", str(input_path),
                "--outputFile", str(output_path),
            ]

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "TensorRT Qwen 추론 실패\n"
                    f"returncode={result.returncode}\n"
                    f"stdout=\n{result.stdout}\n"
                    f"stderr=\n{result.stderr}"
                )

            if not output_path.exists():
                raise RuntimeError(
                    "TensorRT Qwen 출력 파일이 생성되지 않았습니다.\n"
                    f"stdout=\n{result.stdout}\n"
                    f"stderr=\n{result.stderr}"
                )

            raw = output_path.read_text(encoding="utf-8")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return self._clean_output(raw)

        extracted = self._extract_text(parsed)
        return self._clean_output(extracted)