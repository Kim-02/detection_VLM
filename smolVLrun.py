import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration


class SmolVLMRunner:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            _attn_implementation="eager",
        ).to(self.device)

        self.model.eval()

    def _clean_output(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()
        text = text.replace("Assistant:", "")
        text = text.replace("User:", "")
        text = text.replace("<end_of_utterance>", "")
        text = text.strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = " ".join(lines).strip()

        # 지시문 복사처럼 보이는 경우 제거
        bad_prefixes = [
            "1~2문장만 답하세요",
            "1문장만 답하세요",
            "한국어로",
            "이미지는",
            "작업자 수는",
        ]
        for prefix in bad_prefixes:
            if text.startswith(prefix):
                return ""

        return text

    @torch.no_grad()
    def infer(self, image_input, user_text: str, max_new_tokens: int = 32) -> str:
        """
        image_input:
        - str: 이미지 파일 경로
        - PIL.Image.Image: PIL 이미지 객체
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_only = generated_ids[:, input_length:]

        text = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._clean_output(text)