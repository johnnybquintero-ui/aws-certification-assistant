import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class LanguageModel:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Load the tokenizer and pretrained language model."""

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
        )

        # Put the model into inference mode rather than training mode.
        self.model.eval()

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Generate text using the loaded language model."""

        # Keep the model's instructions separate from the user's input.
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        # Format and tokenise the messages using TinyLlama's chat template.
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        # Generate text without calculating gradients.
        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # The generated output also contains the original prompt tokens.
        # Remove them so that only TinyLlama's new response remains.
        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, prompt_length:]

        # Convert the generated token IDs back into readable text.
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    def extract_requirement(self, user_input: str) -> str:
        """Extract one concise technical requirement from the user's input."""

        return self._generate(
            system_prompt=(
                "You are a text rewriter for an AWS classifier. "
                "Rewrite the user's need as one short technical capability. "
                "Do not answer the request. Do not name or recommend an AWS "
                "service. Return one sentence only, without headings or bullets."
            ),
            user_prompt=user_input,
            max_new_tokens=60,
        )

    def generate_reply(
        self,
        original_input: str,
        extracted_requirement: str,
        predictions: list[tuple[str, float]],
    ) -> str:
        """Generate a response from the classifier's highest prediction."""

        if not predictions:
            raise ValueError("The classifier returned no predictions.")

        # The classifier returns predictions ordered by confidence.
        top_service, confidence = predictions[0]
        display_service = top_service.upper()

        # TinyLlama explains the classifier's selection but does not replace it.
        explanation = self._generate(
            system_prompt=(
                "You explain results produced by an AWS service classifier. "
                "Write one short conversational sentence explaining why the "
                "selected service might meet the requirement. Mention only the "
                "selected service. Do not change the classifier's selection. "
                "Do not use headings, bullet points or percentages. If the "
                "confidence is below 50%, make it clear that the match is "
                "uncertain."
            ),
            user_prompt=(
                f"Original request: {original_input}\n"
                f"Extracted requirement: {extracted_requirement}\n"
                f"Selected service: {display_service}\n"
                f"Confidence: {confidence:.1%}"
            ),
            max_new_tokens=60,
        )

        # Build the factual classifier result in Python so TinyLlama cannot
        # silently replace the selected service or confidence score.
        return (
            f"The classifier suggests {display_service} with "
            f"{confidence:.1%} confidence. {explanation}"
        )
