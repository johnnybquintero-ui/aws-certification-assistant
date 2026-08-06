import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


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

    def generate_reply(
        self,
        original_input: str,
        predictions: list[tuple[str, float]],
        context: str,
        show_classifier: bool,
    ) -> str:
        """Generate a response using classification and RAG context."""

        if not predictions:
            raise ValueError("The classifier returned no predictions.")

        # The first prediction is the classifier's highest-scoring service.
        top_service, top_confidence = predictions[0]
        display_service = top_service.upper()

        classification_output = "\n".join(
            f"- {service.upper()}: {probability:.1%} confidence"
            for service, probability in predictions
        )

        if show_classifier:
            response_instructions = f"""
            The user is requesting an AWS service recommendation.

            The classifier's top suggestion is {display_service}.
            Explain why this service may suit the user's requirement, but only
            when that explanation is supported by the supplied context.

            Do not replace the classifier's top suggestion with another service.
            If the context does not support the suggestion, state that the
            available context is insufficient.

            CLASSIFIER OUTPUT:
            ==================

            {classification_output}
            """
        else:
            response_instructions = """
            The user is asking a general AWS certification question.

            Answer the user's question directly using the supplied context.
            Do not mention the classifier or its predictions.
            Do not turn the answer into a recommendation for one AWS service.
            """

        system_prompt = f"""
        You are a helpful AWS certification assistant.

        REQUEST INSTRUCTIONS:
        =====================

        {response_instructions}

        GROUNDING RULES:
        ================

        - Answer using only information supported by the supplied context.
        - Do not invent AWS services, features, facts, or exam content.
        - Treat classifier predictions as suggestions, not factual evidence.
        - Do not claim that an AWS service exists unless it is named in the context.
        - If the context does not contain enough information, clearly state that
        the available context is insufficient.
        - Answer the user's exact question.
        - Produce a complete answer and do not end mid-sentence.
        - Reply in two to four short, friendly and informative sentences.

        SUPPLIED CONTEXT:
        =================

        {context}

        END OF CONTEXT
        ==============
        """

        explanation = self._generate(
            system_prompt=system_prompt,
            user_prompt=original_input,
            max_new_tokens=250,
        )

        # Python controls whether the classifier result is shown to the user.
        if show_classifier:
            return (
                f"The classifier suggests {display_service} "
                f"with {top_confidence:.1%} confidence. "
                f"{explanation.strip()}"
            )

        return explanation.strip()
