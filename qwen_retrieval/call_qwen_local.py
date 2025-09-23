import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
from typing import List, Union, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class MultimodalRetrievalClient:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        """
        Initialize Qwen2.5VL local client

        Args:
            model_path: Path to the model (local path or HuggingFace model name)
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.model_path = model_path

        # Auto-detect device if not specified
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                print("CUDA not available, using CPU")
        else:
            self.device = device

        print(f"Loading model: {model_path}")
        print("This may take a few minutes...")

        # Load model and tokenizer
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            ).eval()

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            print("Model loaded successfully!")

        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

        # Maintain conversation history for multi-turn mode
        self.message_history = [
            {"role": "system",
             "content": "You are a professional multimodal data web search expert, specialized in providing high-quality supporting data for image generation systems."}
        ]

    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path"""
        try:
            if not Path(image_path).exists():
                raise Exception(f"Image file not found: {image_path}")

            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {str(e)}")

    def create_message_content(self, text: str = None, image_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Create message content with text and/or images for Qwen2.5VL"""
        content = []

        # Add images first
        if image_paths:
            for image_path in image_paths:
                try:
                    image = self.load_image(image_path)
                    content.append({
                        "type": "image",
                        "image": image
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {image_path}: {str(e)}")

        # Add text
        if text and text.strip():
            content.append({
                "type": "text",
                "text": text
            })

        return content if content else [{"type": "text", "text": ""}]

    def _generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """Generate response using local Qwen2.5VL model"""
        try:
            # Convert messages to the format expected by Qwen2.5VL
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process images and text
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Trim input tokens and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return output_text.strip()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "Error: GPU out of memory. Try reducing image size or batch size."
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def send_message(self, text: str = None, image_paths: List[str] = None):
        """Send a multimodal message to local Qwen2.5VL model (MULTI-TURN mode)"""
        if not text and not image_paths:
            return "Error: Must provide either text or images"

        # Create message content
        content = self.create_message_content(text, image_paths)

        # Add user message to conversation history
        self.message_history.append({
            "role": "user",
            "content": content
        })

        # Generate response
        assistant_message = self._generate_response(self.message_history)

        if not assistant_message.startswith("Error:"):
            # Add assistant reply to conversation history
            self.message_history.append({
                "role": "assistant",
                "content": assistant_message
            })
        else:
            # Remove the failed message from history
            self.message_history.pop()

        return assistant_message

    def send_single_message(self, text: str = None, image_paths: List[str] = None,
                            system_prompt: str = None):
        """Send a single-turn message WITHOUT maintaining conversation history"""
        if not text and not image_paths:
            return "Error: Must provide either text or images"

        # Use custom system prompt or default one
        if system_prompt is None:
            system_prompt = self.message_history[0]["content"]

        # Create fresh message list for single turn (no history)
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Create message content
        content = self.create_message_content(text, image_paths)

        # Add user message
        messages.append({
            "role": "user",
            "content": content
        })

        # Generate and return response directly
        return self._generate_response(messages)

    # Multi-turn convenience methods
    def send_text_only(self, text: str):
        """Send text-only message (MULTI-TURN convenience method)"""
        return self.send_message(text=text)

    def send_image_only(self, image_paths: List[str]):
        """Send image-only message (MULTI-TURN convenience method)"""
        return self.send_message(image_paths=image_paths)

    def send_text_and_images(self, text: str, image_paths: List[str]):
        """Send text and images together (MULTI-TURN convenience method)"""
        return self.send_message(text=text, image_paths=image_paths)

    # Single-turn convenience methods
    def send_single_text(self, text: str, system_prompt: str = None):
        """Send single-turn text-only message"""
        return self.send_single_message(text=text, system_prompt=system_prompt)

    def send_single_image(self, image_paths: List[str], text: str = None, system_prompt: str = None):
        """Send single-turn image analysis"""
        return self.send_single_message(text=text, image_paths=image_paths, system_prompt=system_prompt)

    def send_single_text_and_images(self, text: str, image_paths: List[str], system_prompt: str = None):
        """Send single-turn text and images together"""
        return self.send_single_message(text=text, image_paths=image_paths, system_prompt=system_prompt)

    # Conversation management methods
    def get_message_history(self):
        """Return the complete conversation history"""
        return self.message_history

    def clear_conversation(self):
        """Clear conversation history while preserving system prompt"""
        self.message_history = [self.message_history[0]]

    def set_system_prompt(self, system_prompt):
        """Update the system prompt for multi-turn conversations"""
        self.message_history[0]["content"] = system_prompt

    def get_conversation_count(self):
        """Get the number of conversation turns (excluding system message)"""
        return len(self.message_history) - 1

    def export_conversation(self):
        """Export conversation in a readable format"""
        conversation_text = ""
        for message in self.message_history:
            role = message["role"].capitalize()
            content = message["content"]

            if isinstance(content, list):
                # Multimodal message
                text_parts = []
                image_count = 0
                for item in content:
                    if item["type"] == "text":
                        text_parts.append(item["text"])
                    elif item["type"] == "image":
                        image_count += 1

                content_str = " ".join(text_parts)
                if image_count > 0:
                    content_str += f" [+ {image_count} image(s)]"
            else:
                content_str = content

            conversation_text += f"{role}: {content_str}\n\n"
        return conversation_text

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")

    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        else:
            return "CUDA not available"


#  Example
if __name__ == "__main__":
    # Initialize client
    print("Initializing Qwen2.5VL client...")
    client = MultimodalRetrievalClient(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",  # or use local path
        device="auto"  # auto detect device
    )

    print(f"Memory usage: {client.get_memory_usage()}")

    # Example 1: Send pure text message
    print("\n=== Text Only Example ===")
    response = client.send_text_only("你好，请介绍一下你自己")
    print("Response:", response)

    # Example 2: Send image analysis request
    print("\n=== Image Analysis Example ===")
    # response = client.send_single_image(["path/to/your/image.jpg"], "请详细描述这张图片")
    # print("Image analysis:", response)

    # Example 3: Send text and image
    print("\n=== Text + Image Example ===")
    # response = client.send_text_and_images("这张图片中有什么？", ["path/to/image.jpg"])
    # print("Combined response:", response)

    # Clear GPU cache
    client.clear_gpu_cache()
    print(f"Final memory usage: {client.get_memory_usage()}")