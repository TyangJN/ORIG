import openai
from openai import OpenAI
import base64
from pathlib import Path
import mimetypes
from typing import List, Union, Dict, Any


class MultimodalRetrievalClient:
    def __init__(self, api_key: str, model: str):
        # gpt-4o supports vision
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # Maintain conversation history for multi-turn mode
        self.message_history = [
            {"role": "system",
             "content": "You are a professional multimodal data web search expert, specialized in providing high-quality supporting data for image generation systems."}
        ]

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")

    def get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type of the image"""
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type and mime_type.startswith('image/'):
            return mime_type
        # Default to jpeg if can't determine
        return "image/jpeg"

    def create_message_content(self, text: str = None, image_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Create message content with text and/or images"""
        content = []

        # Add text if provided
        if text and text.strip():
            content.append({
                "type": "text",
                "text": text
            })

        # Add images if provided
        if image_paths:
            for image_path in image_paths:
                try:
                    if not Path(image_path).exists():
                        raise Exception(f"Image file not found: {image_path}")

                    base64_image = self.encode_image(image_path)
                    mime_type = self.get_image_mime_type(image_path)

                    content.append({
                        "type": "image_url",
                        "image_url": {
                            # "image_url": f"data:image/jpeg;base64,{base64_image}",
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"  # or "low" for faster processing
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {image_path}: {str(e)}")

        return content if content else [{"type": "text", "text": ""}]

    def send_message(self, text: str = None, image_paths: List[str] = None):
        """Send a multimodal message to the OpenAI API (MULTI-TURN mode)"""
        if not text and not image_paths:
            return "Error: Must provide either text or images"

        # Create message content
        content = self.create_message_content(text, image_paths)

        # Add user message to conversation history
        self.message_history.append({
            "role": "user",
            "content": content
        })

        try:
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                max_completion_tokens=8192
            )

            # Get assistant reply
            assistant_message = response.choices[0].message.content

            # Add assistant reply to conversation history
            self.message_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            # Remove the failed message from history
            self.message_history.pop()
            return f"Error: {str(e)}"

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

        try:
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=8192
            )

            # Return assistant reply directly (no history update)
            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

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
                    elif item["type"] == "image_url":
                        image_count += 1

                content_str = " ".join(text_parts)
                if image_count > 0:
                    content_str += f" [+ {image_count} image(s)]"
            else:
                content_str = content

            conversation_text += f"{role}: {content_str}\n\n"
        return conversation_text





