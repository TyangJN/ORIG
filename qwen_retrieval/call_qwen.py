import os
import base64
from pathlib import Path
import mimetypes
from typing import List, Union, Dict, Any, Optional
from openai import OpenAI


class MultimodalRetrievalClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-vl-max-latest",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        Initialize Qwen Vision Language client using OpenAI compatible interface

        Args:
            api_key: DashScope API key (if None, will use DASHSCOPE_API_KEY environment variable)
            model: Model name (default: qwen-vl-max-latest)
            base_url: API base URL for DashScope compatible mode
        """
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url
        )

        # Default system prompt for multimodal analysis
        self.default_system_prompt = "You are a professional multimodal data web search expert, specialized in providing high-quality supporting data for image generation systems."

        # Maintain conversation history for multi-turn mode
        self.message_history = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.default_system_prompt}]
            }
        ]

    def encode_image(self, image_path: str) -> str:
        """Encode local image to base64 string"""
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

    def create_message_content(self, text: Optional[str] = None,
                               image_paths: Optional[List[str]] = None,
                               image_urls: Optional[List[str]] = None,
                               video_mode: bool = False) -> List[Dict[str, Any]]:
        """Create message content with text and/or images for Qwen VL models"""
        content = []

        # Handle video mode (multiple images as video sequence)
        if video_mode and (image_paths or image_urls):
            video_frames = []

            # Process local image files for video
            if image_paths:
                for image_path in image_paths:
                    try:
                        if not Path(image_path).exists():
                            raise Exception(f"Image file not found: {image_path}")

                        base64_image = self.encode_image(image_path)
                        mime_type = self.get_image_mime_type(image_path)
                        video_frames.append(f"data:{mime_type};base64,{base64_image}")
                    except Exception as e:
                        print(f"Warning: Could not process image {image_path}: {str(e)}")

            # Process image URLs for video
            if image_urls:
                video_frames.extend(image_urls)

            if video_frames:
                content.append({
                    "type": "video",
                    "video": video_frames
                })
        else:
            # Handle single images mode (original behavior)
            # Add images first (from local files)
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
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        })
                    except Exception as e:
                        print(f"Warning: Could not process image {image_path}: {str(e)}")

            # Add images from URLs
            if image_urls:
                for image_url in image_urls:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    })

        # Add text content
        if text and text.strip():
            content.append({
                "type": "text",
                "text": text
            })

        return content if content else [{"type": "text", "text": ""}]

    def _make_api_call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Make API call to Qwen VL model using OpenAI interface"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 8192),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature', 'top_p']}
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

    def send_message(self, text: Optional[str] = None,
                     image_paths: Optional[List[str]] = None,
                     image_urls: Optional[List[str]] = None, **kwargs):
        """Send a multimodal message to Qwen VL API (MULTI-TURN mode)"""
        if not text and not image_paths and not image_urls:
            return "Error: Must provide either text, image paths, or image URLs"

        # Create message content
        content = self.create_message_content(text, image_paths, image_urls)

        # Add user message to conversation history
        self.message_history.append({
            "role": "user",
            "content": content
        })

        # Make API call
        assistant_message = self._make_api_call(self.message_history, **kwargs)

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

    def send_single_message(self, text: Optional[str] = None,
                            image_paths: Optional[List[str]] = None,
                            image_urls: Optional[List[str]] = None,
                            system_prompt: Optional[str] = None, **kwargs):
        """Send a single-turn message WITHOUT maintaining conversation history"""
        if not text and not image_paths and not image_urls:
            return "Error: Must provide either text, image paths, or image URLs"

        # Use custom system prompt or default one
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        # Create fresh message list for single turn (no history)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }
        ]

        # Create message content
        content = self.create_message_content(text, image_paths, image_urls)

        # Add user message
        messages.append({
            "role": "user",
            "content": content
        })

        # Make API call and return result directly
        return self._make_api_call(messages, **kwargs)

    # Multi-turn convenience methods
    def send_text_only(self, text: str, **kwargs):
        """Send text-only message (MULTI-TURN convenience method)"""
        return self.send_message(text=text, **kwargs)

    def send_image_only(self, image_paths: List[str] = None, image_urls: List[str] = None, **kwargs):
        """Send image-only message (MULTI-TURN convenience method)"""
        return self.send_message(image_paths=image_paths, image_urls=image_urls, **kwargs)

    def send_text_and_images(self, text: str, image_paths: List[str] = None,
                             image_urls: List[str] = None, **kwargs):
        """Send text and images together (MULTI-TURN convenience method)"""
        return self.send_message(text=text, image_paths=image_paths, image_urls=image_urls, **kwargs)

    def send_url_image(self, image_url: str, text: Optional[str] = None, **kwargs):
        """Send single URL image with optional text (MULTI-TURN convenience method)"""
        return self.send_message(text=text, image_urls=[image_url], **kwargs)

    # Single-turn convenience methods
    def send_single_text(self, text: str, system_prompt: Optional[str] = None, **kwargs):
        """Send single-turn text-only message"""
        return self.send_single_message(text=text, system_prompt=system_prompt, **kwargs)

    def send_single_image(self, image_paths: List[str] = None, image_urls: List[str] = None,
                          text: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs):
        """Send single-turn image analysis"""
        return self.send_single_message(text=text, image_paths=image_paths,
                                        image_urls=image_urls, system_prompt=system_prompt, **kwargs)

    def send_single_text_and_images(self, text: str, image_paths: List[str] = None,
                                    image_urls: List[str] = None, system_prompt: Optional[str] = None, **kwargs):
        """Send single-turn text and images together"""
        return self.send_single_message(text=text, image_paths=image_paths,
                                        image_urls=image_urls, system_prompt=system_prompt, **kwargs)

    def send_single_url_image(self, image_url: str, text: Optional[str] = None,
                              system_prompt: Optional[str] = None, **kwargs):
        """Send single-turn URL image analysis"""
        return self.send_single_message(text=text, image_urls=[image_url],
                                        system_prompt=system_prompt, **kwargs)

    # Conversation management methods
    def get_message_history(self):
        """Return the complete conversation history"""
        return self.message_history

    def clear_conversation(self):
        """Clear conversation history while preserving system prompt"""
        self.message_history = [self.message_history[0]]

    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for multi-turn conversations"""
        self.message_history[0]["content"][0]["text"] = system_prompt
        self.default_system_prompt = system_prompt

    def get_conversation_count(self):
        """Get the number of conversation turns (excluding system message)"""
        return (len(self.message_history) - 1) // 2

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

    # Video sequence analysis (using image list to simulate video frames)
    def analyze_video_sequence(self, image_paths: List[str], text: Optional[str] = None,
                               fps: int = 2, system_prompt: Optional[str] = None, **kwargs):
        """
        Analyze a sequence of images as video frames

        Args:
            image_paths: List of image paths representing video frames
            text: Optional question about the video
            fps: Frames per second (for context)
            system_prompt: Optional system prompt
        """
        if not image_paths:
            return "Error: Must provide image paths for video analysis"

        prompt = text or "这段视频描绘的是什么景象？"
        if fps and fps > 0:
            prompt = f"[这些图像是以{fps}fps采样的视频帧] {prompt}"

        return self.send_single_message(
            text=prompt,
            image_paths=image_paths,
            system_prompt=system_prompt,
            **kwargs
        )


# 使用示例
if __name__ == "__main__":
    # 初始化客户端 (API key 从环境变量获取)
    client = MultimodalRetrievalClient(
        model="qwen-vl-max-latest"  # 或者使用其他Qwen VL模型
    )

    # 示例1: 发送纯文本消息
    print("=== 示例1: 纯文本 ===")
    response = client.send_text_only("Hello, how are you?")
    print("Text response:", response)

    # 示例2: 发送URL图片分析请求
    print("\n=== 示例2: URL图片分析 ===")
    response = client.send_single_url_image(
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg",
        "图中描绘的是什么景象？"
    )
    print("Image analysis:", response)

    # 示例3: 多轮对话
    print("\n=== 示例3: 多轮对话 ===")
    client.clear_conversation()  # 清除之前的对话历史

    # 第一轮：图片分析
    response1 = client.send_url_image(
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg",
        "图中描绘的是什么景象？"
    )
    print("第一轮输出：", response1)

    # 第二轮：基于图片写诗
    response2 = client.send_text_only("做一首诗描述这个场景")
    print("第二轮输出：", response2)

    # 示例4: 本地图片分析 (需要本地图片文件)
    # print("\n=== 示例4: 本地图片分析 ===")
    # response = client.send_single_image(
    #     image_paths=["path/to/your/local/image.jpg"],
    #     text="请分析这张图片的内容"
    # )
    # print("Local image analysis:", response)

    # 示例5: 视频序列分析 (需要多个图片文件)
    # print("\n=== 示例5: 视频序列分析 ===")
    # video_frames = ["frame1.jpg", "frame2.jpg", "frame3.jpg", "frame4.jpg"]
    # response = client.analyze_video_sequence(
    #     image_paths=video_frames,
    #     text="这段视频描绘的是什么景象？",
    #     fps=2
    # )
    # print("Video analysis:", response)

    # 查看对话历史
    print("\n=== 对话历史 ===")
    print(f"对话轮数: {client.get_conversation_count()}")
    print("完整对话:")
    print(client.export_conversation())