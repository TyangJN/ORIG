import openai
import json
import requests
import re
import time
from PIL import Image
from io import BytesIO
import os

import retrieval_prompt as prompt
import retrieval_prompt_img as prompt_img
import retrieval_prompt_txt as prompt_txt

from serpapi import GoogleSearch
from gpt_retrieval.call_gpt import *


OPENAI_API_KEY = ""
SERPER_API_KEY = ""
SERPAPI_API_KEY = ""
JINA_API_KEY = ""

def parse_input(user_input: str) -> tuple:
    """Parse user input to extract text and image paths"""
    parts = user_input.strip().split()
    text_parts = []
    image_paths = []

    for part in parts:
        # Check if part looks like a file path and is an image
        if ('.' in part and
                any(part.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']) and
                Path(part).exists()):
            image_paths.append(part)
        else:
            text_parts.append(part)

    text = " ".join(text_parts) if text_parts else None
    return text, image_paths


def generate_warm_up_plan(input_prompt):
    chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5-mini")

    plan = chat_client.send_single_message(text=prompt.warm_up_search.format(input_prompt),
                                               system_prompt=prompt.warm_up_search_system)

    queries = []
    for line in plan.split('\n'):
        if 'Text Retrieval:' in line:
            queries.append(line.split('Text Retrieval:')[1].strip())
    
    chat_client.clear_conversation()
    
    # questions = []
    # queries = []
    # in_subq = False

    # for line in plan.splitlines():
    #     line = line.strip()
    #     if not line:
    #         continue

    #     if line.startswith("<Sub-Questions>"):
    #         in_subq = True
    #         continue
    #     if line.startswith("</Sub-Questions>"):
    #         in_subq = False
    #         continue

    #     if in_subq:
    #         questions.append(line)
    #     elif line.startswith("Text Retrieval:"):
    #         queries.append(line.split("Text Retrieval:")[1].strip())

    return plan, queries


def search_t2t_serper(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "gl": "sg"
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)

    data = json.loads(response.text)

    results = data.get("organic", [])

    return results


def search_t2t_serpapi(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]

    return organic_results


def web_select(webs, query):
    webs_info = f"""
    Based on the search query "{query}", select the 1-3 most relevant websites from the list below.
    Prioritize selecting 1 website unless multiple sources are clearly needed.

    Your selection should be based on:
    - Title relevance to the query
    - Description content matching the information needed
    
    Available websites:
    """

    for id, web in enumerate(webs):
        title = web.get('title', 'No title')
        snippet = web.get('snippet', 'No snippet')

        webs_info += f"web_{id}: Title: {title}\nDescription: {snippet}\n\n"

    webs_info += """Please respond with only the website IDs you select (e.g., "web_0" or "web_0, web_2").
    Focus on quality over quantity - prefer 1 highly relevant site over multiple less relevant ones."""

    webs_info += """IMPORTANT: Respond with ONLY the IDs of the websites you select, formatted as a JSON list.
    Example: ["web_0"] or ["web_0", "web_2"]
    Do not include any explanations or extra text.
    """

    system_prompt = "You are a helpful assistant."

    chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5-mini")


    selected_ids = chat_client.send_single_message(text=webs_info,
                                           system_prompt=system_prompt)

    ids = [int(re.search(r"\d+", x).group()) for x in selected_ids if re.search(r"\d+", x)]

    chat_client.clear_conversation()

    return ids


def extract_t2t_results(web_ids, webs, input_prompt, query):
    # System prompt + context
    system_prompt = f"""
    You are given multiple pieces of web content retrieved for the search query: "{query}".
    Your task is to extract and synthesize only the key factual information that answers the query.

    Focus on:
    - Basic definitions and descriptions
    - Key characteristics and attributes  
    - Essential facts relevant to the image generation prompt: "{input_prompt}"

    Combine information from all sources into a single, concise summary with only the most important facts needed for accurate image generation.
    Do NOT include URLs, navigation text, advertisements, or irrelevant content.
    """

    webs_info = """
    Available websites content:
    """

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Md-Link-Style": "discarded",
        "X-Retain-Images": "none",
        "X-Return-Format": "markdown"
    }

    successful_fetches = 0
    target_fetches = len(web_ids)  # 目标获取数量

    tried_ids = set()
    for web_id in web_ids:
        if successful_fetches >= target_fetches:
            break

        tried_ids.add(web_id)
        content = fetch_single_web_content(web_id, webs, headers)

        if content:
            webs_info += f"\n---\nThe searched content for website_{successful_fetches+1}: \n{content}"
            successful_fetches += 1

    if successful_fetches < target_fetches:
        # print(f"Only got {successful_fetches}/{target_fetches} sources, trying fallback...")
        for web_id in range(len(webs)):
            if successful_fetches >= target_fetches:
                break
            if web_id in tried_ids:
                continue

            tried_ids.add(web_id)
            content = fetch_single_web_content(web_id, webs, headers)

            if content:
                webs_info += f"\n---\nThe searched content for website_{successful_fetches + 1}: \n{content}"
                successful_fetches += 1

    if successful_fetches == 0:
        return "No content available - all sources failed to load."

    chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5-mini")

    result = chat_client.send_single_message(text=webs_info,
                                                   system_prompt=system_prompt)
    
    chat_client.clear_conversation()

    return result


def fetch_single_web_content(web_id, webs, headers):
    if web_id >= len(webs):
        print(f"[WARN] web_id {web_id} out of range")
        return None

    link = webs[web_id].get("link")
    if not link:
        print(f"[WARN] No link found for web_id {web_id}")
        return None

    try:
        url = f"https://r.jina.ai/{link}"
        response = requests.get(url, headers=headers, timeout=50)
        response.raise_for_status()
        content = response.text.strip()

        blocked_indicators = [
            "Verify you are human", "Cloudflare", "Just a moment...",
            "Ray ID:", "403 Forbidden", "Access Denied"
        ]

        if any(indicator in content for indicator in blocked_indicators):
            print(f"[WARN] Blocked content detected for web_id {web_id}")
            return None

        if len(content) < 200:
            print(f"[WARN] Content too short for web_id {web_id}")
            return None

        return content

    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch web_id {web_id}: {e}")
        return None


def round_t2t_results(queries, input_prompt, round_id):
    sub_queries_data = []

    results = f"For image generation prompt: {input_prompt}, the model needs to know the following information for background knowledge:\n\n"

    for idx, query in enumerate(queries):
        webs = search_t2t_serper(query)
        web_ids = web_select(webs, query)
        sub_query_result = extract_t2t_results(web_ids, webs, input_prompt, query)
        
        results += f"{sub_query_result}\n\n"

        webs_data = []

        for web_id in web_ids:
            if not webs[web_id]["title"]:
                webs[web_id]["title"] = "No title"
            if not webs[web_id]["snippet"]:
                webs[web_id]["snippet"] = "No snippet"

            web_data = {
                "url": webs[web_id]["link"],
                "title": webs[web_id]["title"],
                "content": webs[web_id]["snippet"],
            }

            webs_data.append(web_data)

        sub_queries_data.append({
            "sub_query_id": f"r{round_id}_sq{idx}_txt",
            "query": query,
            "search_type": "text",
            "search_results": webs,
            "selected_webs": webs_data,
            "sub_query_result": sub_query_result,
        })
    

    return sub_queries_data, results


def round_t2t_summary(plan, results):
    system_prompt = """
    You are given a search plan and its corresponding retrieval results.
    Your task:
    1. Extract only the most relevant information that directly addresses the search plan questions.
    2. Remove redundant, low-value, or tangential details.
    3. Write a clear, concise summary (4–8 sentences max), prefer 4-6 sentences.
    4. Focus on information that improves visual completeness for image generation.
    """

    round_input = f"<SearchPlan>\n{plan}\n</SearchPlan>\n\n<Results>\n{results}\n</Results>"

    chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5-mini")

    summary = chat_client.send_single_message(
        text=round_input,
        system_prompt=system_prompt
    )

    chat_client.clear_conversation()

    return summary


def search_t2i_serper(query):
    url = "https://google.serper.dev/images"

    payload = json.dumps({
    "q": query,
    })
    headers = {
    'X-API-KEY': '11bf3fffbd608b0d5b13bbfb1a50671cc41207fb',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    data = json.loads(response.text)

    results = data.get("images", [])

    return results


def search_t2i_serpapi(query, retry_attempts=3):
    params = {
        "engine": "google_images",
        "q": query,
        "location": "Austin, TX, Texas, United States",
        "api_key": SERPAPI_API_KEY
    }


    for i in range(retry_attempts):
        try:
            search = GoogleSearch(params)
            results = search.get_dict()  # 获取图片搜索结果
            organic_results = results["images_results"]
            print(organic_results[0])
            return organic_results

        except Exception as e:
            print(f"Attempt {i + 1} failed: {e}")
            if i < retry_attempts - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print("All retries failed.")
                return False


def download_imgs(img_urls, round_id, sq_id, background_knowledge, save_dir="downloads"):
    """
    Download image files from a list of image dictionaries.
    """
    img_data = []
    for idx, data in enumerate(img_urls):
        if len(img_data) > 3:
            return img_data
        img_url = data.get("imageUrl")
        if not img_url:
            continue
        try:
            response = requests.get(img_url,timeout=10)
            response.raise_for_status()  
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)
            img_path = save_dir + f"round{round_id}_sq{sq_id}_{idx}.png"
            image.save(img_path, format='PNG')

            meta_data = {
                "img_id": f"r{round_id}_sq{sq_id}_{idx}",
                "img_url": data.get("imageUrl"),
                "img_path": img_path,
                "title": data.get("title"),
                "imageWidth": data.get("imageWidth"),
                "imageHeight": data.get("imageHeight"),
                "thumbnailUrl": data.get("thumbnailUrl"),
                "thumbnailWidth": data.get("thumbnailWidth"),
                "thumbnailHeight": data.get("thumbnailHeight"),
                "source": data.get("source"),
                "domain": data.get("domain"),
                "link": data.get("link"),
                "googleUrl": data.get("googleUrl"),
            }

            img_data.append(meta_data)
        
        except Exception as e:
            print(f"Failed to download or save image from {img_url}: {e}")

    return img_data


def extract_t2i_results(img_data, input_prompt, query, background_knowledge):
    system_prompt = f"""
    CRITICAL INSTRUCTIONS:
    You are given a list of images and the search query: {query}.
    Select the SINGLE most relevant image based on the input prompt: {input_prompt} and background knowledge: {background_knowledge}.

    Selection Criteria:
    - Relevance: Must directly match the search intent and depict the key entity, scene, or style requested.
    - Clarity: Prefer visually clear, high-quality, representative images.
    - Informativeness: Prefer images that show distinctive visual details useful for generation.
    - Avoid duplicates, icons, watermarks, or low-quality variants.

    MANDATORY OUTPUT FORMAT:

    <reason>
    Write 1–2 sentences briefly explaining why you selected this image (focus on relevance and clarity).
    </reason>

    <image_id>
    INDEX
    </image_id>

    Where INDEX is a 0-based integer referring to the image list order.
    Do NOT output anything outside these two sections.
    """

    img_info = "\nAvailable Images:\n"
    imgs = []

    for idx, data in enumerate(img_data):
        title = data.get("title", "No Title")
        img_info += f"\n The title for <image_{idx}>: {title}"
        imgs.append(data.get("img_path"))

    

    chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5-mini")

    result = chat_client.send_single_message(text=img_info,image_paths=imgs,
                                                system_prompt=system_prompt)

    match = re.search(r"<image_id>\s*(\d+)\s*</image_id>", result)
    if match:
        selected_idx = int(match.group(1))
    else:
        selected_idx = 0 
    

    chat_client.clear_conversation()

    return result, selected_idx

    
def round_t2i_results(queries, input_prompt, round_id, background_knowledge, save_dir="downloads"):
    sub_queries_data = []
    img_results = []

    for idx, query in enumerate(queries):
        img_urls = search_t2i_serper(query)

        img_data = download_imgs(img_urls=img_urls,
                                 round_id=round_id,
                                 sq_id=idx,
                                 background_knowledge=background_knowledge,
                                 save_dir=save_dir)

        sub_query_result, selected_idx = extract_t2i_results(img_data, input_prompt, query, background_knowledge)

        sub_queries_data.append({
            "sub_query_id": f"r{round_id}_sq{idx}_img",
            "query": query,
            "search_type": "image",
            "search_results": img_data,
            "selected_reason": sub_query_result,
            "sub_query_result": {
                "image_id": f"r{round_id}_sq{idx}_{selected_idx}",
                "image_path": img_data[selected_idx]["img_path"],
            }
        })

        img_results.append({
            "title": img_data[selected_idx]["title"],
            "query": query,
            "img_path": img_data[selected_idx]["img_path"],
        })

    return sub_queries_data, img_results


def coarse_filtered(input_prompt, txt_results, img_title, img_path):

    txt_content = ""
    img_content = ""
    
    for txt_result in txt_results:
        txt_content += f"\n{txt_result}"

    for idx, img in enumerate(img_title):
        img_content += f"The Title for <image_{idx}>\n{img}"

    loop_coarse_filtered_input = prompt.loop_coarse_filtered.format(input_prompt, txt_content, img_content)

    chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5")

    result = chat_client.send_single_message(text=loop_coarse_filtered_input, image_paths=img_path,
                                                system_prompt=prompt.loop_coarse_filtered_system)

    chat_client.clear_conversation()

    return result


