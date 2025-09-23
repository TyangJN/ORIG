from qwen_retrieval.pipeline import *
import os
import json
from concurrent.futures import ThreadPoolExecutor
import argparse

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_prompt_to_json(idx, modality, prompt_content, base_path):
    """Save prompt to prompts.json file in the specified idx directory"""

    json_file_path = f'{base_path}/prompts.json'

    # Read existing data
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
    else:
        data = {}

    # Save prompt directly with modality as key
    data[modality] = prompt_content

    # Save to file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_line(data_item, search_results_path, gen_results_path, gen_model, max_rounds, modality):
    input_prompt = data_item['prompt_en']
    idx = data_item['id']

    try:
        # Generate prompt for corresponding modality
        if modality == 'mm':
        ensure_directory(f'{search_results_path}/{idx}/{modality}')
            content = web_search(idx, input_prompt, f'{search_results_path}/{idx}', max_rounds, modality)
            prompt = content_refine(idx, input_prompt, f'{search_results_path}/{idx}/{modality}', content)

        elif modality == 'txt':
            ensure_directory(f'{search_results_path}/{idx}/{modality}')
            content = web_search(idx, input_prompt, f'{search_results_path}/{idx}', max_rounds, modality)
            prompt = content_refine_txt(idx, input_prompt, f'{search_results_path}/{idx}/{modality}', content)

        elif modality == 'img':
            ensure_directory(f'{search_results_path}/{idx}/{modality}')
            content = web_search(idx, input_prompt, f'{search_results_path}/{idx}', max_rounds, modality)
            prompt = content_refine_img(idx, input_prompt, f'{search_results_path}/{idx}/{modality}', content)

        elif modality == 'cot':
            prompt = cot(input_prompt)

        elif modality == 'dir':
            prompt = input_prompt
        else:
            print(f"Unknown modality: {modality}")

        # 保存当前modality的prompt到JSON
        save_prompt_to_json(idx, modality, prompt, f'{search_results_path}/{idx}')
        print(f"Saved {modality} prompt for idx {idx}")

        # 执行图像生成（使用所有可用的prompts）
        img_generation(idx, prompt, f'{gen_results_path}', gen_model, modality)

    except Exception as e:
        print(f"Error processing {idx}: {e}")


def main(search_results_path, gen_results_path, dataset, gen_model, max_rounds, modality):
    # Store web-search results
    ensure_directory(search_results_path)

    # Store multimodel based image generation results
    ensure_directory(gen_results_path)

    with open(dataset, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    with ThreadPoolExecutor(max_workers=10) as executor:
        for data_item in data_lines:
            executor.submit(process_line, data_item, search_results_path, gen_results_path, gen_model, max_rounds, modality)


if __name__ == "__main__":
    """
    Qwen-based Retrieval Pipeline!!!!
    
    This is a parallel processing script for executing search and generation tasks.
    It uses ThreadPoolExecutor to process each data item in the dataset in parallel.
    The processing of each data item includes:
    1. Execute search tasks
    2. Generate prompts for corresponding modality
    3. Save current modality prompt to JSON
    4. Execute image generation (using all available prompts)
    
    Parameter descriptions:
    --search_model: Search model name, options are "qwen".
    --gen_model: Generation model name, options are "gemini_gen", "openai_gen" or "qwen_gen".
    --dataset: Dataset path, options are "data/FIG-Eval/prompts.jsonl" or "data/FIG-Eval/prompts_tiny.jsonl".
    --meta_path: Result save path.
    --max_rounds: Maximum number of search rounds.
    --modality: Task type, options are "mm", "txt", "img" or "cot".
    
    Usage example:
    python main_mp.py --search_model qwen --gen_model openai_gen --dataset data/FIG-Eval/prompts.jsonl --meta_path data --max_rounds 3 --modality mm
    
    Notes:
    1. The choice of search model and generation model will affect the execution effectiveness of tasks.
    2. Dataset path and result save path need to be adjusted according to actual situation.
    3. Maximum search rounds determine the depth of search, can be adjusted according to actual needs.
    4. Task type determines the type of generation task, can be adjusted according to actual needs.
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run retrieval + generation pipeline")

    parser.add_argument(
        "--search_model",
        type=str,
        default="qwen",
        choices=["qwen"],
        help="Retrieval backbone model name. Choose from: 'qwen'."
    )

    parser.add_argument(
        "--gen_model",
        type=str,
        default="openai_gen",
        choices=["gemini_gen", "openai_gen", "qwen_gen", "flux_context"],
        help="Generation model to use. Options: 'gemini_gen', 'openai_gen', 'qwen_gen', 'flux_context'"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(BASE_DIR, "data/FIG-Eval/prompts.jsonl"),
        # default=os.path.join(BASE_DIR, "data/FIG-Eval/prompts_tiny.jsonl"),
        help="Path to dataset file (.jsonl)"
    )

    parser.add_argument(
        "--meta_path",
        type=str,
        default=os.path.join(BASE_DIR, "data"),
        help="Output directory for results"
    )

    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="max number of retrieval rounds"
    )

    parser.add_argument(
        "--modality",
        type=str,
        default='mm',
        help="choose from 'mm','txt','img', 'cot','cot'"
    )

    args = parser.parse_args()
    meta_path = f"{args.meta_path}/{args.search_model}_based_search"
    meta_path_search = f"{meta_path}/search_content"
    meta_path_gen = f"{meta_path}/{args.gen_model}"
    main(meta_path_search, meta_path_gen, args.dataset, args.gen_model, args.max_rounds, args.modality)
