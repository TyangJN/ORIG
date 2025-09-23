import os
import json

from openpyxl.styles.builtins import output

from qwen_retrieval.search_component import *
from qwen_retrieval.call_qwen import *

from utils.gen_component import *
from utils.search_results_manager import SearchResultsManager

from main_qwen_mp import *

QWEN_API_KEY = "YOUR_QWEN_API_KEY"

def warm_up_search(idx, input_prompt, search_results_path):
    # Record warm-up results    
    results_manager = SearchResultsManager(search_results_path, session_id=idx, warm_up=True)

    # Step 1: Initialize Warm-up session
    data = results_manager.initialize_session(input_prompt)

    # Step 2: Check current progress (resume supported)
    progress = results_manager.get_current_progress()

    if progress["session_exists"] and progress["completed_rounds"] > 0:
        # Read rounds data and append to results
        results = ""
        for round in data[idx]["rounds"]:
            results += round["round_result"]
        return results
    else:
        round_id = 1

        plan, queries = generate_warm_up_plan(input_prompt)

        sub_queries_data, results = round_t2t_results(queries, input_prompt, round_id)

        results = round_t2t_summary(plan, results)

        results_manager.add_round(round_id=round_id,
                                  round_plan=plan,
                                  round_result=results,
                                  sub_queries_data=sub_queries_data)

        # for sub_query_data in sub_queries_data:
        #     results_manager.add_sub_queries(round_id, plan, results, sub_query_data)

        results_final = [results]

        results_manager.finalize_session(results_final)
    
    return results


def web_search(idx, input_prompt, search_results_path, max_rounds, modality='mm'):
    # Initialize Warm-up
    warm_up_results = warm_up_search(idx, input_prompt, search_results_path)

    search_results_path = f'{search_results_path}/{modality}'

    if modality == 'mm':
        loop_results = search_loop(warm_up_results, max_rounds, idx, input_prompt, search_results_path)
    elif modality == 'txt':
        loop_results = search_loop_txt(warm_up_results, max_rounds, idx, input_prompt, search_results_path)
    elif modality == 'img':
        loop_results = search_loop_img(warm_up_results, max_rounds, idx, input_prompt, search_results_path)
    else:
        print('invalid modality, choose mm, txt, img')

    return loop_results


def search_loop(warm_up_results, max_rounds, idx, input_prompt, search_results_path):

    results_manager = SearchResultsManager(search_results_path, session_id=idx, warm_up=False)

    # Step 1: Initialize Warm-up session
    data = results_manager.initialize_session(input_prompt)

    # Step 2: Check current progress (resume supported)
    progress = results_manager.get_current_progress()

    # Step 3.1: Check if the search loop is completed, if yes, return the results
    if progress["status"] == "completed":
        results = data[idx]["final_result"]
        return results

    else:
        # Step 3.2: If the search loop is not completed, initialize the search loop
        ensure_directory(f'{search_results_path}/pics/')
        img_save_dir = f'{search_results_path}/pics/'

        chat_client = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

        background_knowledge = f"""
        The background knowledge for the image generation prompt {input_prompt} is: \n\n
        {warm_up_results}
        """

        img_title = ""
        img_path = []
        img_path_final = []
        img_title_final = []
        txt_results_final = []

        for round_id in range(1, max_rounds + 1):
            round_data = []
            round_results = {}

            # Step 3.2.1: If the round is the first round, let the MLLM explore the relevant information
            if round_id == 1:
                loop_input = prompt.loop_search_first.format(input_prompt, background_knowledge)
                chat_client.set_system_prompt(prompt.loop_search_frist_system)
            else:
                # Step 3.2.2: If the round is not the first round, let the MLLM judge if the sufficient information is found, if not, let the MLLM explore the relevant information
                loop_input = prompt.loop_search_else.format(input_prompt, background_knowledge, img_title)
                chat_client.set_system_prompt(prompt.loop_search_else_system)

            # Step 3.2.3: Let the MLLM generate the search plan
            plan = chat_client.send_message(text=loop_input, image_paths=img_path)

            # Step 3.2.4: If the MLLM judges that the sufficient information is found, stop the search loop
            if "SUFFICIENT" in plan:
                print(f"Round {round_id}: Assessment is SUFFICIENT, stopping search loop")
                break

            # Step 3.2.5: Let the MLLM generate the search plan
            img_queries = []
            txt_queries = []

            # Step 3.2.6: recognize the text and image queries
            for line in plan.split('\n'):
                if 'Text Retrieval:' in line:
                    txt_queries.append(line.split('Text Retrieval:')[1].strip())
                if 'Image Retrieval:' in line:
                    img_queries.append(line.split('Image Retrieval:')[1].strip())

            # Step 3.2.7: If the text queries are not empty, let the MLLM summarize the textual retrieval results
            if len(txt_queries) > 0:
                txt_sub_queries_data, txt_results = round_t2t_results(txt_queries, input_prompt, round_id)
                txt_results = round_t2t_summary(plan, txt_results)
                txt_results_final.append(txt_results)
                background_knowledge = txt_results

                round_results["txt_results"] = txt_results

                for txt_sub_query_data in txt_sub_queries_data:
                    round_data.append(txt_sub_query_data)
                    # results_manager.add_sub_queries(round_id, plan, txt_results, txt_sub_query_data)

            # Step 3.2.8: If the image queries are not empty, let the MLLM summarize the image retrieval results
            if len(img_queries) > 0:
                img_sub_queries_data, img_results = round_t2i_results(queries=img_queries,
                                                                      input_prompt=input_prompt,
                                                                      round_id=round_id,
                                                                      background_knowledge=background_knowledge,
                                                                      save_dir=img_save_dir)

                round_results["img_results"] = img_results

                img_title_temp = ""
                img_path_temp = []

                for img_result in img_results:
                    img_title_temp += img_result.get('title', 'No title') + "\n"
                    img_path_temp.append(img_result.get("img_path", ""))

                    img_path_final.append(img_result.get("img_path", ""))
                    img_title_final.append(img_result.get('title', 'No title'))

                img_title = img_title_temp
                img_path = img_path_temp

                for img_sub_query_data in img_sub_queries_data:
                    round_data.append(img_sub_query_data)
                    # results_manager.add_sub_queries(round_id, plan, img_results, img_sub_query_data)

            if len(round_data) > 0:
                results_manager.add_round(round_id=round_id,
                                          round_plan=plan,
                                          round_result=round_results,
                                          sub_queries_data=round_data)

        # Step 3.3: Let the MLLM generate the coarse filtered content according to the results of the search loop
        chat_client.set_system_prompt(prompt.loop_coarse_filtered_system)
        coarse_filtered_content = chat_client.send_message(text=prompt.loop_coarse_filtered)
        
        # coarse_filtered_content = coarse_filtered(input_prompt, txt_results_final, img_title_final, img_path_final)

        results_loop = {
            "img_title": img_title_final,
            "img_path": img_path_final,
            "coarse_filtered_content": coarse_filtered_content
        }

        results_manager.finalize_session(results_loop)

        chat_client.clear_conversation()

        return results_loop


def search_loop_img(warm_up_results, max_rounds, idx, input_prompt, search_results_path):

    results_manager = SearchResultsManager(search_results_path, session_id=idx, warm_up=False)

    # Step 1: Initialize Loop-search session
    data = results_manager.initialize_session(input_prompt)

    # Step 2: Check current progress (resume supported)
    progress = results_manager.get_current_progress()

    # Step 3.1: Check if the search loop is completed, if yes, return the results
    if progress["status"] == "completed":
        results = data[idx]["final_result"]
        return results

    else:
        # Step 3.2: If the search loop is not completed, initialize the search loop
        ensure_directory(f'{search_results_path}/pics/')    
        img_save_dir = f'{search_results_path}/pics/'

        chat_client = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

        background_knowledge = f"""
        The background knowledge for the image generation prompt {input_prompt} is: \n\n
        {warm_up_results}
        """

        img_title = ""
        img_path = []
        img_path_final = []
        img_title_final = []

        for round_id in range(1, max_rounds + 1):
            round_data = []
            round_results = {}

            # Step 3.2.1: If the round is the first round, let the MLLM explore the relevant information
            if round_id == 1:
                loop_input = prompt_img.loop_search_first.format(input_prompt, background_knowledge)
                chat_client.set_system_prompt(prompt_img.loop_search_frist_system)
            else:
                # Step 3.2.2: If the round is not the first round, let the MLLM judge if the sufficient information is found, if not, let the MLLM explore the relevant information
                loop_input = prompt_img.loop_search_else.format(input_prompt, img_title)
                chat_client.set_system_prompt(prompt_img.loop_search_else_system)

            # Step 3.2.3: Let the MLLM generate the search plan
            plan = chat_client.send_message(text=loop_input, image_paths=img_path)

            # Step 3.2.4: If the MLLM judges that the sufficient information is found, stop the search loop
            if "SUFFICIENT" in plan:
                print(f"Round {round_id}: Assessment is SUFFICIENT, stopping search loop")
                break

            # Step 3.2.5: Let the MLLM generate the search plan
            img_queries = []

            for line in plan.split('\n'):
                if 'Image Retrieval:' in line:
                    img_queries.append(line.split('Image Retrieval:')[1].strip())

            if len(img_queries) > 0:
                img_sub_queries_data, img_results = round_t2i_results(queries=img_queries,
                                                                      input_prompt=input_prompt,
                                                                      round_id=round_id,
                                                                      background_knowledge=background_knowledge,
                                                                      save_dir=img_save_dir)

                round_results["img_results"] = img_results

                img_title_temp = ""
                img_path_temp = []

                for img_result in img_results:
                    img_title_temp += img_result.get('title', 'No title') + "\n"
                    img_path_temp.append(img_result.get("img_path", ""))

                    img_path_final.append(img_result.get("img_path", ""))
                    img_title_final.append(img_result.get('title', 'No title'))

                img_title = img_title_temp
                img_path = img_path_temp

                for img_sub_query_data in img_sub_queries_data:
                    round_data.append(img_sub_query_data)
                    # results_manager.add_sub_queries(round_id, plan, img_results, img_sub_query_data)

            if len(round_data) > 0:
                results_manager.add_round(round_id=round_id,
                                          round_plan=plan,
                                          round_result=round_results,
                                          sub_queries_data=round_data)

        # coarse_filtered_content = coarse_filtered(input_prompt, txt_results_final, img_title_final, img_path_final)

        results_loop = {
            "img_title": img_title_final,
            "img_path": img_path_final,
            "coarse_filtered_content": background_knowledge
        }

        results_manager.finalize_session(results_loop)

        chat_client.clear_conversation()

        return results_loop


def search_loop_txt(warm_up_results, max_rounds, idx, input_prompt, search_results_path):
    results_manager = SearchResultsManager(search_results_path, session_id=idx, warm_up=False)

    # Step 1: Initialize Warm-up session
    data = results_manager.initialize_session(input_prompt)

    # Step 2: Check current progress (resume supported)
    progress = results_manager.get_current_progress()

    if progress["status"] == "completed":
        results = data[idx]["final_result"]
        return results

    else:
        chat_client = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

        background_knowledge = f"""
        The background knowledge for the image generation prompt {input_prompt} is: \n\n
        {warm_up_results}
        """

        for round_id in range(1, max_rounds + 1):
            round_data = []
            round_results = {}

            loop_input = prompt_txt.loop_search_else.format(input_prompt, background_knowledge)
            chat_client.set_system_prompt(prompt.loop_search_else_system)

            plan = chat_client.send_message(text=loop_input)

            if "SUFFICIENT" in plan:
                print(f"Round {round_id}: Assessment is SUFFICIENT, stopping search loop")
                break

            txt_queries = []

            for line in plan.split('\n'):
                if 'Text Retrieval:' in line:
                    txt_queries.append(line.split('Text Retrieval:')[1].strip())

            if len(txt_queries) > 0:
                txt_sub_queries_data, txt_results = round_t2t_results(txt_queries, input_prompt, round_id)
                txt_results = round_t2t_summary(plan, txt_results)
                background_knowledge = txt_results

                round_results["txt_results"] = txt_results

                for txt_sub_query_data in txt_sub_queries_data:
                    round_data.append(txt_sub_query_data)

            if len(round_data) > 0:
                results_manager.add_round(round_id=round_id,
                                          round_plan=plan,
                                          round_result=round_results,
                                          sub_queries_data=round_data)

        chat_client.set_system_prompt(prompt_txt.loop_coarse_filtered_system)
        coarse_filtered_content = chat_client.send_message(text=prompt_txt.loop_coarse_filtered)

        results_loop = {
            "coarse_filtered_content": coarse_filtered_content
        }

        results_manager.finalize_session(results_loop)

        chat_client.clear_conversation()

        return results_loop


def content_refine(idx, input_prompt, search_results_path, mm_content):
    json_path = f"{search_results_path}/refined_results.json"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data["status"] == "completed":
                output_prompt = {
                    'prompt': data['prompts'],
                    'img_path': data['filtered_images'],
                }
                return output_prompt
    
    # Filter images
    img_path = mm_content["img_path"]
    img_title = mm_content["img_title"]
    coarse_filtered_content = mm_content["coarse_filtered_content"]
    filtered_img_path = []
    filtered_img_title = []

    # filtering images
    chat_img_filtered = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

    img_content = ""
    for idx, title in enumerate(img_title):
        img_content += f"The title for <image_{idx}>: {title}\n"
    
    img_filtered_input = prompt.img_filtered.format(coarse_filtered_content, img_content)

    img_filtered_results = chat_img_filtered.send_single_message(text=img_filtered_input, image_paths=img_path,
                                                system_prompt=prompt.img_filtered_system)
    
    img_filtered_match = re.search(r"<selected_images>(.*?)</selected_images>", img_filtered_results, re.DOTALL)

    if img_filtered_match:
        img_filtered_idx = [int(x.strip()) for x in img_filtered_match.group(1).splitlines() if x.strip().isdigit()]
    else:
        img_filtered_idx = [0]

    for idx in img_filtered_idx:
        filtered_img_path.append(img_path[idx])
        filtered_img_title.append(img_title[idx])
    
    # refine multimodal content
    chat_mm_refined = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

    filtered_img_content = ""
    for idx, title in enumerate(filtered_img_title):
        filtered_img_content += f"The title for <image_{idx}>: {title}\n"

    mm_refined_input = prompt.mm_refined.format(coarse_filtered_content, filtered_img_content)

    mm_refined_results = chat_mm_refined.send_single_message(text=mm_refined_input, image_paths=filtered_img_path,
                                                system_prompt=prompt.mm_refined_system)

    chat_prompt_enhance = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

    prompt_enhance_input = prompt.prompt_enhance.format(input_prompt, mm_refined_results, filtered_img_content)

    prompt_enhance = chat_prompt_enhance.send_single_message(text=prompt_enhance_input, image_paths=filtered_img_path,
                                                   system_prompt=prompt.prompt_enhance_system)


    data = {
        "idx": idx,
        "status": "completed",
        "original_images": img_path,
        "filtered_reason": img_filtered_results,
        "mm_refined_results": mm_refined_results,
        "filtered_images": filtered_img_path,
        "prompts": prompt_enhance,
    }

    # Generate a JSON file to store the information above
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    output_prompt = {
        'prompt': prompt_enhance,
        'img_path': filtered_img_path,
    }
    
    return output_prompt


def content_refine_txt(idx, input_prompt, search_results_path, txt_content):
    json_path = f"{search_results_path}/refined_results.json"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data["status"] == "completed":
                output_prompt = {
                    'prompt': data['prompts'],
                }
                return output_prompt

    chat_prompt_enhance = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

    prompt_enhance_input = prompt_txt.prompt_enhance.format(input_prompt, txt_content['coarse_filtered_content'])

    prompt_enhance = chat_prompt_enhance.send_single_message(text=prompt_enhance_input,
                                                             system_prompt=prompt_txt.prompt_enhance_system)

    data = {
        "idx": idx,
        "status": "completed",
        "prompts": prompt_enhance,
    }

    # Generate a JSON file to store the information above
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    output_prompt = {
        'prompt': prompt_enhance,
    }

    return output_prompt


def content_refine_img(idx, input_prompt, search_results_path, img_content):
    json_path = f"{search_results_path}/refined_results.json"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data["status"] == "completed":
                output_prompt = {
                    'prompt': data['prompts'],
                    'img_path': data['filtered_images'],
                }
                return output_prompt

    # Filter images
    img_path = img_content["img_path"]
    img_title = img_content["img_title"]
    coarse_filtered_content = img_content["coarse_filtered_content"]
    filtered_img_path = []
    filtered_img_title = []

    # filtering images
    chat_img_filtered = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

    img_content = ""
    for idx, title in enumerate(img_title):
        img_content += f"The title for <image_{idx}>: {title}\n"

    img_filtered_input = prompt_img.img_filtered.format(img_content)

    img_filtered_results = chat_img_filtered.send_single_message(text=img_filtered_input, image_paths=img_path,
                                                                 system_prompt=prompt_img.img_filtered_system)

    img_filtered_match = re.search(r"<selected_images>(.*?)</selected_images>", img_filtered_results, re.DOTALL)

    if img_filtered_match:
        img_filtered_idx = [int(x.strip()) for x in img_filtered_match.group(1).splitlines() if x.strip().isdigit()]
    else:
        img_filtered_idx = [0]

    for idx in img_filtered_idx:
        filtered_img_path.append(img_path[idx])
        filtered_img_title.append(img_title[idx])

    filtered_img_content = ""
    for idx, title in enumerate(filtered_img_title):
        filtered_img_content += f"The title for <image_{idx}>: {title}\n"

    chat_prompt_enhance = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

    prompt_enhance_input = prompt_img.prompt_enhance.format(input_prompt, filtered_img_content)

    prompt_enhance = chat_prompt_enhance.send_single_message(text=prompt_enhance_input, image_paths=filtered_img_path,
                                                             system_prompt=prompt_img.prompt_enhance_system)

    data = {
        "idx": idx,
        "status": "completed",
        "original_images": img_path,
        "filtered_reason": img_filtered_results,
        "filtered_images": filtered_img_path,
        "prompts": prompt_enhance,
    }

    # Generate a JSON file to store the information above
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    output_prompt = {
        'prompt': prompt_enhance,
        'img_path': filtered_img_path,
    }

    return output_prompt


def cot(input_prompt):

    cot_client = MultimodalRetrievalClient(QWEN_API_KEY, model="qwen2.5-vl-72b-instruct")

    cot_prompt = cot_client.send_single_message(text=prompt.cot_prompt.format(input_prompt),
                                                system_prompt=prompt.prompt_system)

    cot_client.clear_conversation()

    output_prompt = {
        "cot": cot_prompt
    }

    return output_prompt


def img_generation(idx, prompt, gen_path, gen_model, modality):
    # Options: 'gemini_gen', 'openai_gen', 'qwen_gen', 'flux_context'

    if gen_model == "openai_gen":
        gen_gpt(idx, prompt, gen_path, modality)
    elif gen_model == "qwen_gen":
        gen_qwen(idx, prompt, gen_path, modality)
    elif gen_model == "flux_context":
        gen_flux(idx, prompt, gen_path, modality)
    elif gen_model == "gemini_gen":
        gen_gemini(idx, prompt, gen_path, modality)
    else:
        raise ValueError(f"Invalid generation model: {gen_model}")

# def enhanced_prompt(input_prompt, mm_refined_results, filtered_img_path, filtered_img_title, coarse_filtered_content, img_path, img_title):
#
#     mm_chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5")
#
#     img_content = ""
#     for idx, title in enumerate(filtered_img_title):
#         img_content += f"The title for <image_{idx}>: {title}\n"
#
#     mm_prompt_input = prompt.mm_prompt.format(input_prompt, mm_refined_results, img_content)
#
#     mm_prompt = mm_chat_client.send_single_message(text=mm_prompt_input, image_paths=filtered_img_path,
#                                                 system_prompt=prompt.mm_prompt_system)
#
#     mm_chat_client.clear_conversation()
#
#     txt_chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5")
#
#     txt_prompt_input = prompt.txt_prompt.format(input_prompt, coarse_filtered_content)
#
#     txt_prompt = txt_chat_client.send_single_message(text=txt_prompt_input,
#                                                 system_prompt=prompt.prompt_system)
#
#     txt_chat_client.clear_conversation()
#
#     img_chat_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5")
#
#     img_content = ""
#     for idx, title in enumerate(img_title):
#         img_content += f"The title for <image_{idx}>: {title}\n"
#
#     img_prompt_input = prompt.img_prompt.format(input_prompt, img_content)
#
#     img_prompt = img_chat_client.send_single_message(text=img_prompt_input, image_paths=img_path,
#                                                 system_prompt=prompt.prompt_system)
#     img_chat_client.clear_conversation()
#
#
#     cot_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5")
#
#     cot_prompt = cot_client.send_single_message(text=prompt.cot_prompt.format(input_prompt),
#                                                 system_prompt=prompt.prompt_system)
#
#     cot_client.clear_conversation()
#
#     prompts = {
#         "mm_prompt": mm_prompt,
#         "txt_prompt": txt_prompt,
#         "img_prompt": img_prompt,
#         "cot_prompt": cot_prompt
#     }
#
#     return prompts

