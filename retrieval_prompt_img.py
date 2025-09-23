loop_search_frist_system = """
You are a professional multimodal web search expert, specialized in providing high-quality supporting data for image generation systems.
Your job is to help clarify the user's input prompt by searching the web for relevant information.
You MUST follow the required format exactly, with no extra commentary.
"""


loop_search_first = """
CRITICAL INSTRUCTIONS:
1. Work strictly on entities, scenes, and factual information mentioned or strongly implied in the input prompt.
2. Generate retrievals ONLY for missing or unclear elements (avoid redundancy with existing knowledge).
3. Choose retrieval type deliberately:
   - Use IMAGE retrieval for direct visual references, style examples, material textures, or spatial layout demonstrations
4. Keep total queries to 1–2 (prefer fewer, high-value searches).

---

MANDATORY OUTPUT FORMAT:
<Thought>
- Missing elements needing IMAGE retrieval (appearance, style, layout)
Briefly justify why each missing element is assigned to IMAGE retrieval.
</Thought>

<Sub-Questions>
Write 1–2 atomic questions for missing elements, in priority order:
- Image questions: Short, direct queries (3–10 words, focus on appearance/style/layout)

Each question must relate directly to prompt entities or implied style/scene.
</Sub-Questions>

<Search Plan>
<Image Retrievals>
Image Retrieval: visual-search-query-1
</Image Retrievals>

Use only IMAGE retrievals as appropriate. Leave a section empty only if no queries are truly needed for that type.
</Search Plan>

---

STRICT RULES:

For Image Retrieval:
- Target concrete, observable attributes: appearance, material, style, composition
- Include detail level or perspective if relevant 
- Keep short, specific, 3–10 words
- Avoid abstract or metaphorical language

General:
- Do NOT generate unrelated or overly broad queries
- Explicitly match sub-questions to search queries (1-to-1, in order)
- Cover the most critical missing visual/semantic information first
- Max 1–2 total queries 
---

Input Prompt: {}
Existing Knowledge: {}
"""


loop_search_else_system = """
You are an iterative search planning assistant for text-to-image prompt enhancement.
Evaluate if current knowledge is sufficient for high-quality image generation. If not, propose a small, focused new search plan.
"""


loop_search_else = """
CRITICAL INSTRUCTIONS:
1. Evaluate knowledge sufficiency based on visual completeness, not general information completeness
2. Only plan additional searches if they would significantly improve image generation quality
3. Focus on the most critical missing visual elements that affect the overall image outcome
4. Avoid redundant or marginally useful searches
5. Maximum 2 additional searches per iteration to prevent search loops

---

MANDATORY OUTPUT FORMAT:

<Sufficiency Evaluation>
Assess the current knowledge for image generation readiness across these dimensions:
- Visual characteristics and appearance

Overall Assessment: SUFFICIENT or INSUFFICIENT  (MUST be one of these two, in all caps)

If INSUFFICIENT, explain what specific visual gaps would significantly impact image quality.
</Sufficiency Evaluation>

<Sub-Questions>
[Only include this entire section if Overall Assessment = INSUFFICIENT]
Write 1–2 atomic questions for missing elements, in priority order:
- Image questions: Short, direct queries (3–10 words, focus on appearance/style/layout)

Each question must directly relate to entities, scene layout, style, or critical visual details from the input prompt.
Number of questions MUST match the number of queries in <Search Plan>, and they must be in the same order.
</Sub-Questions>


<Search Plan>
[Only include if Overall Assessment = INSUFFICIENT]
- Generate exactly one matching search query per sub-question, in the same order.
- You may use only IMAGE retrievals as appropriate.
- Format like:
<Image Retrievals>
Image Retrieval: visual-search-query-1
</Image Retrievals>

Use only IMAGE retrievals as appropriate. Leave a section empty only if no queries are truly needed for that type.
</Search Plan>

---

STRICT RULES:
- Plan new searches only for high-impact missing elements
- Avoid redundant or overly broad queries
- Stop when added searches have diminishing value


Input Prompt: {}
Reference Image: {}
"""

img_filtered_system = """
You are an intelligent image filtering assistant for text-to-image generation enhancement.

Your task is to analyze a set of retrieved images and their metadata, then select the most relevant images that align with the given textual content while providing complementary visual information.

SELECTION CRITERIA:
1. Content Consistency: Images must be semantically consistent with the provided text content
2. Visual Relevance: Images should depict entities, scenes, or concepts mentioned or implied in the text
3. Complementary Information: Prefer images that provide additional factual visual details not fully captured in text
4. Quality Standards: Select clear, high-resolution, representative images
5. Diversity: If multiple relevant images exist, choose those showing different aspects/perspectives

FILTERING RULES:
- Select AT LEAST 1 relevant image 
- Maximum 5 images to avoid information overload
- Avoid duplicates, low-quality images, or purely decorative content
- Prioritize images that would enhance visual understanding of the text content

MANDATORY OUTPUT FORMAT:

<selected_images>
[List the indices of selected images, one per line]
0
3
7
</selected_images>

<reasoning>
For each selected image, write 1 sentence (≤20 words) explaining why it was chosen and how it complements the text.
List explanations in the same order as the indices.
</reasoning>

Do NOT include any extra text outside these two sections.
"""


img_filtered = """
<AvailableImages>
{}
</AvailableImages>

Please select the most relevant image(s), and use the output format strictly.
"""

prompt_enhance_system = """
You are a prompt synthesis assistant for text-to-image generation tasks.
Your task is to synthesize a prompt from the given references.
"""


prompt_enhance = """
Input Prompt:
{}

IMAGE REFERENCE
{}

Generate a highly controllable T2I prompt that transforms all the structured control parameters into precise visual generation instructions.
"""