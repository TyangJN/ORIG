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
5. Maximum 3 additional searches per iteration to prevent search loops

---

MANDATORY OUTPUT FORMAT:

<Sufficiency Evaluation>
Assess the current knowledge for image generation readiness across these dimensions:
- All key entities have clear definitions or descriptions
- Visual characteristics, style, and context are sufficiently detailed
- No major ambiguity remains that would harm generation quality

Overall Assessment: SUFFICIENT or INSUFFICIENT  (MUST be one of these two, in all caps)

If INSUFFICIENT, explain what specific visual gaps would significantly impact image quality.
</Sufficiency Evaluation>

<Sub-Questions>
[Only include this entire section if Overall Assessment = INSUFFICIENT]
Write 1–3 atomic questions for missing elements, in priority order:
- Text questions: "What is X?", "What are main characteristics of X?", "What is the process or stages of X?"...

Each question must directly relate to entities, scene layout, style, or critical visual details from the input prompt.
Number of questions MUST match the number of queries in <Search Plan>, and they must be in the same order.
</Sub-Questions>


<Search Plan>
[Only include if Overall Assessment = INSUFFICIENT]
- Generate exactly one matching search query per sub-question, in the same order.
- You should use only TEXT retrievals.
- Format like:
<Text Retrievals>
Text Retrieval: natural-language-query-1
</Text Retrievals>

Use only TEXT retrievals as appropriate.
</Search Plan>

---

STRICT RULES:
- Plan new searches only for high-impact missing elements
- Avoid redundant or overly broad queries
- Stop when added searches have diminishing value


Input Prompt: {}
Reference Knowledge: {}
"""


loop_coarse_filtered_system = """
You are a prompt refinement assistant for text-to-image generation tasks.
Your job is to filter and consolidate all retrieved information from previous steps,
remove duplication, and produce a clean, concise knowledge summary that directly supports the input prompt.

CRITICAL INSTRUCTIONS:
1. Include ONLY information relevant to the input prompt and image generation.
2. Remove repeated facts, verbose explanations, or navigation text.
3. Integrate text-derived insights into a single coherent description.
4. Keep the result factual, compact, and directly usable for prompt enhancement.
5. Do NOT add new information or speculate beyond what was retrieved.

"""


loop_coarse_filtered = """
MANDATORY OUTPUT FORMAT:
Return ONLY the refined knowledge, in plain text. 
Do not include explanations, headers, or formatting other than the final content itself.
"""


loop_coarse_filtered_advanced = """
<InputPrompt>
{}
</InputPrompt>

<RetrievedText>
{}
</RetrievedText>

MANDATORY OUTPUT FORMAT:

<MergedFilteredKnowledge>
[Write the consolidated, deduplicated knowledge here. Only include content that is relevant to the input prompt and improves visual completeness.]
</MergedFilteredKnowledge>
"""


prompt_enhance_system = """
You are a prompt synthesis assistant for text-to-image generation tasks.
Your task is to synthesize a prompt from the given references.
"""


prompt_enhance = """
Input Prompt:
{}

TEXTUAL REFERENCE
{}

Generate a highly controllable T2I prompt that transforms all the structured control parameters into precise visual generation instructions.
"""


