import itertools
import json
import os

import anyio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))

STYLE_SLOTS = [
    ("casual", "casual/conversational, like talking to a friend"),
    ("polite_request", "a polite request with 'please' or 'could you'"),
    ("question", "a question form, e.g. 'Can you...' or 'Would you...'"),
    ("terse", "very terse and minimal, like a label or quick note"),
    ("verbose", "verbose and descriptive, spelling out every detail"),
    ("passive_indirect", "passive or indirect phrasing, e.g. 'The X needs to be placed...'"),
    ("goal_framing", "framed as a goal/outcome, e.g. 'Make sure X is in Y'"),
    ("sequential", "broken into explicit steps, e.g. 'First grab X, then put it in Y'"),
]


def load_instructions(input_file):
    instructions = []
    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            instructions.append(item["task"])
    return instructions


async def call_llm(instruction, num_samples=5):
    styles = list(itertools.islice(itertools.cycle(STYLE_SLOTS), num_samples))
    style_lines = "\n".join(
        f"  {i+1}. Style: {name} — {desc}" for i, (name, desc) in enumerate(styles)
    )
    prompt = f"""Rewrite the following robot instruction in {num_samples} DISTINCTLY DIFFERENT ways.
Each rewrite must use a DIFFERENT style as specified below — do NOT just swap synonyms.

Styles to use (one per rewrite, in order):
{style_lines}

Rules:
- Preserve the EXACT same meaning and objects as the original
- Each rewrite must sound noticeably different from the others

Return ONLY a JSON object:
{{"instructions": ["rewrite1", "rewrite2", ...]}}"""

    instructions = []
    while len(instructions) < num_samples:
        response = await client.chat.completions.parse(
            messages=[
                {
                    "role": "system",
                    "content": "You are generating diverse instruction rewrites for a robot benchmark. Each rewrite must use a completely different style and sentence structure — not just synonym substitution. Preserve meaning exactly. Return ONLY valid JSON.",
                },
                {"role": "user", "content": f'{prompt}\n\nOriginal instruction: "{instruction}"'},
            ],
            model="deepseek-v4-flash",
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        try:
            ins = json.loads(response.choices[0].message.content)["instructions"]
            instructions.extend(ins)
        except (json.JSONDecodeError, KeyError):
            print(f"Failed to parse response for instruction: {instruction}")
            continue
    return instructions[:num_samples]


async def process_instructions(instructions):
    processed_instructions = {}

    async def process_single(instruction):
        result = await call_llm(instruction)
        processed_instructions[instruction] = result

    async with anyio.create_task_group() as tg:
        for instruction in instructions:
            tg.start_soon(process_single, instruction)

    return processed_instructions


if __name__ == "__main__":
    task_files = [
        "playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta/tasks.jsonl",
        "playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta/tasks.jsonl",
        "playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta/tasks.jsonl",
        "playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta/tasks.jsonl",
    ]
    all_instructions = []
    for file in task_files:
        instructions = load_instructions(file)
        all_instructions.extend(instructions)
    processed_instructions = anyio.run(process_instructions, all_instructions)
    with open("processed_instructions.json", "w") as f:
        json.dump(processed_instructions, f, indent=4)
