"""
inference.py — Health-Based Parent-Child Decision Maker
=======================================================
Runs an LLM agent against all 3 tasks and emits [START]/[STEP]/[END] logs.

Environment variables:
  API_BASE_URL   LLM endpoint  (default: HuggingFace router)
  MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
"""
from __future__ import annotations

import asyncio
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["easy_fever", "medium_asthma", "hard_meningitis_risk"]
BENCHMARK = "health_env"
MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are a knowledgeable health triage assistant helping a worried parent assess their sick child.
    
    Your job:
    1. Review the child's symptoms and vitals provided.
    2. Ask 1-2 focused clarifying questions per turn to gather missing critical information.
    3. When you have enough information, end your message with exactly:
       RECOMMENDATION: <pathway>
       where <pathway> is one of: home_care | urgent_care | er_immediately
    
    Guidelines:
    - home_care: mild symptoms, no red flags, stable vitals
    - urgent_care: moderate symptoms, known conditions affected, borderline vitals
    - er_immediately: red flag signs (non-blanching rash, stiff neck, severe respiratory distress, altered consciousness)
    
    Be concise, empathetic, and clinically accurate. Ask about red flags for the age group.
    Never use more than 3 sentences per reply before you have enough info to recommend.
""").strip()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ")[:200]
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM Call ──────────────────────────────────────────────────────────────────

def get_agent_message(
    client: OpenAI,
    observation: Dict[str, Any],
    history: List[str],
) -> str:
    profile = observation.get("child_profile", {})
    symptoms = observation.get("presenting_symptoms", [])
    vitals = observation.get("vitals", {})
    conv = observation.get("conversation_history", [])
    instructions = observation.get("instructions", "")

    history_block = "\n".join(history[-4:]) if history else "None"
    conv_block = "\n".join(f"  {t['role']}: {t['content']}" for t in conv[-6:]) if conv else "None"

    user_prompt = textwrap.dedent(f"""
        CHILD PROFILE: {profile}
        SYMPTOMS: {', '.join(symptoms)}
        VITALS: {vitals}
        
        CONVERSATION SO FAR:
        {conv_block}
        
        PREVIOUS STEPS:
        {history_block}
        
        INSTRUCTIONS: {instructions}
        
        Your response:
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "Can you tell me more about the symptoms?"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "Can you tell me more about the symptoms?"


# ── Episode Runner ────────────────────────────────────────────────────────────

async def run_episode(client: OpenAI, task: str) -> float:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=60.0) as http:
        try:
            # Reset
            reset_resp = await http.post("/reset", json={"task": task})
            reset_resp.raise_for_status()
            result = reset_resp.json()

            for step in range(1, MAX_STEPS + 1):
                if result.get("done"):
                    break

                obs = result["observation"]
                message = get_agent_message(client, obs, history)

                step_resp = await http.post("/step", json={"task": task, "message": message})
                step_resp.raise_for_status()
                result = step_resp.json()

                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                error = result.get("info", {}).get("reason") if reward == 0.0 else None

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=message, reward=reward, done=done, error=error)
                history.append(f"Step {step}: {message[:80]!r} -> reward {reward:+.2f}")

                if done:
                    break

            score = min(max(sum(rewards), 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: Dict[str, float] = {}
    for task in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"[DEBUG] Running task: {task}", flush=True)
        score = await run_episode(client, task)
        all_scores[task] = score

    print(f"\n{'='*50}", flush=True)
    print("[DEBUG] === FINAL SCORES ===", flush=True)
    for task, score in all_scores.items():
        print(f"[DEBUG]   {task}: {score:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[DEBUG]   AVERAGE: {avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
