"""
Health-Based Parent-Child Decision Maker — OpenEnv Environment
==============================================================
Simulates a pediatric health triage scenario where an AI agent must
assess a child's symptoms, ask clarifying questions, and recommend
the correct care pathway (home care / urgent care / ER).
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Pydantic Models ───────────────────────────────────────────────────────────

class Observation(BaseModel):
    child_profile: Dict[str, Any] = Field(description="Age, weight, known conditions")
    presenting_symptoms: List[str] = Field(description="Current symptoms reported by parent")
    vitals: Dict[str, Any] = Field(description="Temperature, heart rate, SpO2, etc.")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    current_step: int = Field(default=0)
    task_name: str = Field(default="")
    instructions: str = Field(default="")


class Action(BaseModel):
    message: str = Field(description="Agent's message/question/recommendation to the parent")


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""


# ── Scenario Bank ─────────────────────────────────────────────────────────────

SCENARIOS = {
    "easy_fever": {
        "child_profile": {"age_years": 4, "weight_kg": 16, "conditions": []},
        "presenting_symptoms": ["fever 38.2°C", "mild runny nose", "normal appetite"],
        "vitals": {"temperature_c": 38.2, "heart_rate_bpm": 98, "spo2_pct": 99, "respiratory_rate": 22},
        "correct_pathway": "home_care",
        "red_flags": [],
        "task": "easy",
        "description": "4-year-old with low-grade fever and mild cold symptoms. Clearly home-care appropriate.",
    },
    "medium_asthma": {
        "child_profile": {"age_years": 7, "weight_kg": 22, "conditions": ["mild asthma"]},
        "presenting_symptoms": ["wheezing", "mild shortness of breath", "cough", "fever 37.8°C"],
        "vitals": {"temperature_c": 37.8, "heart_rate_bpm": 110, "spo2_pct": 95, "respiratory_rate": 28},
        "correct_pathway": "urgent_care",
        "red_flags": ["SpO2 95% borderline", "known asthma"],
        "task": "medium",
        "description": "7-year-old asthmatic with mild respiratory distress. Needs urgent care, not ER.",
    },
    "hard_meningitis_risk": {
        "child_profile": {"age_years": 2, "weight_kg": 12, "conditions": []},
        "presenting_symptoms": [
            "high fever 39.8°C",
            "neck stiffness",
            "photophobia",
            "irritability",
            "non-blanching rash",
        ],
        "vitals": {"temperature_c": 39.8, "heart_rate_bpm": 148, "spo2_pct": 97, "respiratory_rate": 32},
        "correct_pathway": "er_immediately",
        "red_flags": ["non-blanching rash", "neck stiffness", "photophobia", "tachycardia"],
        "task": "hard",
        "description": "2-year-old with classic meningitis warning signs. Must be sent to ER immediately.",
    },
}

PATHWAY_KEYWORDS = {
    "home_care": ["home", "rest", "fluids", "monitor", "paracetamol", "ibuprofen", "acetaminophen", "tylenol"],
    "urgent_care": ["urgent care", "clinic", "doctor today", "same-day", "pediatrician", "inhaler", "nebulizer"],
    "er_immediately": ["emergency", "er", "911", "ambulance", "hospital now", "immediately", "call 999"],
}

CLARIFYING_QUESTIONS_SCORE = {
    "easy": ["duration", "temperature", "appetite"],
    "medium": ["inhaler", "breathing", "spo2", "oxygen", "asthma", "wheezing"],
    "hard": ["rash", "stiff", "light", "consciousness", "blanch", "fontanelle"],
}


# ── Environment ───────────────────────────────────────────────────────────────

class HealthEnv:
    """
    OpenEnv-compatible environment for pediatric health triage.
    Three tasks: easy (home care), medium (urgent care), hard (ER).
    """

    MAX_STEPS = 8

    def __init__(self, task: str = "easy_fever"):
        self._task_key = task
        self._scenario: Dict[str, Any] = {}
        self._history: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._done: bool = False
        self._final_recommendation: Optional[str] = None
        self._questions_asked: List[str] = []

    # ── OpenEnv API ────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        scenario = SCENARIOS[self._task_key]
        self._scenario = scenario
        self._history = []
        self._step_count = 0
        self._done = False
        self._final_recommendation = None
        self._questions_asked = []

        obs = Observation(
            child_profile=scenario["child_profile"],
            presenting_symptoms=scenario["presenting_symptoms"],
            vitals=scenario["vitals"],
            conversation_history=[],
            current_step=0,
            task_name=self._task_key,
            instructions=(
                "You are a health triage assistant helping a worried parent. "
                "Ask relevant clarifying questions, then recommend ONE of: "
                "'home_care', 'urgent_care', or 'er_immediately'. "
                "End your final message with: RECOMMENDATION: <pathway>"
            ),
        )
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {"scenario": scenario["description"]},
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        message = action.get("message", "")
        self._step_count += 1
        self._history.append({"role": "agent", "content": message})

        reward, breakdown, reason = self._compute_reward(message)
        done = self._is_done(message)
        self._done = done

        obs = Observation(
            child_profile=self._scenario["child_profile"],
            presenting_symptoms=self._scenario["presenting_symptoms"],
            vitals=self._scenario["vitals"],
            conversation_history=self._history.copy(),
            current_step=self._step_count,
            task_name=self._task_key,
            instructions="Continue triage. End with: RECOMMENDATION: <pathway>",
        )

        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": {"breakdown": breakdown, "reason": reason},
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task": self._task_key,
            "step": self._step_count,
            "done": self._done,
            "scenario": self._scenario,
            "history": self._history,
            "final_recommendation": self._final_recommendation,
        }

    # ── Reward Logic ───────────────────────────────────────────────────────

    def _compute_reward(self, message: str) -> tuple[float, Dict[str, float], str]:
        msg_lower = message.lower()
        breakdown: Dict[str, float] = {}
        task_level = self._scenario.get("task", "easy")

        # 1. Clarifying questions bonus (partial credit, up to 0.3)
        relevant_keywords = CLARIFYING_QUESTIONS_SCORE.get(task_level, [])
        new_questions = [kw for kw in relevant_keywords if kw in msg_lower and kw not in self._questions_asked]
        if new_questions:
            self._questions_asked.extend(new_questions)
            q_score = min(len(new_questions) * 0.1, 0.3)
            breakdown["clarifying_questions"] = q_score
        else:
            breakdown["clarifying_questions"] = 0.0

        # 2. Final recommendation check
        rec_score = 0.0
        reason = "No recommendation yet"
        if "recommendation:" in msg_lower:
            correct = self._scenario["correct_pathway"]
            if correct in msg_lower:
                rec_score = 0.7
                reason = f"Correct pathway '{correct}' recommended"
                self._final_recommendation = correct
            else:
                # Partial: wrong pathway but at least gave one
                rec_score = 0.1
                reason = "Wrong pathway recommended"
                for pathway in PATHWAY_KEYWORDS:
                    if any(kw in msg_lower for kw in PATHWAY_KEYWORDS[pathway]):
                        self._final_recommendation = pathway
                        break

        breakdown["correct_recommendation"] = rec_score

        # 3. Red flag acknowledgment for hard task
        red_flag_score = 0.0
        if task_level == "hard":
            flags_mentioned = sum(
                1 for flag in self._scenario.get("red_flags", [])
                if any(word in msg_lower for word in flag.lower().split())
            )
            red_flag_score = min(flags_mentioned * 0.05, 0.2)
            breakdown["red_flags_acknowledged"] = red_flag_score
        else:
            breakdown["red_flags_acknowledged"] = 0.0

        # 4. Penalty: dangerous recommendation (e.g. home care for ER case)
        penalty = 0.0
        if "recommendation:" in msg_lower:
            correct = self._scenario["correct_pathway"]
            if correct == "er_immediately" and any(kw in msg_lower for kw in PATHWAY_KEYWORDS["home_care"]):
                penalty = -0.5
                breakdown["dangerous_recommendation_penalty"] = penalty
                reason = "DANGEROUS: Recommended home care for ER-level case"

        breakdown["penalty"] = penalty
        total = sum(breakdown.values())
        total = round(min(max(total, 0.0), 1.0), 4)
        return total, breakdown, reason

    def _is_done(self, message: str) -> bool:
        if self._step_count >= self.MAX_STEPS:
            return True
        if "recommendation:" in message.lower():
            return True
        return False


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_task(task_key: str, conversation: List[Dict[str, str]]) -> float:
    """
    Standalone grader: replays a conversation and returns a score in [0, 1].
    """
    env = HealthEnv(task=task_key)
    env.reset()
    total_reward = 0.0
    max_reward = 1.0  # We normalise against 1.0

    for turn in conversation:
        if turn["role"] == "agent":
            result = env.step({"message": turn["content"]})
            total_reward += result["reward"]
            if result["done"]:
                break

    return round(min(max(total_reward, 0.0), 1.0), 4)
