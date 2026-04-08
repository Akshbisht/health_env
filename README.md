---
title: Parent-Child Health Decision Maker
emoji: 🧒
colorFrom: red
colorTo: yellow
sdk: docker
sdk_version: "0.15.1"
python_version: "3.10.12"
app_file: server.py
pinned: false
---
# Health-Based Parent-Child Decision Maker


An OpenEnv-compatible reinforcement learning environment where an AI agent assists a worried parent in triaging their sick child to the correct care pathway.


---

## Motivation


Parents face a high-stakes, high-anxiety decision daily: **"Should I take my child to the ER, urgent care, or just keep them home?"** Getting this wrong in either direction has real consequences — over-triaging burdens emergency services; under-triaging risks lives. This environment trains agents to ask the right clarifying questions and apply clinical red-flag reasoning.


---

## Tasks


| Task | Difficulty | Correct Pathway | Key Challenge |
|------|-----------|----------------|---------------|
| `easy_fever` | Easy | `home_care` | Avoid over-triaging a simple cold |
| `medium_asthma` | Medium | `urgent_care` | Recognise borderline SpO2 + known asthma |
| `hard_meningitis_risk` | Hard | `er_immediately` | Catch non-blanching rash + neck stiffness fast |


---

## Observation Space


```json
{
  "child_profile":          { "age_years": 4, "weight_kg": 16, "conditions": [] },
  "presenting_symptoms":    ["fever 38.2°C", "mild runny nose"],
  "vitals":                 { "temperature_c": 38.2, "heart_rate_bpm": 98, "spo2_pct": 99, "respiratory_rate": 22 },
  "conversation_history":   [ { "role": "agent", "content": "..." } ],
  "current_step":           0,
  "task_name":              "easy_fever",
  "instructions":           "..."
}
```


## Action Space


```json
{ "message": "string — agent response/question/recommendation" }
```


To trigger episode end, include `RECOMMENDATION: <pathway>` anywhere in the message.


---

## Reward Function


| Component | Max Value | Condition |
|-----------|-----------|-----------|
| Clarifying questions | +0.3 | Asking task-relevant keywords |
| Correct recommendation | +0.7 | Exact pathway match |
| Red-flag acknowledgment | +0.2 | Hard task only |
| Dangerous recommendation | -0.5 | Home care for ER case |


Total score is clamped to `[0.0, 1.0]`.


---

## API Endpoints


| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start/restart an episode |
| `POST` | `/step` | Send agent action, get observation + reward |
| `GET`  | `/state` | Inspect current environment state |
| `GET`  | `/tasks` | List all available tasks |


**Reset request:**
```json
{ "task": "easy_fever" }
```


**Step request:**
```json
{ "task": "easy_fever", "message": "Does the child have any rashes?" }
```


---

## Setup & Usage


### Local


```bash
pip install -r requirements.txt
uvicorn server:app --port 7860


# Test reset
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"easy_fever"}'
```


### Docker


```bash
docker build -t health-env .
docker run -p 7860:7860 health-env
```


### Run Inference


```bash
export HF_TOKEN=your_key
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860
python inference.py
```


---

## Baseline Scores (Qwen2.5-72B-Instruct)


| Task | Score |
|------|-------|
| easy_fever | ~0.80 |
| medium_asthma | ~0.65 |
| hard_meningitis_risk | ~0.55 |


---

## HuggingFace Space


Tagged with `openenv`. Space URL: `https://akaashdfjndf-health-env.hf.space`
