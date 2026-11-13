import json

transcript_path = r"C:\Users\jithe\.gemini\antigravity\brain\9e3c917b-950c-4504-9ade-73420a273417\.system_generated\logs\transcript_full.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        if '"step_index":925' in line:
            data = json.loads(line)
            with open("scratch/user_prompt_925.txt", "w", encoding="utf-8") as out:
                out.write(data.get("content", ""))
            print("Extracted user prompt length:", len(data.get("content", "")))
            break

