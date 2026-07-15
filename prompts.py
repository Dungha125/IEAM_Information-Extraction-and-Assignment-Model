"""Shared prompts & helpers for CV Field Mapper fine-tuning."""

from __future__ import annotations

import json
from typing import Any, Dict

SYSTEM_PROMPT = """You fix CV JSON field placement (Vietnamese/English).
Output ONLY one JSON object. No markdown. Do not invent facts.
Rules:
- company = company/org name; position = job title. Swap if reversed.
- education = school, major, GPA. Never put school/GPA in awards.
- awards = prizes/honors only.
- languages = spoken languages only (English, Japanese). Never Python/Java/React.
- skills.categories.Language = programming languages.
- Split multiple jobs/projects into separate array items.
"""


def format_user_prompt(draft: Dict[str, Any], sections: Dict[str, str]) -> str:
    section_parts = []
    for key, text in sections.items():
        if text and str(text).strip():
            section_parts.append(f"[{key}]\n{str(text).strip()}")
    sections_blob = "\n\n".join(section_parts)
    return (
        "Fix this CV draft. Return JSON with keys: "
        "personal_info, education, experience, projects, skills, languages, "
        "certificates, awards, activities.\n\n"
        f"DRAFT:\n{json.dumps(draft, ensure_ascii=False)}\n\n"
        f"SECTIONS:\n{sections_blob}\n"
    )


def empty_profile() -> Dict[str, Any]:
    return {
        "personal_info": {
            "name": None,
            "email": None,
            "phone": None,
            "location": None,
            "job_title": None,
            "github": None,
            "linkedin": None,
            "summary": None,
        },
        "education": [],
        "experience": [],
        "projects": [],
        "skills": {"explicit": [], "categories": {}, "soft_skills": []},
        "languages": [],
        "certificates": [],
        "awards": [],
        "activities": [],
    }
