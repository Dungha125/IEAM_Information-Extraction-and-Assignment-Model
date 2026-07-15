#!/usr/bin/env python3
"""Generate synthetic CV field-mapping SFT pairs (wrong draft → correct JSON)."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prompts import SYSTEM_PROMPT, empty_profile, format_user_prompt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

VN_NAMES = [
    "Hà Mạnh Dũng", "Nguyễn Văn An", "Trần Thị Bình", "Lê Minh Châu", "Phạm Quốc Đạt",
    "Hoàng Thu Hà", "Võ Thanh Phong", "Đặng Mỹ Linh", "Bùi Đức Huy", "Ngô Bảo Trâm",
]
EN_NAMES = [
    "Alex Nguyen", "Sarah Tran", "James Le", "Mia Pham", "Chris Vo",
    "Linda Hoang", "David Bui", "Emma Ngo", "Ryan Dang", "Sophie Vu",
]
COMPANIES = [
    "FPT Software", "Smart Solution VietNam", "Lumi R&D VietNam", "VNG Corporation",
    "Tiki Trading", "Shopee Vietnam", "MoMo", "NashTech", "KMS Technology",
    "Công ty ABC", "CMC Global", "Axon Active", "Be Group", "Zalo Group",
]
TITLES = [
    "Backend Developer", "Frontend Developer", "Fullstack Developer",
    "Thực tập sinh Fullstack", "Thực tập sinh AI Engineer", "Java Developer",
    "Mobile Developer", "DevOps Engineer", "Data Engineer", "QA Engineer",
]
SCHOOLS = [
    "Học viện Công nghệ Bưu chính Viễn thông",
    "Đại học Bách khoa Hà Nội",
    "Đại học Công nghệ - ĐHQGHN",
    "PTIT",
    "UIT",
    "University of Science",
]
MAJORS = ["CNTT", "Khoa học máy tính", "Công nghệ thông tin", "Software Engineering", "AI"]
PROG_SKILLS = ["Python", "Java", "JavaScript", "TypeScript", "C++", "Go", "PHP"]
BACKEND = ["FastAPI", "NestJS", "Spring Boot", "Django", "NodeJS", "Express"]
FRONTEND = ["React", "Angular", "Vue", "Next.js"]
DBS = ["MySQL", "PostgreSQL", "MongoDB", "Redis"]
SPOKEN = [
    ("English", "IELTS 6.5"),
    ("English", "TOEIC 800"),
    ("Japanese", "N3"),
    ("Japanese", "N2"),
    ("Chinese", "HSK 4"),
]
AWARDS = [
    "Top 10 ICPC Regional",
    "Học bổng khuyến khích học tập",
    "Champion Hackathon 2024",
    "Giải Nhì Olympic Tin học",
    "Best Freshman Project",
]
PROJECT_NAMES = [
    "HỆ THỐNG QUẢN LÝ ĐOÀN THANH NIÊN PTIT",
    "HỆ THỐNG THỰC HÀNH LẬP TRÌNH ẢO CODEPTIT",
    "Interview Processing Platform",
    "E-commerce Inventory Dashboard",
    "Smart Home IoT Gateway",
    "Resume Ranking Service",
]


def _rand_email(name: str) -> str:
    slug = "".join(c for c in name.lower() if c.isalnum())[:12] or "user"
    return f"{slug}{random.randint(1, 99)}@gmail.com"


def _rand_phone() -> str:
    return f"0{random.randint(3, 9)}{random.randint(10000000, 99999999)}"


def make_gold(lang: str = "vn") -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return (correct_profile, section_texts)."""
    name = random.choice(VN_NAMES if lang == "vn" else EN_NAMES)
    job_title = random.choice(TITLES)
    n_jobs = random.randint(2, 4)
    companies = random.sample(COMPANIES, k=min(n_jobs, len(COMPANIES)))
    titles = [random.choice(TITLES) for _ in companies]

    experience = []
    exp_lines = []
    for i, (co, title) in enumerate(zip(companies, titles)):
        year = 2020 + i
        time_s = f"{year} - {year + 1}" if i < len(companies) - 1 else f"{year} - Nay"
        ach = [
            f"Phát triển REST API với {random.choice(BACKEND)}",
            f"Triển khai module {random.choice(FRONTEND)}",
        ]
        experience.append({
            "company": co,
            "position": title,
            "time": time_s,
            "start_date": str(year),
            "end_date": None if "Nay" in time_s else str(year + 1),
            "description": None,
            "achievements": ach,
        })
        if random.random() < 0.5:
            exp_lines.append(f"{co} - {title}\n{time_s}\n- {ach[0]}\n- {ach[1]}")
        else:
            exp_lines.append(f"{co}\n{title}\n{time_s}\n- {ach[0]}\n- {ach[1]}")

    school = random.choice(SCHOOLS)
    major = random.choice(MAJORS)
    gpa = f"{random.uniform(2.8, 3.9):.1f}"
    edu_year = f"20{random.randint(18, 22)} - 20{random.randint(23, 26)}"
    education = [{
        "school": school,
        "major": major,
        "degree": "Cử nhân" if lang == "vn" else "Bachelor",
        "year": edu_year,
        "start_date": edu_year.split(" - ")[0],
        "end_date": edu_year.split(" - ")[1],
        "gpa": gpa,
    }]

    n_proj = random.randint(2, 3)
    projects = []
    proj_lines = []
    for pname in random.sample(PROJECT_NAMES, k=n_proj):
        techs = random.sample(PROG_SKILLS + BACKEND + FRONTEND + DBS, k=4)
        resp = ["Thiết kế DB schema", "Implement API endpoints", "UI dashboard"]
        projects.append({
            "name": pname,
            "description": f"Dự án {pname.title()}",
            "technologies": techs,
            "responsibilities": resp,
            "github": f"https://github.com/user/{pname.lower().replace(' ', '-')[:20]}",
            "demo": None,
        })
        proj_lines.append(
            f"{pname}\n{edu_year}\nCông nghệ: {', '.join(techs)}\n- {resp[0]}\n- {resp[1]}"
        )

    prog = random.sample(PROG_SKILLS, k=3)
    be = random.sample(BACKEND, k=2)
    fe = random.sample(FRONTEND, k=2)
    db = random.sample(DBS, k=2)
    explicit = prog + be + fe + db
    skills = {
        "explicit": explicit,
        "categories": {
            "Language": prog,
            "Backend": be,
            "Frontend": fe,
            "Database": db,
        },
        "soft_skills": ["Teamwork", "Problem solving"],
    }

    spoken = random.sample(SPOKEN, k=random.randint(1, 2))
    languages = [{"language": a, "proficiency": b} for a, b in spoken]

    awards = []
    award_lines = []
    for title in random.sample(AWARDS, k=random.randint(1, 2)):
        awards.append({"title": title, "issuer": None, "date": "2024", "description": None})
        award_lines.append(f"{title} (2024)")

    activities = [{
        "organization": "CLB Lập trình",
        "role": "Thành viên",
        "period": "2022 - 2024",
        "description": "Tổ chức workshop",
    }]

    certs = [{
        "name": "AWS Cloud Practitioner" if random.random() < 0.5 else "Toeic",
        "issuer": "Amazon" if random.random() < 0.5 else "IIG",
        "date": "2023",
    }]

    profile = empty_profile()
    profile["personal_info"] = {
        "name": name,
        "email": _rand_email(name),
        "phone": _rand_phone(),
        "location": "Hà Nội" if lang == "vn" else "Ho Chi Minh City",
        "job_title": job_title,
        "github": f"https://github.com/{name.split()[-1].lower()}",
        "linkedin": None,
        "summary": f"Software engineer focused on {random.choice(BACKEND)}",
    }
    profile["education"] = education
    profile["experience"] = experience
    profile["projects"] = projects
    profile["skills"] = skills
    profile["languages"] = languages
    profile["certificates"] = certs
    profile["awards"] = awards
    profile["activities"] = activities

    sections = {
        "personal": (
            f"{name}\n{job_title}\n{profile['personal_info']['email']}\n"
            f"{profile['personal_info']['phone']}\n{profile['personal_info']['location']}"
        ),
        "education": f"{school}\nNgành: {major}\nGPA: {gpa}\n{edu_year}",
        "experience": "\n".join(exp_lines),
        "projects": "\n".join(proj_lines),
        "skills": (
            f"Language\n{', '.join(prog)}\nBackend\n{', '.join(be)}\n"
            f"Frontend\n{', '.join(fe)}\nDatabase\n{', '.join(db)}"
        ),
        "languages": "\n".join(f"{a} - {b}" for a, b in spoken),
        "awards": "\n".join(award_lines),
        "activities": "CLB Lập trình\nThành viên\n2022 - 2024\n- Tổ chức workshop",
        "certificates": f"{certs[0]['name']} - {certs[0]['issuer']} ({certs[0]['date']})",
    }
    return profile, sections


def inject_errors(gold: Dict[str, Any]) -> Dict[str, Any]:
    """Create a wrong deterministic draft from gold profile."""
    draft = copy.deepcopy(gold)
    ops = []

    # Always apply a few realistic mistakes
    if draft["experience"] and random.random() < 0.85:
        # Swap company/position on first job
        e0 = draft["experience"][0]
        e0["company"], e0["position"] = e0["position"], e0["company"]
        ops.append("swap_title")

    if len(draft["experience"]) >= 2 and random.random() < 0.7:
        # Merge jobs into one blob in achievements
        first = draft["experience"][0]
        rest = draft["experience"][1:]
        for other in rest:
            first["achievements"] = list(first.get("achievements") or []) + [
                other["company"], other["position"],
                *(other.get("achievements") or []),
            ]
        draft["experience"] = [first]
        ops.append("merge_jobs")

    if draft["education"] and random.random() < 0.75:
        edu = draft["education"][0]
        draft["awards"] = list(draft.get("awards") or []) + [{
            "title": edu["school"],
            "issuer": None,
            "date": edu.get("year"),
            "description": f"GPA: {edu.get('gpa')}",
        }]
        if random.random() < 0.6:
            draft["education"] = []
        ops.append("edu_in_awards")

    if draft["skills"].get("explicit") and random.random() < 0.8:
        prog = (draft["skills"].get("categories") or {}).get("Language") or draft["skills"]["explicit"][:2]
        draft["languages"] = list(draft.get("languages") or []) + [
            {"language": p, "proficiency": None} for p in prog[:2]
        ]
        ops.append("prog_in_languages")

    if draft["projects"] and len(draft["projects"]) >= 2 and random.random() < 0.5:
        p0 = draft["projects"][0]
        for p in draft["projects"][1:]:
            p0["responsibilities"] = list(p0.get("responsibilities") or []) + [
                p.get("name") or "", *(p.get("responsibilities") or [])
            ]
        draft["projects"] = [p0]
        ops.append("merge_projects")

    draft["_injected"] = ops
    return draft


def build_sample(lang: str) -> Dict[str, Any]:
    gold, sections = make_gold(lang)
    draft = inject_errors(gold)
    # Strip meta from completion
    completion_obj = copy.deepcopy(gold)
    user = format_user_prompt(
        {k: v for k, v in draft.items() if not str(k).startswith("_")},
        sections,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(completion_obj, ensure_ascii=False)},
        ],
        "meta": {
            "lang": lang,
            "injected": draft.get("_injected", []),
            "completion_chars": len(json.dumps(completion_obj, ensure_ascii=False)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CV field-mapper SFT JSONL")
    parser.add_argument("--n", type=int, default=1600, help="Total samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(args.n):
        lang = "vn" if i % 3 != 0 else "en"
        samples.append(build_sample(lang))

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val, train = samples[:n_val], samples[n_val:]

    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    for path, rows in ((train_path, train), (val_path, val)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg_chars = sum(s["meta"]["completion_chars"] for s in samples) / len(samples)
    print(f"Wrote {len(train)} train → {train_path}")
    print(f"Wrote {len(val)} val → {val_path}")
    print(f"Avg completion chars: {avg_chars:.0f}")


if __name__ == "__main__":
    main()
