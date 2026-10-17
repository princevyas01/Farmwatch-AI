"""Dataset extraction and provenance script for FarmWatch AI.

Safely handles Windows filename restrictions (e.g., sanitizing '?' or ':' in web-scraped filenames)
when unpacking the official PlantDoc benchmark repository from git objects into data/raw/plantdoc/.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def sanitize_filename(filename: str) -> str:
    """Sanitize invalid Windows NTFS characters and truncate excessively long filenames."""
    parts = filename.replace("\\", "/").split("/")
    sanitized_parts = []
    for i, part in enumerate(parts):
        clean_part = re.sub(r'[<>:"/\\|?*]', '_', part)
        clean_part = clean_part.strip('. ')
        # If this is the filename (last segment) and too long, truncate with hash
        if i == len(parts) - 1 and len(clean_part) > 80:
            stem = Path(clean_part).stem
            suffix = Path(clean_part).suffix
            import hashlib
            h = hashlib.sha1(part.encode('utf-8')).hexdigest()[:8]
            clean_part = f"{stem[:50]}_{h}{suffix}"
        sanitized_parts.append(clean_part)
    return "/".join(sanitized_parts)


def get_windows_safe_path(p: Path) -> str:
    abs_str = str(p.resolve())
    if os.name == 'nt' and not abs_str.startswith('\\\\?\\'):
        return f"\\\\?\\{abs_str}"
    return abs_str


def extract_git_tree_files(repo_dir: Path, target_dir: Path) -> Tuple[int, int, List[str]]:
    """Extract all tracked files from a git repository safely to target_dir with path sanitization."""
    cmd = ["git", "-C", str(repo_dir), "ls-tree", "-r", "HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    extracted_count = 0
    skipped_count = 0
    extracted_classes = set()

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        mode, obj_type, sha, raw_path = parts
        if obj_type != "blob":
            continue

        clean_path = sanitize_filename(raw_path)
        dest_path = target_dir / clean_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        cat_cmd = ["git", "-C", str(repo_dir), "cat-file", "-p", sha]
        cat_result = subprocess.run(cat_cmd, capture_output=True, check=True)

        safe_path = get_windows_safe_path(dest_path)
        with open(safe_path, "wb") as f:
            f.write(cat_result.stdout)

        extracted_count += 1
        clean_parts = clean_path.split("/")
        if len(clean_parts) >= 3 and clean_parts[0] in ("train", "test"):
            extracted_classes.add(clean_parts[1])

    return extracted_count, skipped_count, sorted(list(extracted_classes))


def setup_raw_dataset(project_root: Path) -> Dict[str, any]:
    """Inspect and unpack raw dataset files for the project."""
    raw_fieldplant = project_root / "data" / "raw" / "fieldplant"
    raw_plantdoc = project_root / "data" / "raw" / "plantdoc"

    print("Checking dataset status...")
    # Check if fieldplant has downloaded files
    has_fieldplant = raw_fieldplant.exists() and any(raw_fieldplant.iterdir())

    if has_fieldplant:
        print(f"Primary FieldPlant data detected at: {raw_fieldplant}")
        return {
            "dataset_name": "FieldPlant v11",
            "source_type": "primary",
            "path": str(raw_fieldplant),
            "license": "CC BY 4.0",
            "citation": "FieldPlant (IEEE Access 2023, DOI: 10.1109/ACCESS.2023.3263042)",
        }

    # Otherwise use PlantDoc fallback repository
    git_dir = raw_plantdoc / ".git"
    if git_dir.exists():
        print(f"Extracting PlantDoc dataset from git objects into {raw_plantdoc} with Windows path sanitization...")
        extracted, skipped, classes = extract_git_tree_files(raw_plantdoc, raw_plantdoc)
        print(f"Successfully extracted {extracted} files across {len(classes)} classes into {raw_plantdoc}.")
        return {
            "dataset_name": "PlantDoc-Dataset (Fallback)",
            "source_type": "fallback",
            "path": str(raw_plantdoc),
            "files_extracted": extracted,
            "classes_detected": classes,
            "license": "CC BY 4.0",
            "citation": "PlantDoc: A Dataset for Visual Plant Disease Detection (ACM CoDS-COMAD 2020)",
            "fallback_reason": "Roboflow Universe FieldPlant v11 export requires manual browser authentication/API key. PlantDoc used as the verified open-access CC BY 4.0 field-imagery fallback.",
        }

    raise RuntimeError(
        "Neither FieldPlant nor PlantDoc dataset found. Please place FieldPlant in data/raw/fieldplant/ or clone PlantDoc to data/raw/plantdoc/."
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    status = setup_raw_dataset(root)
    print("Dataset setup summary:", status)
