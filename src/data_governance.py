import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
META_DIR = DATA_DIR / "metadata"
VERSIONS_LOG = META_DIR / "versions.jsonl"
LATEST_PROFILE = META_DIR / "latest_profile.json"

XLSX_PATH = ROOT / "NUST Bank-Product-Knowledge.xlsx"
FAQ_PATH = ROOT / "faq.json"
QA_PATH = DATA_DIR / "qa_pairs.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _summary(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "avg": 0, "max": 0}
    return {"min": min(values), "avg": round(mean(values), 2), "max": max(values)}


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _profile_pairs(pairs: list[dict]) -> dict:
    missing_question = 0
    missing_answer = 0
    duplicate_count = 0

    source_counter = Counter()
    product_counter = Counter()

    q_lengths = []
    a_lengths = []

    seen = set()
    for p in pairs:
        q = str(p.get("question", "")).strip()
        a = str(p.get("answer", "")).strip()
        s = str(p.get("source", "unknown")).strip() or "unknown"
        prod = str(p.get("product", "unknown")).strip() or "unknown"

        source_counter[s] += 1
        product_counter[prod] += 1

        if not q:
            missing_question += 1
        if not a:
            missing_answer += 1

        q_lengths.append(_word_count(q))
        a_lengths.append(_word_count(a))

        key = (_normalize(q), _normalize(a))
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)

    return {
        "total_pairs": len(pairs),
        "missing_question": missing_question,
        "missing_answer": missing_answer,
        "duplicate_pairs": duplicate_count,
        "question_length_words": _summary(q_lengths),
        "answer_length_words": _summary(a_lengths),
        "sources": dict(source_counter),
        "top_products": dict(product_counter.most_common(15)),
    }


def _profile_chunks(chunks: list[dict]) -> dict:
    source_counter = Counter()
    product_counter = Counter()
    text_lengths = []
    empty_text = 0

    for c in chunks:
        text = str(c.get("text", "")).strip()
        source_counter[str(c.get("source", "unknown")).strip() or "unknown"] += 1
        product_counter[str(c.get("product", "unknown")).strip() or "unknown"] += 1
        if not text:
            empty_text += 1
        text_lengths.append(_word_count(text))

    return {
        "total_chunks": len(chunks),
        "empty_chunk_text": empty_text,
        "chunk_length_words": _summary(text_lengths),
        "sources": dict(source_counter),
        "top_products": dict(product_counter.most_common(15)),
    }


def _build_profile(tag: str | None = None) -> dict:
    pairs = _load_json(QA_PATH)
    chunks = _load_json(CHUNKS_PATH)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag or "",
        "input_files": {
            "xlsx": str(XLSX_PATH.name),
            "faq": str(FAQ_PATH.name),
        },
        "file_hashes": {
            "xlsx_sha256": _sha256(XLSX_PATH),
            "faq_sha256": _sha256(FAQ_PATH),
            "qa_pairs_sha256": _sha256(QA_PATH),
            "chunks_sha256": _sha256(CHUNKS_PATH),
        },
        "pairs_profile": _profile_pairs(pairs),
        "chunks_profile": _profile_chunks(chunks),
    }


def _version_id(profile: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = (profile["file_hashes"].get("chunks_sha256") or "nohash")[:8]
    return f"v_{stamp}_{digest}"


def _write_profile_and_version(profile: dict) -> tuple[Path, Path]:
    META_DIR.mkdir(parents=True, exist_ok=True)
    version = _version_id(profile)

    profile_path = META_DIR / f"profile_{version}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    with open(LATEST_PROFILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    version_entry = {
        "version": version,
        "timestamp": profile["generated_at"],
        "tag": profile.get("tag", ""),
        "records": {
            "pairs": profile["pairs_profile"]["total_pairs"],
            "chunks": profile["chunks_profile"]["total_chunks"],
        },
        "hashes": profile["file_hashes"],
        "artifacts": {
            "profile": str(profile_path.relative_to(ROOT)),
            "latest_profile": str(LATEST_PROFILE.relative_to(ROOT)),
            "qa_pairs": str(QA_PATH.relative_to(ROOT)) if QA_PATH.exists() else None,
            "chunks": str(CHUNKS_PATH.relative_to(ROOT)) if CHUNKS_PATH.exists() else None,
        },
    }

    with open(VERSIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(version_entry, ensure_ascii=False) + "\n")

    return profile_path, VERSIONS_LOG


def main() -> None:
    parser = argparse.ArgumentParser(description="Data profiling and versioning utility")
    parser.add_argument("--run-pipeline", action="store_true", help="Regenerate qa_pairs.json and chunks.json first")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index after pipeline run")
    parser.add_argument("--tag", default="", help="Optional tag for this data version")
    args = parser.parse_args()

    if args.run_pipeline:
        from data_pipeline import run_pipeline

        run_pipeline()

    if args.rebuild_index:
        from retriever import build_index

        build_index()

    profile = _build_profile(tag=args.tag)
    profile_path, versions_log = _write_profile_and_version(profile)

    print("Data governance run completed")
    print(f"Pairs: {profile['pairs_profile']['total_pairs']}")
    print(f"Chunks: {profile['chunks_profile']['total_chunks']}")
    print(f"Profile: {profile_path}")
    print(f"Versions log: {versions_log}")


if __name__ == "__main__":
    main()
