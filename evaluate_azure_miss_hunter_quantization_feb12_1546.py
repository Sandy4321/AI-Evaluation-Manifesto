"""
history

evaluate_azure_miss_hunter_quantization_feb11_1531_wo_fulltext.py
"""

import os
import re
import sys
import time
import uuid
import json
import random
import hashlib
import datetime
import shutil
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pypdf
from dotenv import load_dotenv

from azure.core.exceptions import ServiceResponseError
from http.client import RemoteDisconnected

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
    BinaryQuantizationCompression,
    ScalarQuantizationCompression,
)
from azure.search.documents.models import VectorizedQuery

try:
    from azure.search.documents.indexes.models import RescoringOptions
except Exception:
    RescoringOptions = None

from openai import AzureOpenAI

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# =============================
# CONFIG (EDIT)
# =============================
INDEX_NAME = "index-eval-ablation-miss-hunter"

DATA_FOLDER = "data"
RUNS_ROOT = "runs"



MAX_BASE_CHUNKS_TO_PROBE = 2000 #900 #10 #20 #900 #300#1234

CHUNK_SIZE = 150#200
OVERLAP = 10#75#100

HNSW_M = 4
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 100

K = 5
INDEXING_WAIT_SEC = 20


RANDOM_SEED = 42

PERTURB_MODE = "insert"   # "remove" or "insert"

# REMOVE MODE
N_RANDOM_SETS = 10 # how many random removing happen if PERTURB_MODE = "remove"
N_WORDS_REMOVE = 7 #23
MAX_ABLATIONS_PER_CHUNK = 200
MIN_WORD_LEN = 5

# INSERT MODE
INSERT_ENABLED =  True #True  False
INSERT_TEXT = " not at all "
INSERT_AFTER_WORD_NUM = 10 #17

# Quantization: "none" | "scalar" | "binary"
QUANTIZATION_MODE = "scalar"
ENABLE_RERANK_WITH_ORIGINAL = False
OVERSAMPLING = 2.0

# Matryoshka (MRL) local scoring (prefix dims + renormalize)
USE_MATRYOSHKA = False
MATRYOSHKA_DIMS = 10

PREVIEW_CHARS = 260

BATCH_EMBED = 16
BATCH_UPLOAD = 500

# =============================
# STORAGE SWITCH 
# =============================
# "all"  -> store full chunk text in index field contentStored
# "head" -> store only first N chars in contentStored (saves Total storage)
STORE_TEXT_MODE = "head"  # "all" or "head"
STORE_TEXT_HEAD_CHARS = 50 #10 #240  # used only when STORE_TEXT_MODE="head"


STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","so","to","of","in","on","at","for","with",
    "as","by","from","is","are","was","were","be","been","being","it","this","that","these","those",
    "i","you","he","she","we","they","them","his","her","our","their","my","your","me","him","us",
    "not","no","yes","do","does","did","doing","done","can","could","should","would","may","might",
    "will","shall","have","has","had","having","into","over","under","between","within","without",
    "also","just","than","too","very","more","most","less","least","such"
}

_word_re = re.compile(r"\b[\w']+\b", re.UNICODE)


# =============================
# REPRODUCIBLE RUN FOLDER
# =============================
def select_random_docs(docs, N):
    if N <= 0:
        return []
    k = min(N, len(docs))
    return random.sample(docs, k)

def create_run_folder() -> Tuple[str, Path, Path, Path, Path]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_ROOT) / f"run_{ts}"
    reports_dir = run_dir / "reports"
    outputs_dir = run_dir / "outputs"
    snapshot_dir = run_dir / "snapshot"
    snapshot_data_dir = snapshot_dir / "data"

    reports_dir.mkdir(parents=True, exist_ok=False)
    outputs_dir.mkdir(parents=True, exist_ok=False)
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    try:
        src_script = Path(__file__).resolve()
        shutil.copy2(src_script, snapshot_dir / src_script.name)
    except Exception as e:
        print(f"⚠️ Could not snapshot script (__file__ unavailable?): {e}")

    data_src = Path(DATA_FOLDER).resolve()
    if data_src.exists() and data_src.is_dir():
        shutil.copytree(data_src, snapshot_data_dir)
    else:
        print(f"⚠️ DATA_FOLDER not found or not a directory: {data_src}")

    return ts, run_dir, reports_dir, outputs_dir, snapshot_data_dir


def run_suffix(use_matryoshka: bool, perturb_mode: str, quant_mode: str, n_chunks: int, store_mode: str, head_chars: int) -> str:
    mtag = "M" if use_matryoshka else "woM"
    pm = (perturb_mode or "").lower().strip()
    ptag = "INS" if pm == "insert" else "Rem"
    qm = (quant_mode or "none").lower().strip()
    qtag = {"none": "F", "scalar": "S", "binary": "B"}.get(qm, "F")

    sm = (store_mode or "all").lower().strip()
    stag = "ALLTXT" if sm == "all" else f"HEAD{int(head_chars)}"
    return f"{mtag}_{ptag}_{qtag}_{stag}_nCH{int(n_chunks)}"


def rename_run_folder(run_dir: Path, ts: str, suffix: str) -> Path:
    new_dir = Path(RUNS_ROOT) / f"run_{ts}_{suffix}"
    if new_dir == run_dir:
        return run_dir
    if new_dir.exists():
        new_dir = Path(RUNS_ROOT) / f"run_{ts}_{suffix}_{uuid.uuid4().hex[:6]}"
    shutil.move(str(run_dir), str(new_dir))
    return new_dir


# =============================
# PARAMETER HARMONIZATION
# =============================
def harmonize_quantization_params(
    quant_mode: str, enable_rerank: bool, oversampling: Optional[float]
) -> Tuple[str, bool, Optional[float]]:
    qm = (quant_mode or "none").lower().strip()
    if qm not in ("none", "scalar", "binary"):
        qm = "none"
    if qm == "none":
        return qm, False, None

    rr = bool(enable_rerank)
    if not rr:
        return qm, False, None

    osamp = float(oversampling) if oversampling is not None else 2.0
    if osamp < 1.0:
        osamp = 1.0
    return qm, True, osamp


# =============================
# Helpers
# =============================
def preview_text(s: str, n: int = PREVIEW_CHARS) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[:n] + " ...")


def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def vec_hash(vec: np.ndarray, decimals: int = 8) -> str:
    v = np.round(vec.astype(np.float32), decimals=decimals)
    return hashlib.sha256(v.tobytes()).hexdigest()[:16]


def normalize_rows(V: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return V / norms


def matryoshka_prefix_dims(full_dims: int, use_matryoshka: bool, prefix_dims: int) -> int:
    if not use_matryoshka:
        return int(full_dims)
    k = int(prefix_dims)
    if k <= 0:
        return int(full_dims)
    return int(min(full_dims, k))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[0]
    if k >= n:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("-\n", "")
    text = text.replace("\n\n", "<PARA_BREAK>")
    text = text.replace("\n", " ")
    text = text.replace("<PARA_BREAK>", "\n\n")
    text = re.sub(r" +", " ", text)
    return text.strip()


def chunk_text_paragraphwise(text: str, chunk_size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            parts = splitter.split_text(p)
            if len(parts) > 1 and len(parts[-1]) < int(chunk_size * 0.2):
                parts[-2] = parts[-2] + " " + parts[-1]
                parts.pop()
            chunks.extend(parts)
    return chunks


def make_content_stored(full_text: str) -> str:
    s = (full_text or "").strip()
    mode = (STORE_TEXT_MODE or "all").lower().strip()
    if mode == "all":
        return s
    # "head"
    n = int(STORE_TEXT_HEAD_CHARS)
    if n <= 0:
        return ""  # extreme mode (store nothing)
    return s if len(s) <= n else (s[:n] + " ...")


def read_pdf_chunks(folder: str) -> List[Dict[str, Any]]:
    folderp = Path(folder)
    pdfs = list(folderp.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in {folderp.resolve()}")
        return []

    docs: List[Dict[str, Any]] = []
    print(f"📂 Reading {len(pdfs)} PDFs from snapshot data folder...")
    for fp in pdfs:
        reader = pypdf.PdfReader(fp)
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        cleaned = clean_pdf_text("\n\n".join(pages))
        if not cleaned:
            continue
        chunks = chunk_text_paragraphwise(cleaned, CHUNK_SIZE, OVERLAP)
        for i, ch in enumerate(chunks, 1):
            full = ch
            docs.append(
                {
                    "title": fp.stem,
                    "source": fp.name,
                    "chunk_id": f"{fp.stem}_chunk{i}",
                    "content_full": full,                      # used for embeddings + perturbations
                    "contentStored": make_content_stored(full), # what we store/return from Azure
                }
            )
    print(f"✅ Total original chunks: {len(docs)}")
    random_docs = select_random_docs(docs, N=MAX_BASE_CHUNKS_TO_PROBE)
    print(f"✅ Total actual random  chunks: {len(random_docs)}")
    return random_docs


def _tsv_safe(x: str) -> str:
    return (x or "").replace("\t", " ").replace("\n", " ").strip()


def _jsonl_write(fp, obj: Dict[str, Any]):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


# =============================
# INSERT perturbation
# =============================
def insert_text_after_word_number(text: str, insert_text: str, after_word_num_1based: int) -> Tuple[str, Dict[str, Any]]:
    insert_text = insert_text if insert_text is not None else ""
    n_req = int(after_word_num_1based)

    words = list(_word_re.finditer(text))
    wc = len(words)

    if wc == 0:
        pos = len(text)
        new_text = text + insert_text
        meta = {
            "inserted_text": insert_text,
            "requested_after_word_num": n_req,
            "actual_after_word_num": 0,
            "word_count": 0,
            "insert_char_pos": pos,
            "appended_to_end": True,
        }
        return new_text, meta

    if n_req <= 0:
        pos = 0
        new_text = text[:pos] + insert_text + text[pos:]
        meta = {
            "inserted_text": insert_text,
            "requested_after_word_num": n_req,
            "actual_after_word_num": 0,
            "word_count": wc,
            "insert_char_pos": pos,
            "appended_to_end": False,
        }
        return new_text, meta

    if wc < n_req:
        pos = len(text)
        new_text = text + insert_text
        meta = {
            "inserted_text": insert_text,
            "requested_after_word_num": n_req,
            "actual_after_word_num": wc,
            "word_count": wc,
            "insert_char_pos": pos,
            "appended_to_end": True,
        }
        return new_text, meta

    m = words[n_req - 1]
    pos = m.end()
    new_text = text[:pos] + insert_text + text[pos:]
    meta = {
        "inserted_text": insert_text,
        "requested_after_word_num": n_req,
        "actual_after_word_num": n_req,
        "word_count": wc,
        "insert_char_pos": pos,
        "appended_to_end": False,
    }
    return new_text, meta


# =============================
# REMOVE perturbation
# =============================
def unique_eligible_word_spans(text: str) -> List[Tuple[str, int, int]]:
    """One span per UNIQUE eligible word (case-insensitive)."""
    seen = set()
    out: List[Tuple[str, int, int]] = []
    for m in _word_re.finditer(text):
        start, end = m.start(), m.end()
        w = text[start:end]
        wl = w.lower()

        if len(w) < MIN_WORD_LEN:
            continue
        if wl in STOPWORDS:
            continue
        if wl in seen:
            continue

        seen.add(wl)
        out.append((wl, start, end))
    return out


def remove_many_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text

    spans_sorted = sorted(spans, key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in spans_sorted:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s < pe:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])

    out = text
    for s, e in sorted([(a, b) for a, b in merged], key=lambda x: x[0], reverse=True):
        out = out[:s] + out[e:]

    out = out.strip()
    out = re.sub(r"\s{2,}", " ", out)
    return out


def generate_removed_queries_fixed_nsets(
    base_text: str,
    rng: random.Random,
    n_random_sets: int,
    n_words_remove: int,
    max_attempts_factor: int = 250,
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], Dict[str, Any]]:
    spans = unique_eligible_word_spans(base_text)
    n = len(spans)
    gsize = max(1, int(n_words_remove))
    req = max(0, int(n_random_sets))

    meta: Dict[str, Any] = {
        "eligible_words": n,
        "n_words_remove": gsize,
        "requested_sets": req,
        "max_unique_sets_possible": 0,
        "generated_sets": 0,
        "attempts": 0,
    }

    if n == 0 or req == 0:
        return [], meta

    if n < gsize:
        return [], meta

    comb_space = math.comb(n, gsize)
    meta["max_unique_sets_possible"] = comb_space
    target = min(req, comb_space, int(MAX_ABLATIONS_PER_CHUNK))

    idxs = list(range(n))
    seen = set()
    out: List[Tuple[str, List[Dict[str, Any]]]] = []

    max_attempts = max(1500, target * max_attempts_factor)
    attempts = 0

    while len(out) < target and attempts < max_attempts:
        chosen = tuple(sorted(rng.sample(idxs, gsize)))
        if chosen in seen:
            attempts += 1
            continue
        seen.add(chosen)

        removed_items: List[Dict[str, Any]] = []
        removal_spans: List[Tuple[int, int]] = []
        ok = True

        for j in chosen:
            wl, start, end = spans[j]
            removed_original = base_text[start:end]
            if removed_original.lower() != wl:
                ok = False
                break
            removed_items.append({"word_lower": wl, "word": removed_original, "start": start, "end": end})
            removal_spans.append((start, end))

        if not ok:
            attempts += 1
            continue

        new_text = remove_many_spans(base_text, removal_spans)
        out.append((new_text, removed_items))
        attempts += 1

    meta["generated_sets"] = len(out)
    meta["attempts"] = attempts
    return out, meta


# =============================
# Azure clients
# =============================
def get_aoai_client():
    load_dotenv(override=True)
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    emb_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    if not all([api_key, endpoint, emb_deployment]):
        print("❌ Missing .env vars for Azure OpenAI.")
        sys.exit(1)

    aoai = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
    resp = aoai.embeddings.create(input=["ping"], model=emb_deployment)
    dims = len(resp.data[0].embedding)
    return aoai, emb_deployment, dims


def get_search_clients():
    load_dotenv(override=True)
    service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    if not all([service_endpoint, admin_key]):
        print("❌ Missing .env vars for Azure AI Search.")
        sys.exit(1)

    cred = AzureKeyCredential(admin_key)
    index_client = SearchIndexClient(endpoint=service_endpoint, credential=cred)
    search_client = SearchClient(endpoint=service_endpoint, index_name=INDEX_NAME, credential=cred)
    return index_client, search_client


def embed_texts(aoai: AzureOpenAI, emb_deployment: str, texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), BATCH_EMBED):
        batch = texts[i : i + BATCH_EMBED]
        resp = aoai.embeddings.create(input=batch, model=emb_deployment)
        out.extend([d.embedding for d in resp.data])
    return out


def _set_rescoring(comp_obj, enable_rerank: bool, oversampling: Optional[float]):
    if RescoringOptions is not None:
        try:
            if enable_rerank:
                comp_obj.rescoring_options = RescoringOptions(
                    enable_rescoring=True,
                    default_oversampling=float(oversampling) if oversampling is not None else 1.0,
                )
            else:
                comp_obj.rescoring_options = RescoringOptions(enable_rescoring=False)
            return
        except Exception:
            pass

    try:
        comp_obj.rerank_with_original_vectors = bool(enable_rerank)
        if enable_rerank and oversampling is not None:
            comp_obj.default_oversampling = float(oversampling)
        return
    except Exception:
        raise RuntimeError("Upgrade azure-search-documents: pip install -U azure-search-documents")


# =============================
# INDEX: VECTOR-ONLY (NO BM25) + contentStored
# =============================
def build_index_vector_only(dims: int, qm: str, rr: bool, osamp: Optional[float]) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="title", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True, retrievable=True),

        # Store only preview or full text depending on STORE_TEXT_MODE
        SimpleField(name="contentStored", type=SearchFieldDataType.String, retrievable=True),

        # Vector field (searchable only for vector engine)
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dims,
            vector_search_profile_name="vec-profile",
        ),
    ]

    algo = HnswAlgorithmConfiguration(
        name="hnsw-algo",
        parameters=HnswParameters(
            m=HNSW_M,
            ef_construction=HNSW_EF_CONSTRUCTION,
            ef_search=HNSW_EF_SEARCH,
            metric="cosine",
        ),
    )

    compressions = None
    compression_name = None
    if qm != "none":
        compression_name = f"{qm}-quant"
        comp = (
            BinaryQuantizationCompression(compression_name=compression_name)
            if qm == "binary"
            else ScalarQuantizationCompression(compression_name=compression_name)
        )
        _set_rescoring(comp, rr, osamp)
        compressions = [comp]

    profile = VectorSearchProfile(
        name="vec-profile",
        algorithm_configuration_name="hnsw-algo",
        compression_name=compression_name,
    )
    vector_search = VectorSearch(algorithms=[algo], compressions=compressions, profiles=[profile])
    return SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)


def ensure_index(index_client: SearchIndexClient, dims: int, qm: str, rr: bool, osamp: Optional[float]):
    idx = build_index_vector_only(dims, qm, rr, osamp)
    try:
        index_client.delete_index(INDEX_NAME)
    except Exception:
        pass
    index_client.create_index(idx)
    print(f"✅ Created VECTOR-ONLY index (no BM25): {INDEX_NAME}")
    print(f"✅ Text storage mode: STORE_TEXT_MODE={STORE_TEXT_MODE!r}, head_chars={STORE_TEXT_HEAD_CHARS}")


def upload_docs(search_client: SearchClient, docs: List[Dict[str, Any]], vectors: List[List[float]]):
    actions = []
    for d, v in zip(docs, vectors):
        actions.append(
            {
                "id": str(uuid.uuid4()),
                "title": d["title"],
                "source": d["source"],
                "chunk_id": d["chunk_id"],
                "contentStored": d["contentStored"],
                "contentVector": v,
            }
        )

    for i in range(0, len(actions), BATCH_UPLOAD):
        batch = actions[i : i + BATCH_UPLOAD]
        res = search_client.upload_documents(documents=batch)
        failed = [r for r in res if not r.succeeded]
        if failed:
            print(f"⚠️ Upload batch {i} had {len(failed)} failures.")
    print(f"✅ Uploaded {len(actions)} docs.")


def azure_ann_topk(search_client: SearchClient, qvec: np.ndarray) -> List[Dict[str, Any]]:
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            vq = VectorizedQuery(vector=qvec.tolist(), k_nearest_neighbors=K, fields="contentVector")
            results = search_client.search(
                search_text=None,
                vector_queries=[vq],
                select=["chunk_id", "contentStored"],
                top=K
            )
            # Materialize to trigger network calls inside try/except
            out = []
            for r in results:
                out.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "azure_score": float(r["@search.score"]),
                        "contentStored": r.get("contentStored", ""),
                    }
                )
            
            # Valid result obtained; sleep briefly to avoid saturating connection/rate-limits
            time.sleep(0.02)
            return out

        except ServiceResponseError as e:
            # This catches the "Connection aborted" / RemoteDisconnected errors
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt)
                # Jitter could be added, but simple exp backoff is likely enough here
                print(f"⚠️ Search connection error (attempt {attempt+1}/{max_retries+1}). Retrying in {sleep_time}s... Error: {e}")
                time.sleep(sleep_time)
            else:
                print(f"❌ Search failed after {max_retries+1} attempts.")
                raise e
    return []


def clear_all_documents(search_client: SearchClient, batch_size: int = 1000):
    total_deleted = 0
    while True:
        results = search_client.search(search_text="*", select=["id"], top=batch_size)
        ids = [{"id": r["id"]} for r in results]
        if not ids:
            break
        delete_actions = [{"@search.action": "delete", "id": d["id"]} for d in ids]
        resp = search_client.upload_documents(delete_actions)
        total_deleted += sum(1 for r in resp if r.succeeded)
        if len(ids) < batch_size:
            break
    print(f"✅ Cleared documents. Total deleted: {total_deleted}")


# =============================
# MAIN
# =============================
def main():
    ts, run_dir, reports_dir, outputs_dir, snapshot_data_dir = create_run_folder()

    mode = (PERTURB_MODE or "remove").strip().lower()
    if mode not in ("remove", "insert"):
        raise ValueError("PERTURB_MODE must be 'remove' or 'insert'")

    if (STORE_TEXT_MODE or "").lower().strip() not in ("all", "head"):
        raise ValueError("STORE_TEXT_MODE must be 'all' or 'head'")

    rng = random.Random(RANDOM_SEED)
    qm, rr, osamp = harmonize_quantization_params(QUANTIZATION_MODE, ENABLE_RERANK_WITH_ORIGINAL, OVERSAMPLING)

    print(f"📦 Run folder: {run_dir.resolve()}")
    print(f"⚙️ Mode: {mode}")
    print(f"🧾 STORE_TEXT_MODE={STORE_TEXT_MODE!r} head_chars={STORE_TEXT_HEAD_CHARS}")

    aoai, emb_deployment, dims = get_aoai_client()
    index_client, search_client = get_search_clients()

    docs = read_pdf_chunks(str(snapshot_data_dir))
    if not docs:
        return

    suf = run_suffix(USE_MATRYOSHKA, mode, qm, len(docs), STORE_TEXT_MODE, STORE_TEXT_HEAD_CHARS)
    new_run_dir = rename_run_folder(run_dir, ts, suf)
    if new_run_dir != run_dir:
        run_dir = new_run_dir
        reports_dir = run_dir / "reports"
        outputs_dir = run_dir / "outputs"
        snapshot_data_dir = run_dir / "snapshot" / "data"
        print(f"📁 Renamed run folder → {run_dir.resolve()}")

    # Embeddings always use FULL TEXT
    print(f"\n🔄 Embedding {len(docs)} chunks for indexing...")
    doc_vectors_list: List[List[float]] = []
    for i in range(0, len(docs), BATCH_EMBED):
        batch = docs[i : i + BATCH_EMBED]
        vecs = embed_texts(aoai, emb_deployment, [d["content_full"] for d in batch])
        doc_vectors_list.extend(vecs)
        print(f"Embeddings(index): {len(doc_vectors_list)}/{len(docs)}", end="\r")
    print("\n✅ Index embeddings done.")

    V = np.array(doc_vectors_list, dtype=np.float32)
    k_local = matryoshka_prefix_dims(dims, USE_MATRYOSHKA, MATRYOSHKA_DIMS)
    Vn = normalize_rows(V[:, :k_local])
    docs_by_id = {d["chunk_id"]: d for d in docs}

    ensure_index(index_client, dims, qm, rr, osamp)
    clear_all_documents(search_client)
    upload_docs(search_client, docs, doc_vectors_list)
    time.sleep(INDEXING_WAIT_SEC)

    full_report = reports_dir / f"full_report_{ts}.txt"
    err_mismatch_report = reports_dir / f"error_retrieval_set_mismatch_{ts}.txt"
    err_base_miss_report = reports_dir / f"error_ann_base_miss_{ts}.txt"
    rows_report = reports_dir / f"rows_{ts}.tsv"
    jsonl_out = outputs_dir / f"base_variants_embeddings_{ts}.jsonl"
    skipped_chunks_report = reports_dir / f"skipped_chunks_{ts}.tsv"

    header = []
    header.append("AZURE ANN MISS-HUNTER (VECTOR-ONLY INDEX + STORE_TEXT SWITCH)\n")
    header.append(f"Timestamp: {ts}\n")
    header.append(f"Run folder: {run_dir.resolve()}\n")
    header.append(f"Index: {INDEX_NAME}\n")
    header.append(f"Total chunks indexed: {len(docs)}\n")
    header.append(f"K={K}\n")
    header.append(f"PERTURB_MODE: {mode}\n")
    header.append(f"STORE_TEXT_MODE: {STORE_TEXT_MODE}  head_chars={STORE_TEXT_HEAD_CHARS}\n")
    header.append(f"Quantization: {qm}\n")
    header.append(f"Matryoshka local scoring: {bool(USE_MATRYOSHKA)}  (effective_dims={k_local})\n")
    header.append("=" * 110 + "\n\n")

    full_report.write_text("".join(header), encoding="utf-8")
    err_mismatch_report.write_text("".join(header) + "ERROR TYPE: RETRIEVAL_SET_MISMATCH\n" + "=" * 110 + "\n\n", encoding="utf-8")
    err_base_miss_report.write_text("".join(header) + "ERROR TYPE: ANN_BASE_MISS\n" + "=" * 110 + "\n\n", encoding="utf-8")

    rows_report.write_text(
        "\t".join(
            [
                "ts","base_num","base_chunk_id","query_kind",
                "removed_words_lower_csv","removed_words_original_csv","removed_starts_csv","removed_ends_csv",
                "insert_text","insert_after_word_num_requested","insert_after_word_num_actual",
                "insert_word_count","insert_char_pos","insert_appended_to_end",
                "eligible_words","requested_random_sets","max_unique_sets_possible",
                "ann_has_base","ann_rank_base","local_has_base","local_rank_base",
                f"overlap@{K}",f"recall@{K}",
                "ann_top_ids","local_top_ids",
            ]
        ) + "\n",
        encoding="utf-8"
    )

    skipped_chunks_report.write_text(
        "\t".join(["ts", "chunk_id", "reason", "eligible_words", "n_words_remove", "requested_random_sets", "max_unique_sets_possible"]) + "\n",
        encoding="utf-8"
    )

    jsonl_fp = open(jsonl_out, "w", encoding="utf-8")

    used_base_chunks = 0
    skipped_base_chunks = 0
    total_queries = 0
    mismatch_errors = 0
    base_miss_errors = 0

    order = list(range(len(docs)))
    rng.shuffle(order)

    try:
        for doc_idx in order:
            if used_base_chunks >= MAX_BASE_CHUNKS_TO_PROBE:
                break

            base = docs[doc_idx]
            base_id = base["chunk_id"]
            base_text = base["content_full"]  # FULL text for perturbations

            eligible_words = ""
            requested_sets = ""
            max_unique_sets_possible = ""

            variants: List[Dict[str, Any]] = [
                {"text": base_text, "kind": "ORIGINAL", "removed": None, "insert": None, "qid": f"{base_id}::ORIGINAL"}
            ]

            if mode == "insert":
                if INSERT_ENABLED:
                    ins_text, ins_meta = insert_text_after_word_number(base_text, INSERT_TEXT, INSERT_AFTER_WORD_NUM)
                    variants.append({"text": ins_text, "kind": "INSERTED", "removed": None, "insert": ins_meta, "qid": f"{base_id}::INSERTED::after{INSERT_AFTER_WORD_NUM}"})

            else:
                spans = unique_eligible_word_spans(base_text)
                eligible = len(spans)
                eligible_words = str(eligible)
                requested_sets = str(int(N_RANDOM_SETS))

                if eligible < int(N_WORDS_REMOVE):
                    skipped_base_chunks += 1
                    with open(skipped_chunks_report, "a", encoding="utf-8") as f:
                        f.write("\t".join([ts, base_id, "eligible_words_lt_N_WORDS_REMOVE", str(eligible), str(N_WORDS_REMOVE), str(N_RANDOM_SETS), ""]) + "\n")
                    continue

                comb_space = math.comb(eligible, int(N_WORDS_REMOVE))
                max_unique_sets_possible = str(comb_space)

                if comb_space < int(N_RANDOM_SETS):
                    skipped_base_chunks += 1
                    with open(skipped_chunks_report, "a", encoding="utf-8") as f:
                        f.write("\t".join([ts, base_id, "comb_space_lt_N_RANDOM_SETS", str(eligible), str(N_WORDS_REMOVE), str(N_RANDOM_SETS), str(comb_space)]) + "\n")
                    continue

                removed_variants, _ = generate_removed_queries_fixed_nsets(
                    base_text=base_text,
                    rng=rng,
                    n_random_sets=int(N_RANDOM_SETS),
                    n_words_remove=int(N_WORDS_REMOVE),
                )

                if not removed_variants:
                    skipped_base_chunks += 1
                    with open(skipped_chunks_report, "a", encoding="utf-8") as f:
                        f.write("\t".join([ts, base_id, "no_removed_variants_generated", str(eligible), str(N_WORDS_REMOVE), str(N_RANDOM_SETS), str(comb_space)]) + "\n")
                    continue

                for (new_text, removed_items) in removed_variants:
                    suffix = "__".join([f"{it['word_lower']}@{it['start']}" for it in removed_items])
                    variants.append({"text": new_text, "kind": "REMOVED", "removed": removed_items, "insert": None, "qid": f"{base_id}::REMOVED::{suffix}"})

            used_base_chunks += 1
            base_num = used_base_chunks

            variant_texts = [v["text"] for v in variants]
            vvecs_list = embed_texts(aoai, emb_deployment, variant_texts)

            _jsonl_write(
                jsonl_fp,
                {
                    "timestamp": ts,
                    "run_folder": str(run_dir.resolve()),
                    "base_chunk_id": base_id,
                    "base_text": base_text,
                    "base_text_hash": text_hash(base_text),
                    "base_embedding": vvecs_list[0],
                    "variants": [
                        {
                            "query_id": variants[i]["qid"],
                            "query_kind": variants[i]["kind"],
                            "removed": variants[i]["removed"],
                            "insert": variants[i]["insert"],
                            "text": variants[i]["text"],
                            "embedding": vvecs_list[i],
                        }
                        for i in range(len(variants))
                    ],
                },
            )

            Q = np.array(vvecs_list, dtype=np.float32)

            for qi in range(Q.shape[0]):
                qmeta = variants[qi]
                qtext = qmeta["text"]
                qkind = qmeta["kind"]
                qid = qmeta["qid"]
                removed_items = qmeta["removed"]
                ins = qmeta["insert"]

                if removed_items:
                    removed_lower_csv = ",".join([it["word_lower"] for it in removed_items])
                    removed_orig_csv = ",".join([it["word"] for it in removed_items])
                    removed_start_csv = ",".join([str(it["start"]) for it in removed_items])
                    removed_end_csv = ",".join([str(it["end"]) for it in removed_items])
                else:
                    removed_lower_csv = removed_orig_csv = removed_start_csv = removed_end_csv = ""

                if ins:
                    ins_text = ins.get("inserted_text", "")
                    ins_req = str(ins.get("requested_after_word_num", ""))
                    ins_act = str(ins.get("actual_after_word_num", ""))
                    ins_wc = str(ins.get("word_count", ""))
                    ins_pos = str(ins.get("insert_char_pos", ""))
                    ins_app = str(bool(ins.get("appended_to_end", False)))
                else:
                    ins_text = ins_req = ins_act = ins_wc = ins_pos = ins_app = ""

                qvec = Q[qi]

                # LOCAL exact cosine (optional Matryoshka prefix)
                q_slice = qvec[:k_local] if USE_MATRYOSHKA else qvec
                qn = normalize_vector(q_slice)

                scores = Vn @ qn
                top_local_idx = topk_from_scores(scores, K)
                local_ids = [docs[j]["chunk_id"] for j in top_local_idx]
                local_scores = [float(scores[j]) for j in top_local_idx]
                local_has_base = base_id in local_ids
                local_rank = (local_ids.index(base_id) + 1) if local_has_base else None

                # AZURE ANN
                ann = azure_ann_topk(search_client, qvec)
                ann_ids = [r["chunk_id"] for r in ann]
                ann_scores = [r["azure_score"] for r in ann]
                ann_has_base = base_id in ann_ids
                ann_rank = (ann_ids.index(base_id) + 1) if ann_has_base else None

                overlap = len(set(ann_ids) & set(local_ids))
                recall = overlap / float(K)
                total_queries += 1

                with open(rows_report, "a", encoding="utf-8") as f:
                    f.write(
                        "\t".join(
                            [
                                ts, str(base_num), base_id, qkind,
                                removed_lower_csv, removed_orig_csv, removed_start_csv, removed_end_csv,
                                _tsv_safe(ins_text), ins_req, ins_act, ins_wc, ins_pos, ins_app,
                                str(eligible_words), str(requested_sets), str(max_unique_sets_possible),
                                str(bool(ann_has_base)), "" if ann_rank is None else str(ann_rank),
                                str(bool(local_has_base)), "" if local_rank is None else str(local_rank),
                                f"{overlap}/{K}", f"{recall:.4f}",
                                _tsv_safe("|".join(ann_ids)), _tsv_safe("|".join(local_ids)),
                            ]
                        ) + "\n"
                    )

                entry = []
                entry.append(f"BASE {base_num}/{MAX_BASE_CHUNKS_TO_PROBE}  base_chunk_id: {base_id}\n")
                entry.append(f"  query_kind: {qkind}\n")
                entry.append(f"  query_id: {qid}\n")
                entry.append(f"  STORE_TEXT_MODE: {STORE_TEXT_MODE}  head_chars={STORE_TEXT_HEAD_CHARS}\n")
                if mode == "remove":
                    entry.append(f"  remove_meta: eligible_words={eligible_words} requested_sets={requested_sets} max_unique_sets_possible={max_unique_sets_possible} n_words_remove={N_WORDS_REMOVE}\n")
                if qkind == "REMOVED":
                    entry.append(
                        f"  removed_words_lower: {removed_lower_csv}  removed_words: {removed_orig_csv} "
                        f"(starts={removed_start_csv}, ends={removed_end_csv})\n"
                    )
                if qkind == "INSERTED":
                    entry.append(
                        f"  insert_text: {repr(ins_text)} after_word_requested={ins_req} after_word_actual={ins_act} "
                        f"word_count={ins_wc} insert_char_pos={ins_pos} appended_to_end={ins_app}\n"
                    )
                entry.append(f"  query_text_hash: {text_hash(qtext)}  query_vector_hash: {vec_hash(qvec)}\n")
                entry.append(f"  ANN_has_base: {ann_has_base} (rank={ann_rank}) | LOCAL_has_base: {local_has_base} (rank={local_rank})\n")
                entry.append(f"  Overlap@{K}: {overlap}/{K}  Recall@{K}: {recall:.2f}\n")
                entry.append(f"  Base preview(full): {preview_text(base_text)}\n")
                entry.append(f"  Query preview(full): {preview_text(qtext)}\n\n")

                entry.append("--- AZURE ANN TOP-K (contentStored) ---\n")
                for rnk, (cid, sc) in enumerate(zip(ann_ids, ann_scores), 1):
                    stored = next((x.get("contentStored", "") for x in ann if x["chunk_id"] == cid), "")
                    entry.append(f"  [{rnk}] {cid} | azure_score={sc:.6f} | stored: {preview_text(stored)}\n")

                entry.append("\n--- LOCAL EXACT COSINE TOP-K (full text previews) ---\n")
                for rnk, (cid, sc) in enumerate(zip(local_ids, local_scores), 1):
                    ch_text = docs_by_id.get(cid, {}).get("content_full", "")
                    entry.append(f"  [{rnk}] {cid} | local_cos={sc:.6f} | full: {preview_text(ch_text)}\n")

                entry.append("\n" + "-" * 110 + "\n\n")

                with open(full_report, "a", encoding="utf-8") as f:
                    f.writelines(entry)

                if overlap < K:
                    mismatch_errors += 1
                    with open(err_mismatch_report, "a", encoding="utf-8") as f:
                        f.writelines(entry)

                if not ann_has_base:
                    base_miss_errors += 1
                    with open(err_base_miss_report, "a", encoding="utf-8") as f:
                        f.writelines(entry)

            print(
                f"[used_base {used_base_chunks}/{MAX_BASE_CHUNKS_TO_PROBE}] total_queries={total_queries} "
                f"mismatch={mismatch_errors} base_miss={base_miss_errors} skipped={skipped_base_chunks}",
                end="\r",
            )

    finally:
        jsonl_fp.close()

    print("\n\n✅ DONE.")
    print(f"Run folder: {run_dir.resolve()}")
    print(f"Used base chunks: {used_base_chunks}")
    print(f"Skipped chunks: {skipped_base_chunks} (see {skipped_chunks_report.name})")
    print(f"Total queries: {total_queries}")
    print(f"Mismatch errors: {mismatch_errors}")
    print(f"ANN base-miss errors: {base_miss_errors}")


if __name__ == "__main__":
    main()
