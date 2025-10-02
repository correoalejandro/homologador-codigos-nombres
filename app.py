#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app for homologation candidate search (embeddings + cosine similarity).

Usage:
    streamlit run app.py

Environment:
    - Requires a .env with OPENAI_API_KEY, optional OPENAI_BASE_URL
    - Python deps: streamlit, pandas, numpy, python-dotenv, openai, pyarrow, tqdm

Main features:
    1) Single query search: user types a description/name to homologate, gets Candidatos a comparar (máx. 5) candidates from the catalog.
    2) Batch homologation: user uploads a CSV of items (with a text column) and gets Candidatos a comparar (máx. 5) candidates per row.
    3) Catalog loading: CSV/Parquet with columns for code + description; can have precomputed embeddings or not.
    4) Export: download CSVs with candidate results.
"""

import os
import io
import json
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path


# encrypt
import re
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
KEYLEN   = 32

from io import BytesIO


# ---- Optional OpenAI SDK import with friendly error ----
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


import os, streamlit as st
from dotenv import load_dotenv, find_dotenv


# --- Logo institucional ---
from PIL import Image


import base64


def get_secret(name, default=None):
    """
    Load secrets in priority:
    1. Streamlit Cloud (st.secrets)
    2. Environment variables (os.getenv)
    3. Optional default value
    """
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

# === App secrets ===
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
USERNAMES = [u.strip() for u in (get_secret("USERNAMES","") or "").split(",") if u.strip()]
PASSWORDS = [p.strip() for p in (get_secret("PASSWORDS","") or "").split(",") if p.strip()]


st.set_page_config(page_title="Homologador de Estructura de desglose de Trabajo: Códigos y Nombres", page_icon="🧭", layout="wide")

# --- Logo institucional --

logo_path = "Logotipo-ESCUELA-sin-VM.width-380_R7iE0XU.png"
with open(logo_path, "rb") as f:
    logo_bytes = f.read()
logo_b64 = base64.b64encode(logo_bytes).decode()

st.markdown(
    f"""
    <div style="background-color:white; padding:10px; text-align:center;">
        <img src="data:image/png;base64,{logo_b64}" width="280">
        <p style="font-size:14px; margin-top:5px;">
            Escuela Colombiana de Ingeniería Julio Garavito
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Read credentials from env ----
usernames = [u.strip() for u in os.getenv("USERNAMES", "").split(",") if u.strip()]
passwords = [p.strip() for p in os.getenv("PASSWORDS", "").split(",") if p.strip()]

if len(usernames) != len(passwords):
    st.error("Mismatch: USERNAMES and PASSWORDS count differ")
    st.stop()

CREDS = dict(zip(usernames, passwords))

# ---- Simple session-based login ----
def do_logout():
    for k in ("auth_user", "is_auth"):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def guarded_login():
    # already logged in?
    if st.session_state.get("is_auth"):
        st.sidebar.success(f"Logged in as {st.session_state['auth_user']}")
        if st.sidebar.button("Log out"):
            do_logout()
        return True

    # login form
    with st.sidebar.expander("🔒 Login", expanded=True):
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        ok = st.button("Sign in")

    if ok:
        if u in CREDS and CREDS[u] == p:
            st.session_state["is_auth"] = True
            st.session_state["auth_user"] = u
            st.session_state["auth_pass"] = p   # <- usa un key distinto al del widget
            st.rerun()


        else:
            st.error("Credenciales inválidas")
            st.stop()

    st.info("Please log in to continue.")
    st.stop()

# ---- Call login guard before showing app ----
if not guarded_login():
    st.stop()

# ---- Private content of your app ----
st.title("Homologador de Estructura de desglose de Trabajo: Códigos y Nombres")
st.write("✅ Has ingresado.")

# use your OpenAI key safely
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    st.caption("OPENAI_API_KEY loaded from env (hidden).")
else:
    st.warning("No OPENAI_API_KEY found in env.")

# ==================== Utilities ====================


## data password
_hex_re = re.compile(r"^[0-9a-fA-F]+$")
def _derive_key_from_input(input_str: str, salt: bytes) -> bytes:
    s = (input_str or "").strip()
    if _hex_re.fullmatch(s):
        pw_bytes = bytes.fromhex(s)
    else:
        pw_bytes = s.encode("utf-8")
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(pw_bytes)

def load_encrypted_parquet(path_enc: str, password: str) -> pd.DataFrame:
    blob = Path(path_enc).read_bytes()
    header_len = int.from_bytes(blob[:4], "big")
    header = json.loads(blob[4:4+header_len].decode("utf-8"))
    data_ct = blob[4+header_len:]

    data_nonce = bytes.fromhex(header["nonce_data"])
    envelopes = header["envelopes"]

    master_key = None
    # probar todos los sobres
    for env in envelopes:
        try:
            salt = bytes.fromhex(env["salt"])
            nonce = bytes.fromhex(env["nonce"])
            ct = bytes.fromhex(env["ct"])
            k = _derive_key_from_input(password, salt)
            mk = AESGCM(k).decrypt(nonce, ct, None)
            master_key = mk
            break
        except Exception:
            continue

    if master_key is None:
        raise ValueError("La contraseña no corresponde a ningún sobre válido.")

    # descifrar los datos con la master_key
    data = AESGCM(master_key).decrypt(data_nonce, data_ct, None)
    return pd.read_parquet(BytesIO(data))
# ---------------------------
# carga del dataset
# ---------------------------

# usa la contraseña que puso el usuario al loguearse
password = st.session_state.get("auth_pass")
if not password:
    st.warning("Inicia sesión para usar la contraseña como llave del .enc.")
    st.stop()

cat_df = load_encrypted_parquet("data/base_insumos_2.parquet.enc", password)


## data encryption


def _lazy_imports_ok() -> bool:
    """Return True if optional imports are available, otherwise False.

    This is used to give user-friendly messages when the OpenAI SDK is missing.
    """
    return _OPENAI_OK


def load_env() -> None:
    """Load environment variables from a local .env file.

    Notes
    -----
    Looks for OPENAI_API_KEY and optional OPENAI_BASE_URL.
    """
    load_dotenv()  # idempotent


def get_openai_client() -> Optional[OpenAI]:
    """Return an OpenAI client or None if unavailable.

    Returns
    -------
    Optional[OpenAI]
        Instantiated client if the SDK and API key are available; otherwise None.

    Raises
    ------
    RuntimeError
        If the OpenAI SDK is present but OPENAI_API_KEY is missing.
    """
    if not _lazy_imports_ok():
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está definido (archivo .env o variables de entorno).")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def read_table(path_or_buf, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame.

    Parameters
    ----------
    path_or_buf : str or buffer
        Path to the file or file-like buffer.
    sheet : str, optional
        Unused for CSV/Parquet. Present for future Excel support.

    Returns
    -------
    pandas.DataFrame
    """
    if hasattr(path_or_buf, "read"):
        # It's a buffer from st.file_uploader
        name = getattr(path_or_buf, "name", "uploaded")
        if name.lower().endswith(".parquet"):
            return pd.read_parquet(path_or_buf)
        else:
            # Auto-separator attempts
            try:
                return pd.read_csv(path_or_buf)
            except Exception:
                path_or_buf.seek(0)
                for sep in [";", "|", "\t"]:
                    try:
                        return pd.read_csv(path_or_buf, sep=sep)
                    except Exception:
                        path_or_buf.seek(0)
                raise
    else:
        # It's a path
        p = str(path_or_buf)
        if p.lower().endswith(".parquet"):
            return pd.read_parquet(p)
        else:
            try:
                return pd.read_csv(p)
            except Exception:
                for sep in [";", "|", "\t"]:
                    try:
                        return pd.read_csv(p, sep=sep)
                    except Exception:
                        continue
                raise

@st.cache_data(show_spinner=False)
def load_default_catalog(path: str) -> pd.DataFrame:
    # Usa tu lector unificado (CSV/Parquet)
    return read_table(path)

def ensure_list(x):
    """Return x as a list; wrap scalars.

    Parameters
    ----------
    x : Any

    Returns
    -------
    list
    """
    if isinstance(x, list):
        return x
    return [x]


def clean_text_series(s: pd.Series) -> pd.Series:
    """Basic text cleaning: cast to string, strip, and replace NaN.

    Parameters
    ----------
    s : pandas.Series

    Returns
    -------
    pandas.Series
        Cleaned series.
    """
    return s.astype(str).fillna("").str.strip()


def batched(iterable, n):
    """Yield successive n-sized chunks.

    Examples
    --------
    >>> list(batched([1,2,3,4,5], 2))
    [[1,2], [3,4], [5]]
    """
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Normalize rows to L2 norm = 1, avoiding division by zero.

    Parameters
    ----------
    mat : numpy.ndarray
        Array of shape (n, d).

    Returns
    -------
    numpy.ndarray
        Row-normalized array of shape (n, d).
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def topk_cosine(a: np.ndarray, b: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Candidatos a comparar (máx. 5) neighbors in b for each row of a using cosine similarity.

    Parameters
    ----------
    a : numpy.ndarray
        Query matrix of shape (n, d).
    b : numpy.ndarray
        Index matrix of shape (m, d).
    k : int
        Number of neighbors per query.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Indices (n, k) of top neighbors in b and their cosine scores (n, k).
    """
    a_n = normalize_rows(a)
    b_n = normalize_rows(b)
    sims = a_n @ b_n.T
    k_eff = min(k, max(1, sims.shape[1]-1))
    idx_part = np.argpartition(-sims, kth=k_eff, axis=1)[:, :k]
    row_idx = np.arange(sims.shape[0])[:, None]
    row_scores = sims[row_idx, idx_part]
    order = np.argsort(-row_scores, axis=1)
    top_idx = idx_part[row_idx, order]
    top_scores = row_scores[row_idx, order]
    return top_idx, top_scores


def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int = 128, max_retries: int = 5) -> List[List[float]]:
    """Compute embeddings from a list of texts using the OpenAI embeddings API.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI client.
    model : str
        Embedding model name (e.g., 'text-embedding-3-small').
    texts : list of str
        Input texts.
    batch_size : int, optional
        Batch size for API calls, by default 128.
    max_retries : int, optional
        Max retry attempts with exponential backoff, by default 5.

    Returns
    -------
    list of list of float
        One embedding vector per input text, preserving order.
    """
    out: List[List[float]] = []
    for batch in batched(texts, batch_size):
        retries = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [getattr(d, "embedding", getattr(d, "embeddings", None)) for d in resp.data]
                out.extend(vecs)
                break
            except Exception as e:
                if retries >= max_retries:
                    raise
                sleep_s = (2 ** retries) + 0.05 * (retries + 1)
                time.sleep(sleep_s)
                retries += 1
    return out


def ensure_embeddings(df: pd.DataFrame, text_col: str, model: str) -> pd.DataFrame:
    """Ensure a DataFrame has an 'embedding' column; compute it if missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog DataFrame.
    text_col : str
        Name of the text column to embed.
    model : str
        Embedding model to use if embeddings are missing.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an 'embedding' column of lists of floats.

    Notes
    -----
    Requires OPENAI_API_KEY in environment if embeddings need to be computed.
    """
    if "embedding" in df.columns and pd.notnull(df["embedding"]).any():
        # Normalize JSON strings into lists if read from CSV
        if isinstance(df["embedding"].iloc[0], str):
            df = df.copy()
            df["embedding"] = df["embedding"].apply(lambda x: json.loads(x))
        return df

    client = get_openai_client()
    texts = clean_text_series(df[text_col]).tolist()
    vecs = embed_texts(client, model, texts)
    out = df.copy()
    out["embedding"] = vecs
    return out


def candidates_from_catalog(
    catalog_df: pd.DataFrame,
    code_col: str,
    text_col: str,
    k: int = 5,
    keep_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute all-vs-all Candidatos a comparar (máx. 5) candidates inside a catalog.

    Parameters
    ----------
    catalog_df : pandas.DataFrame
        DataFrame with code, text, and 'embedding' columns.
    code_col : str
        Column name for the item code/ID.
    text_col : str
        Column name for the item description/text.
    k : int, optional
        Number of candidates per item, by default 5.
    keep_cols : list of str, optional
        Additional catalog columns to duplicate as *_origen/*_candidato.

    Returns
    -------
    pandas.DataFrame
        Candidate pairs with code/text, optional metadata, cosine scores, and rank.
    """
    keep_cols = keep_cols or []
    emb = np.vstack(catalog_df["embedding"].values).astype(np.float32)
    idx, scores = topk_cosine(emb, emb, k + 1)  # +1 to allow self, we'll remove it

    codes = catalog_df[code_col].astype(str).tolist()
    texts = catalog_df[text_col].astype(str).tolist()

    rows = []
    for i, (nbr_idx, nbr_scores) in enumerate(zip(idx, scores)):
        count = 0
        for col_idx, sc in zip(nbr_idx, nbr_scores):
            if col_idx == i:
                continue
            row = {
                "codigo_origen": codes[i],
                "descripcion_origen": texts[i],
                "codigo_candidato": codes[col_idx],
                "descripcion_candidato": texts[col_idx],
                "score": float(sc),
                "rank": count + 1,
            }
            for c in keep_cols:
                if c in catalog_df.columns:
                    row[f"{c}_origen"] = catalog_df.iloc[i][c]
                    row[f"{c}_candidato"] = catalog_df.iloc[col_idx][c]
            rows.append(row)
            count += 1
            if count >= k:
                break
    return pd.DataFrame(rows)


def candidates_for_queries(
    catalog_df: pd.DataFrame,
    queries: List[str],
    code_col: str,
    text_col: str,
    model: str,
    k: int = 5,
    keep_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute Candidatos a comparar (máx. 5) candidates in the catalog for external query texts.

    Parameters
    ----------
    catalog_df : pandas.DataFrame
        Catalog with code, text, and 'embedding' columns.
    queries : list of str
        Free-text items to homologate.
    code_col : str
        Column name for the catalog code/ID.
    text_col : str
        Column name for the catalog text/description.
    model : str
        Embedding model name for the query texts.
    k : int, optional
        Number of candidates per query, by default 5.
    keep_cols : list of str, optional
        Extra catalog columns to include for context.

    Returns
    -------
    pandas.DataFrame
        Candidate table with one block per query input, sorted by cosine score.
    """
    keep_cols = keep_cols or []
    # Ensure catalog embeddings ready
    cat = ensure_embeddings(catalog_df, text_col=text_col, model=model)
    emb_cat = np.vstack(cat["embedding"].values).astype(np.float32)

    # Embed queries
    client = get_openai_client()
    q_vecs = embed_texts(client, model, [q or "" for q in queries])
    q_mat = np.vstack(q_vecs).astype(np.float32)

    # Similarities
    idx, scores = topk_cosine(q_mat, emb_cat, k)

    codes = cat[code_col].astype(str).tolist()
    texts = cat[text_col].astype(str).tolist()

    rows = []
    for qi, (nbr_idx, nbr_scores) in enumerate(zip(idx, scores)):
        q = queries[qi]
        for rank_i, (col_idx, sc) in enumerate(zip(nbr_idx, nbr_scores), start=1):
            row = {
                "query": q,
                "codigo_candidato": codes[col_idx],
                "descripcion_candidato": texts[col_idx],
                "score": float(sc),
                "rank": rank_i,
            }
            for c in keep_cols:
                if c in cat.columns:
                    row[f"{c}_candidato"] = cat.iloc[col_idx][c]
            rows.append(row)
    return pd.DataFrame(rows)


# ==================== Streamlit UI ====================

st.caption("Buscador semántico que sugiere equivalencias del Diccionario de la EDT ARPRO para estandarizar proyectos.")

with st.sidebar:
    st.header("diccionario base")
    cat_file = st.file_uploader("Sube el Diccionario maestro de la EDT (Excel o Parquet)", type=["csv", "parquet"])
    default_text_col = st.text_input("Columna de texto (diccionario)", value="Descripción_prefijada")
    default_code_col = st.text_input("Columna de código (diccionario)", value="Código")
    keep_cols_inp = st.text_input("Otras columnas a mostrar (coma)", value="Unidad,Categoría")
    keep_cols = [c.strip() for c in keep_cols_inp.split(",") if c.strip()]
    model = st.text_input("Modelo de IA para vectores (embeddings)", value="text-embedding-3-small")
    k = st.number_input("Candidatos a comparar (máx. 5)", min_value=1, max_value=20, value=5, step=1)
    use_default_catalog = st.toggle("Usar Diccionario maestro de la EDT incluido (ejemplo de prueba)", value=True)

    st.markdown("---")
    st.caption("Este módulo usa inteligencia artificial para calcular similitud semántica. Los datos de referencia provienen del Diccionario maestro de la EDT ARPRO.")


# Load catalog
cat_df = None
default_parquet_path = Path("data/base_insumos_2.parquet")

try:
    if use_default_catalog and default_parquet_path.exists():
        # Carga desde repo (cacheado)
        cat_df = load_default_catalog(str(default_parquet_path))
        st.caption(f"diccionario cargado desde {default_parquet_path} ({len(cat_df):,} filas).")
    elif cat_file is not None:
        # Carga desde archivo subido (CSV o Parquet)
        cat_df = read_table(cat_file)
        st.caption(f"diccionario cargado desde archivo subido: {getattr(cat_file, 'name', 'upload')} ({len(cat_df):,} filas).")
    else:
        st.info("Sube un diccionario en la barra lateral o activa 'Usar diccionario incluido (data/catalogo.parquet)'.")
except Exception as e:
    st.error(f"No se pudo leer el diccionario: {e}")
    cat_df = None

# Normaliza posibles embeddings serializados como JSON en CSV
if cat_df is not None and "embedding" in cat_df.columns and isinstance(cat_df["embedding"].iloc[0], str):
    try:
        cat_df["embedding"] = cat_df["embedding"].apply(lambda x: json.loads(x))
    except Exception:
        pass


tab1, tab2 = st.tabs(["Búsqueda individual", "Homologación por grupo(excel)"])

with tab1:
    st.subheader("1) Buscar candidatos para un texto")
    st.write("Escribe el nombre o descripción que quieras homologar y el sistema te mostrará las 5 alternativas más cercanas del Diccionario maestro de la EDT.")

    q_text = st.text_input("Texto a homologar", value="")
    run_single = st.button("Buscar candidatos")

    if run_single:
        if cat_df is None:
            st.warning("Sube primero el diccionario de la EDT en la barra lateral.")
        elif default_text_col not in cat_df.columns or default_code_col not in cat_df.columns:
            st.error("Revisa los nombres de columnas (texto y código) configurados en la barra lateral.")
        else:
            try:
                load_env()
                # Ensure catalog embeddings (compute only if absent)
                cat_df2 = ensure_embeddings(cat_df, text_col=default_text_col, model=model)
                res = candidates_for_queries(
                    catalog_df=cat_df2,
                    queries=[q_text],
                    code_col=default_code_col,
                    text_col=default_text_col,
                    model=model,
                    k=k,
                    keep_cols=keep_cols,
                )
                st.success("Candidatos generados.")
                st.dataframe(res, use_container_width=True)
                csv = res.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV", data=csv, file_name="candidatos_query.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error al generar candidatos: {e}")

with tab2:
    st.subheader("2) Homologar un grupo desde CSV")
    st.write("Sube un CSV con una columna de texto (por ejemplo, 'descripcion'). Se calcularán candidatos para cada fila.")
    batch_file = st.file_uploader("CSV con items a homologar", type=["csv"], key="batch")
    batch_text_col = st.text_input("Columna de texto (grupo)", value="Descripción", key="batch_text")

    run_batch = st.button("Procesar grupo")

    if run_batch:
        if cat_df is None:
            st.warning("Sube primero el diccionario en la barra lateral.")
        else:
            try:
                load_env()
                # Ensure catalog embeddings (compute if missing)
                cat_df2 = ensure_embeddings(cat_df, text_col=default_text_col, model=model)
                # Read batch items
                items_df = read_table(batch_file) if batch_file is not None else None
                if items_df is None or batch_text_col not in items_df.columns:
                    st.error("Revisa el CSV del grupo y el nombre de la columna de texto.")
                else:
                    queries = clean_text_series(items_df[batch_text_col]).tolist()
                    res = candidates_for_queries(
                        catalog_df=cat_df2,
                        queries=queries,
                        code_col=default_code_col,
                        text_col=default_text_col,
                        model=model,
                        k=k,
                        keep_cols=keep_cols,
                    )
                    # Attach row index to map results back
                    res.insert(0, "row_id", np.repeat(np.arange(len(queries)), k))
                    st.success("Candidatos por grupo generados.")
                    st.dataframe(res, use_container_width=True)
                    csv = res.to_csv(index=False).encode("utf-8")
                    st.download_button("Descargar CSV", data=csv, file_name="candidatos_grupo.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error al procesar el grupo: {e}")

st.markdown("---")
st.caption("Esta herramienta calcula similitud semántica (coseno) para sugerir equivalencias en la EDT ARPRO. Un puntaje más alto indica mayor similitud.")
