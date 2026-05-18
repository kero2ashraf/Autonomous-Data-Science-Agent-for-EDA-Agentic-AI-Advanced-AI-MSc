"""
Autonomous Multi-Agent EDA System  ·  v5  —  Per-Agent Model Assignment
========================================================================

Changes vs v4:
  ① Each agent is assigned its optimal model (see AGENT_MODELS dict)
  ② Orchestrator → DeepSeek V3 (powerful synthesis)
  ③ Quality      → Gemma 3 27B (thorough data inspection)
  ④ Stats        → DeepSeek R1  (reasoning-heavy numeric analysis)
  ⑤ Correlation  → DeepSeek R1  (math/reasoning for relationships)
  ⑥ Visualization→ Mistral 7B   (creative chart recommendations)
  ⑦ Sidebar shows per-agent model assignment table
  ⑧ PipelineState stores per-agent model used (for display)
  ⑨ All v4 bugfixes retained

Run:
    pip install streamlit pandas numpy scikit-learn requests plotly \
                python-dotenv openpyxl
    streamlit run eda_multiagent_v5.py
"""

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────────────
import hashlib
import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from urllib3.util.retry import Retry

# ============================================================
# ❶  CONSTANTS & CONFIG
# ============================================================

APP_TITLE           = "Autonomous Multi-Agent EDA System v5"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_URL  = "https://openrouter.ai/api/v1/key"
MAX_REACT_CYCLES    = 3
MIN_INSIGHT_CHARS   = 80

# Backend API key — never shown in UI
_BACKEND_API_KEY = "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"

# ── All available models ───────────────────────────────────────────────────
MODEL_CHOICES = {
    "Auto Router — recommended":  "openrouter/auto",
    "Llama 3.1 8B — free":        "meta-llama/llama-3.1-8b-instruct:free",
    "Mistral 7B — free":          "mistralai/mistral-7b-instruct:free",
    "Gemma 3 27B — free":         "google/gemma-3-27b-it:free",
    "DeepSeek R1 — free":         "deepseek/deepseek-r1:free",
    "DeepSeek V3 — free":         "deepseek/deepseek-chat-v3-0324:free",
}

FALLBACK_MODELS = [
    "openrouter/auto",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-27b-it:free",
]

# ── Per-agent model assignment ─────────────────────────────────────────────
# Each agent gets the model best suited to its task.
# Fallback chain is used if the assigned model is unavailable.
AGENT_MODELS: Dict[str, str] = {
    # Data Quality: Gemma 3 27B — large context, careful attention to detail
    "quality":      "google/gemma-3-27b-it:free",

    # Statistical Analysis: DeepSeek R1 — strong mathematical reasoning
    "stats":        "deepseek/deepseek-r1:free",

    # Correlation Analysis: DeepSeek R1 — excels at numeric relationships & logic
    "corr":         "deepseek/deepseek-r1:free",

    # Visualization: Mistral 7B — fast, creative chart recommendations
    "viz":          "mistralai/mistral-7b-instruct:free",

    # Orchestrator: DeepSeek V3 — best synthesis & report writing
    "orchestrator": "deepseek/deepseek-chat-v3-0324:free",

    # Evaluation baseline: Llama 3.1 8B — lightweight, unbiased baseline
    "baseline":     "meta-llama/llama-3.1-8b-instruct:free",
}

# Human-readable labels for sidebar display
AGENT_MODEL_LABELS: Dict[str, str] = {
    "quality":      "Gemma 3 27B",
    "stats":        "DeepSeek R1",
    "corr":         "DeepSeek R1",
    "viz":          "Mistral 7B",
    "orchestrator": "DeepSeek V3",
    "baseline":     "Llama 3.1 8B",
}

AGENTS: Dict[str, Dict] = {
    "quality": {
        "name": "Data Quality Agent", "icon": "🔍", "color": "#ef4444",
        "system": (
            "You are a Data Quality expert AI agent in a ReAct loop. "
            "You receive pre-computed statistics, a tool plan, and an optional prior error. "
            "Analyse: missing values, duplicates, constant columns, type mismatches. "
            "Give actionable cleaning steps. Use markdown bullets. Be concise."
        ),
    },
    "stats": {
        "name": "Statistical Agent", "icon": "📊", "color": "#4f46e5",
        "system": (
            "You are a Statistical Analysis expert AI agent in a ReAct loop. "
            "Interpret distributions, skewness, outliers, anomalies. "
            "Explain implications for downstream modelling. Use markdown bullets."
        ),
    },
    "corr": {
        "name": "Correlation Agent", "icon": "🔗", "color": "#0ea5e9",
        "system": (
            "You are a Correlation expert AI agent in a ReAct loop. "
            "Identify strong relationships, multicollinearity, group patterns. "
            "Name specific columns. Suggest feature engineering ideas. Use markdown bullets."
        ),
    },
    "viz": {
        "name": "Visualization Agent", "icon": "🎨", "color": "#10b981",
        "system": (
            "You are a Visualization expert AI agent in a ReAct loop. "
            "Recommend the 5 most impactful charts for this exact dataset. "
            "For each: chart type, exact columns, insight revealed. "
            "Use a markdown numbered list."
        ),
    },
    "orchestrator": {
        "name": "Orchestrator Agent", "icon": "🧠", "color": "#f59e0b",
        "system": (
            "You are a senior Data Science Orchestrator AI. "
            "Synthesise findings from 4 specialised agents into one professional EDA report. "
            "Sections: Executive Summary · Key Findings (per agent) · "
            "Top 5 Recommendations · Next Steps. "
            "Use markdown headers and bold highlights. Be authoritative and non-repetitive."
        ),
    },
}

EVAL_DATASETS = ["Titanic", "Iris", "Wine", "Breast Cancer", "Diamonds"]


# ============================================================
# ❷  PYDANTIC-STYLE SCHEMAS
# ============================================================

@dataclass
class ToolInput:
    tool_name: str
    params:    Dict[str, Any]
    agent_id:  str
    attempt:   int = 1

    _REQUIRED: ClassVar[Dict[str, List[str]]] = {
        "describe_column":      ["column"],
        "compute_correlation":  ["columns"],
        "compute_distribution": ["column"],
        "detect_outliers":      ["column"],
        "group_stats":          ["groupby", "target"],
        "missing_summary":      [],
        "dtype_summary":        [],
        "sql_missing_agg":      [],
        "sklearn_describe":     ["columns"],
    }

    def validate(self) -> Tuple[bool, str]:
        required = self._REQUIRED
        if self.tool_name not in required:
            return False, f"❌ Unknown tool '{self.tool_name}'. Valid tools: {list(required.keys())}"
        missing = [k for k in required[self.tool_name] if k not in self.params]
        if missing:
            return False, f"❌ Tool '{self.tool_name}' missing params: {missing}"
        return True, "✅ valid"


@dataclass
class ToolOutput:
    tool_name:  str
    success:    bool
    result:     Any
    error:      Optional[str] = None
    cached:     bool          = False
    elapsed_ms: float         = 0.0


@dataclass
class CodeBlock:
    language:   str
    code:       str
    purpose:    str
    executed:   bool  = False
    output:     str   = ""
    error:      str   = ""

    def execute_python(self, df: pd.DataFrame) -> "CodeBlock":
        local_ns: Dict[str, Any] = {"df": df, "pd": pd, "np": np}
        try:
            exec(self.code, {}, local_ns)  # noqa: S102
            self.output   = str(local_ns.get("_result", "executed ok"))
            self.executed = True
        except Exception as exc:
            self.error    = f"{type(exc).__name__}: {exc}"
            self.executed = False
        return self


@dataclass
class AgentStep:
    step_type:   str
    content:     str
    tool_input:  Optional[ToolInput]  = None
    tool_output: Optional[ToolOutput] = None
    code_block:  Optional[CodeBlock]  = None
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "step_type": self.step_type,
            "content":   self.content,
            "timestamp": self.timestamp,
        }


# ============================================================
# ❸  ERROR-CORRECTION PARSER
# ============================================================

class ErrorCorrectionParser:
    BAD_PATTERNS = [
        r"^(sorry|i cannot|i'm sorry|i don't|as an ai)",
        r"(unable to|cannot assist|not able to provide)",
        r"^\s*$",
    ]
    GOOD_MARKERS = ["**", "-", "#", "•", "\n"]

    def parse(self, text: str, agent_key: str) -> Tuple[bool, str]:
        t = text.strip().lower()

        if len(text.strip()) < MIN_INSIGHT_CHARS:
            return False, (
                f"Response was only {len(text.strip())} chars "
                f"(minimum {MIN_INSIGHT_CHARS}). Expand your analysis."
            )

        for pat in self.BAD_PATTERNS:
            if re.search(pat, t[:120], re.I):
                return False, (
                    "Response appeared to be a refusal or empty. "
                    "You MUST provide analysis based on the pre-computed data."
                )

        has_structure = any(m in text for m in self.GOOD_MARKERS)
        if not has_structure:
            return False, (
                "Response lacked markdown structure (no bullets, headers, or bold). "
                "Reformat with bullet points or headers."
            )

        return True, ""


# ============================================================
# ❹  MEMORY STORE
# ============================================================

class MemoryStore:
    def __init__(self) -> None:
        self._store:  Dict[str, Any] = {}
        self._hits:   int = 0
        self._misses: int = 0

    def get(self, ds_sig: str, name: str, params: Any = None) -> Optional[Any]:
        k = self._key(ds_sig, name, params)
        if k in self._store:
            self._hits += 1
            return self._store[k]
        self._misses += 1
        return None

    def put(self, ds_sig: str, name: str, value: Any, params: Any = None) -> None:
        self._store[self._key(ds_sig, name, params)] = value

    def clear(self) -> None:
        self._store.clear()
        self._hits = self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "keys": len(self._store)}

    @staticmethod
    def _key(ds_sig: str, name: str, params: Any) -> str:
        raw = f"{ds_sig}::{name}::{json.dumps(params, sort_keys=True, default=str)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_or_compute(self, ds_sig: str, name: str,
                       fn: Callable, params: Any = None) -> Any:
        v = self.get(ds_sig, name, params)
        if v is not None:
            return v
        v = fn()
        self.put(ds_sig, name, v, params)
        return v


def _get_mem() -> MemoryStore:
    if "memory_store" not in st.session_state:
        st.session_state["memory_store"] = MemoryStore()
    return st.session_state["memory_store"]


# ============================================================
# ❺  DYNAMIC TOOL-SET SCHEDULER
# ============================================================

class ToolScheduler:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df            = df
        self.n_rows        = len(df)
        self.numeric_cols  = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols      = df.select_dtypes(exclude=np.number).columns.tolist()
        self.low_card_cat  = [c for c in self.cat_cols if df[c].nunique() <= 20]
        self.high_card_cat = [c for c in self.cat_cols if df[c].nunique() >  20]
        self.large         = self.n_rows > 5_000
        self.very_large    = self.n_rows > 10_000
        self.wide_numeric  = len(self.numeric_cols) > 10

    def plan_for_quality(self) -> Dict[str, Any]:
        if self.very_large:
            lib      = "SQL aggregation"
            strategy = (
                "SELECT col, COUNT(*) - COUNT(col) AS missing_count, "
                "COUNT(DISTINCT col) AS unique_count FROM table GROUP BY col"
            )
            tools = ["sql_missing_agg", "dtype_summary"]
        else:
            lib      = "pandas"
            strategy = "df.isna().sum() + df.duplicated() + df.nunique()"
            tools    = ["missing_summary", "dtype_summary"]
        return {
            "library": lib, "strategy": strategy, "tools": tools,
            "note": (
                f"{self.n_rows:,} rows detected. "
                + ("SQL-style agg used for efficiency." if self.very_large
                   else "pandas used — dataset fits in memory.")
            ),
        }

    def plan_for_stats(self) -> Dict[str, Any]:
        if self.large and self.wide_numeric:
            lib      = "sklearn.preprocessing + pandas"
            strategy = (
                "StandardScaler().fit_transform(df[numeric_cols]) "
                "— normalise then describe; IQR outlier detection via np.percentile"
            )
        elif self.numeric_cols:
            lib      = "pandas + scipy.stats"
            strategy = (
                "df[numeric_cols].describe() + "
                "df[col].skew() + df[col].kurtosis() + IQR outlier count"
            )
        else:
            lib      = "pandas"
            strategy = "df[cat_cols].value_counts() — no numeric columns"
        tools = [
            {"tool": "describe_column", "column": c}
            for c in self.numeric_cols[:8]
        ]
        return {
            "library": lib, "strategy": strategy, "tools": tools,
            "note": (
                f"{len(self.numeric_cols)} numeric, {len(self.cat_cols)} categorical cols. "
                + ("Sampled to 5 k rows." if self.large else "Full scan.")
            ),
        }

    def plan_for_corr(self) -> Dict[str, Any]:
        if len(self.numeric_cols) >= 2:
            lib   = "pandas.DataFrame.corr (Pearson) + SQL GROUP BY"
            tools = [{"tool": "compute_correlation",
                      "columns": self.numeric_cols[:12]}]
        else:
            lib   = "SQL COUNT/GROUP BY  (no numeric cols for Pearson)"
            tools = []
        group_tools = [
            {
                "tool":      "group_stats",
                "groupby":   g,
                "target":    self.numeric_cols[0] if self.numeric_cols else "",
                "sql_equiv": (
                    f"SELECT {g}, AVG({self.numeric_cols[0] if self.numeric_cols else 'col'}) "
                    f"FROM table GROUP BY {g}"
                ),
            }
            for g in self.low_card_cat[:3]
            if self.numeric_cols
        ]
        return {
            "library": lib,
            "strategy": "Pearson r matrix + SQL-style GROUP BY means",
            "tools":   tools + group_tools,
            "note": (
                f"{len(self.high_card_cat)} high-cardinality cols skipped. "
                f"{len(self.low_card_cat)} low-cardinality cols used for grouping."
            ),
        }

    def plan_for_viz(self) -> Dict[str, Any]:
        charts: List[str] = []
        nc, cc = self.numeric_cols, self.cat_cols
        if nc:
            charts.append(f"plotly.histogram(df['{nc[0]}'], nbins=30)")
        if len(nc) >= 2:
            charts.append(f"plotly.scatter(df, x='{nc[0]}', y='{nc[1]}')")
            charts.append("plotly.imshow(df[numeric_cols].corr())  # heatmap")
        if self.low_card_cat and nc:
            charts.append(f"plotly.box(df, x='{self.low_card_cat[0]}', y='{nc[0]}')")
        if cc:
            charts.append(f"plotly.bar(df['{cc[0]}'].value_counts())")
        return {
            "library":  "plotly",
            "strategy": "interactive charts for variance + class separation",
            "tools":    charts,
            "note": "Prioritise charts that reveal the strongest signal.",
        }

    def summary(self) -> str:
        lib = "sklearn" if (self.large and self.wide_numeric) else \
              ("SQL" if self.very_large else "pandas/scipy")
        return (
            f"{self.n_rows:,} rows × {len(self.df.columns)} cols | "
            f"Numeric: {len(self.numeric_cols)} | Cat: {len(self.cat_cols)} | "
            f"Scheduler → {lib}"
        )


# ============================================================
# ❻  LANGGRAPH-STYLE STATE MACHINE
# ============================================================

class NodeState(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE    = auto()
    FAILED  = auto()
    SKIPPED = auto()


@dataclass
class PipelineState:
    ds_sig:         str
    ds_name:        str
    df:             Any
    model:          str          # global fallback model (from sidebar selector)
    enabled_agents: List[str]
    node_states:    Dict[str, NodeState]       = field(default_factory=dict)
    agent_findings: Dict[str, str]             = field(default_factory=dict)
    react_traces:   Dict[str, List[AgentStep]] = field(default_factory=dict)
    code_blocks:    Dict[str, List[CodeBlock]]  = field(default_factory=dict)
    retry_counts:   Dict[str, int]             = field(default_factory=dict)
    # NEW: track which model was actually used per agent
    agent_models_used: Dict[str, str]          = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "agents_done": 0, "llm_calls": 0, "errors": 0,
        "elapsed": 0.0, "memory_hits": 0, "memory_misses": 0,
        "retries": 0, "code_runs": 0, "code_errors": 0,
    })
    final_summary: str = ""
    eval_results:  Dict = field(default_factory=dict)


class StateGraph:
    def __init__(self, name: str) -> None:
        self.name    = name
        self._nodes: Dict[str, Callable]                       = {}
        self._edges: List[Tuple[str, str, Optional[Callable]]] = []
        self._entry: Optional[str]                             = None
        self._log:   List[str]                                 = []

    def add_node(self, name: str, fn: Callable) -> "StateGraph":
        self._nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str,
                 condition: Optional[Callable] = None) -> "StateGraph":
        self._edges.append((src, dst, condition))
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    def run(self, state: PipelineState) -> PipelineState:
        if not self._entry:
            raise RuntimeError("No entry node set.")
        cur = self._entry
        while cur:
            if cur not in self._nodes:
                break
            self._log.append(cur)
            state = self._nodes[cur](state)
            nxt   = None
            for src, dst, cond in self._edges:
                if src == cur and (cond is None or cond(state)):
                    nxt = dst
                    break
            cur = nxt
        return state

    @property
    def transitions(self) -> List[str]:
        return list(self._log)


# ============================================================
# OPENROUTER HELPERS
# ============================================================

def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )))
    return s


def _get_api_key() -> str:
    return _BACKEND_API_KEY.strip()


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8501",
        "X-Title":       APP_TITLE,
    }


def check_api_key() -> Tuple[bool, str]:
    key = _get_api_key()
    if not key:
        return False, "No API key configured."
    try:
        r = _session().get(OPENROUTER_KEY_URL, headers=_headers(), timeout=20)
        if r.status_code == 200:
            st.session_state.api_status = "ok"
            return True, "API key valid."
        msg = (r.json().get("error", {}).get("message", "") or r.text)[:200]
        return False, f"HTTP {r.status_code}: {msg}"
    except Exception as e:
        return False, str(e)


def _has_endpoint(model: str) -> bool:
    if model == "openrouter/auto":
        return True
    cache = st.session_state.setdefault("ep_cache", {})
    if model in cache:
        return bool(cache[model])
    if "/" not in model:
        cache[model] = False
        return False
    try:
        a, s = model.split("/", 1)
        r = _session().get(
            f"https://openrouter.ai/api/v1/models/{a}/{s}/endpoints",
            headers=_headers(), timeout=20)
        ok = r.status_code == 200 and \
             len(r.json().get("data", {}).get("endpoints", [])) > 0
        cache[model] = ok
        return ok
    except Exception:
        cache[model] = False
        return False


def _pick_model(preferred: str) -> str:
    """Try preferred model, then agent-appropriate fallbacks, then global fallbacks."""
    for m in [preferred] + FALLBACK_MODELS:
        if m and _has_endpoint(m):
            return m
    return "openrouter/auto"


def call_llm(system: str, user: str, model: str,
             temp: float = 0.15, max_tok: int = 400
             ) -> Tuple[bool, str, str]:
    if not _get_api_key():
        return False, "No API key configured.", ""
    m = _pick_model(model)
    try:
        r = _session().post(
            OPENROUTER_CHAT_URL, headers=_headers(), timeout=90,
            json={
                "model": m, "temperature": temp, "max_tokens": max_tok,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
        )
        if not r.ok:
            err = (r.json().get("error", {}).get("message", "") or r.text)[:200]
            st.session_state.api_status = "error"
            return False, f"HTTP {r.status_code}: {err}", m
        content = r.json()["choices"][0]["message"].get("content", "")
        st.session_state.api_status = "ok"
        st.session_state.setdefault("_llm_calls", 0)
        st.session_state["_llm_calls"] += 1
        return True, _clean(content), m
    except Exception as exc:
        st.session_state.api_status = "error"
        return False, str(exc), m


def test_connection(model: str) -> Tuple[bool, str]:
    ok, msg = check_api_key()
    if not ok:
        return False, msg
    ok, txt, used = call_llm("Test.", "Reply: ready", model, max_tok=10)
    return (True, f"✅ {used} — {txt[:60]}") if ok else (False, txt)


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean(t: Any) -> str:
    if not t:
        return ""
    t = str(t).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+$",    "",  t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}",     "\n\n", t)
    t = re.sub(r"^[ \t]{4,}", "",  t, flags=re.MULTILINE)
    t = re.sub(r"\n[ \t]*[-*][ \t]+", "\n- ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def _sig(df: pd.DataFrame) -> str:
    return hashlib.md5(
        f"{df.shape}|{list(df.columns)}|{df.head(20).to_csv(index=False)}"
        .encode()
    ).hexdigest()


def _num(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def _cat(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


# ============================================================
# MEMORY-CACHED COMPUTATIONS
# ============================================================

def _mem_compute(mem: MemoryStore, ds_sig: str, name: str, fn: Callable) -> Any:
    return mem.get_or_compute(ds_sig, name, fn)


def stat_quality(df: pd.DataFrame, mem: MemoryStore, ds_sig: str) -> str:
    def _c():
        miss     = df.isna().sum()
        miss_pct = (miss / max(len(df), 1) * 100).round(2)
        dup      = int(df.duplicated().sum())
        hi       = miss_pct[miss_pct >= 30].index.tolist()
        const    = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        lines    = [
            f"- {c}: dtype={df[c].dtype}, "
            f"missing={int(miss[c])} ({miss_pct[c]}%), "
            f"unique={int(df[c].nunique(dropna=True))}"
            for c in df.columns
        ]
        return (
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n"
            f"Duplicates: {dup:,}\n"
            f"Total missing: {int(miss.sum()):,}\n"
            f"High-missing (≥30%): {', '.join(hi) or 'None'}\n"
            f"Constant cols: {', '.join(const) or 'None'}\n\n"
            "Per-column:\n" + "\n".join(lines)
        )
    return _mem_compute(mem, ds_sig, "quality_stats", _c)


def stat_describe(df: pd.DataFrame, mem: MemoryStore, ds_sig: str) -> str:
    def _c():
        nums = _num(df)
        if not nums:
            return "No numeric columns."
        desc = df[nums].describe().round(3)
        rows = [
            "| col | count | mean | std | min | 25% | 50% | 75% | max |",
            "|-----|-------|------|-----|-----|-----|-----|-----|-----|",
        ]
        for col in desc.columns[:12]:
            r = desc[col]
            rows.append(
                f"| {col} | {r['count']:.0f} | {r['mean']:.3f} | "
                f"{r['std']:.3f} | {r['min']:.3f} | {r['25%']:.3f} | "
                f"{r['50%']:.3f} | {r['75%']:.3f} | {r['max']:.3f} |"
            )
        return "\n".join(rows)
    return _mem_compute(mem, ds_sig, "df_describe", _c)


def stat_stats(df: pd.DataFrame, mem: MemoryStore, ds_sig: str) -> str:
    def _c():
        nums = _num(df)
        cats = _cat(df)
        lines = []
        for col in nums:
            s = df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile([.25, .75])
            iqr    = q3 - q1
            out    = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            lines.append(
                f"- {col}: mean={s.mean():.3f}, median={s.median():.3f}, "
                f"std={s.std():.3f}, skew={s.skew():.3f}, "
                f"outliers={out} ({out / max(len(df), 1) * 100:.1f}%)"
            )
        cat_lines = []
        for col in cats[:6]:
            vc  = df[col].value_counts(dropna=False).head(5)
            top = ", ".join(f"{i}={int(v)}" for i, v in vc.items())
            cat_lines.append(f"- {col}: {top}")
        return (
            f"Numeric ({len(nums)}):\n" + ("\n".join(lines) or "None") +
            f"\n\nCategorical ({len(cats)}):\n" + ("\n".join(cat_lines) or "None")
        )
    return _mem_compute(mem, ds_sig, "stats", _c)


def stat_corr(df: pd.DataFrame, mem: MemoryStore, ds_sig: str) -> str:
    def _c():
        nums = _num(df)
        cats = _cat(df)
        if len(nums) < 2:
            corr_txt = "< 2 numeric cols — no Pearson correlation."
        else:
            corr  = df[nums].corr(numeric_only=True)
            upper = corr.abs().where(
                np.triu(np.ones(corr.shape), k=1).astype(bool))
            top = upper.stack().sort_values(ascending=False).head(10)
            corr_txt = "\n".join(
                f"- {a} ↔ {b}: r={corr.loc[a, b]:.3f}"
                for (a, b), _ in top.items()
            )
        grp_lines = []
        if cats and nums:
            for cat in cats[:3]:
                if df[cat].nunique() <= 15:
                    g = (df.groupby(cat)[nums[0]].mean()
                           .sort_values(ascending=False).head(6))
                    pairs = ", ".join(f"{i}={v:.3f}" for i, v in g.items())
                    grp_lines.append(f"- mean {nums[0]} by {cat}: {pairs}")
        return (
            "Top correlations:\n" + corr_txt +
            "\n\nGroup means:\n" + ("\n".join(grp_lines) or "None")
        )
    return _mem_compute(mem, ds_sig, "correlations", _c)


def stat_viz(df: pd.DataFrame, mem: MemoryStore, ds_sig: str) -> str:
    def _c():
        nums = _num(df)
        cats = _cat(df)
        return (
            f"Shape: {df.shape[0]:,} × {df.shape[1]}\n"
            f"Numeric: {', '.join(nums) or 'None'}\n"
            f"Categorical: {', '.join(cats) or 'None'}\n"
            f"Missing: {int(df.isna().sum().sum())}\n"
            f"Dtypes: {dict(df.dtypes.astype(str).head(10))}"
        )
    return _mem_compute(mem, ds_sig, "viz_ctx", _c)


# ============================================================
# ❼  REACT LOOP ENGINE  (now receives per-agent model)
# ============================================================

_PARSER = ErrorCorrectionParser()


def _build_react_prompt(agent_key: str, stats_ctx: str,
                        tool_plan: Dict, trace: List[AgentStep],
                        attempt: int, prior_error: str) -> str:
    trace_txt = ""
    for s in trace[-2:]:
        pfx = {
            "thought":     "💭 Thought",
            "action":      "⚡ Action",
            "observation": "👁 Observation",
            "error":       "🔴 Error",
            "code":        "💻 Code",
        }.get(s.step_type, s.step_type)
        trace_txt += f"\n{pfx}: {s.content[:120]}"

    plan_short = {
        "library":  tool_plan.get("library", "pandas"),
        "strategy": str(tool_plan.get("strategy", ""))[:120],
        "note":     str(tool_plan.get("note", ""))[:80],
    }
    plan_txt = json.dumps(plan_short, default=str)
    error_section = (
        f"\nPrior error (fix it): {prior_error[:150]}"
        if prior_error else ""
    )
    return (
        f"You are {AGENTS[agent_key]['name']} (attempt {attempt}/{MAX_REACT_CYCLES}).\n"
        f"Tool: {plan_txt}\n\n"
        f"Data summary:\n{stats_ctx[:600]}\n"
        f"{trace_txt}"
        f"{error_section}\n\n"
        "Give a concise markdown bullet-point EDA analysis. Name specific columns. "
        "Use bold headers. Be brief but insightful."
    )


def _extract_code_blocks(text: str) -> List[CodeBlock]:
    blocks: List[CodeBlock] = []
    for lang, code in re.findall(
            r"```(python|sql|stats)\s*\n(.*?)```", text, re.DOTALL):
        blocks.append(CodeBlock(
            language=lang,
            code=code.strip(),
            purpose="LLM-generated code block",
        ))
    return blocks


def react_loop(
    agent_key: str,
    stats_ctx: str,
    tool_plan: Dict,
    model: str,           # ← now the per-agent assigned model
    state: PipelineState,
) -> Tuple[str, List[AgentStep], List[CodeBlock], int, str]:
    """Returns (output, trace, code_blocks, retries, model_used)."""
    trace:     List[AgentStep] = []
    code_blks: List[CodeBlock] = []
    retries:   int             = 0
    output:    str             = ""
    prior_err: str             = ""
    model_used: str            = model

    for attempt in range(1, MAX_REACT_CYCLES + 1):
        thought = (
            f"[Observe] I have pre-computed {agent_key} statistics from "
            f"MemoryStore. Tool plan uses {tool_plan.get('library', '?')}. "
            + (f"Prior error: {prior_err[:120]}" if prior_err
               else "Starting fresh — no prior error.")
        )
        trace.append(AgentStep("thought", thought))

        prompt = _build_react_prompt(
            agent_key, stats_ctx, tool_plan, trace, attempt, prior_err
        )
        trace.append(AgentStep(
            "action",
            f"[Plan → call_llm] agent={agent_key}, model={AGENT_MODEL_LABELS.get(agent_key, model)}, "
            f"attempt={attempt}, lib={tool_plan.get('library', '?')}"
        ))

        ok, content, used = call_llm(AGENTS[agent_key]["system"], prompt, model)
        model_used = used
        state.metrics["llm_calls"] += 1

        if ok:
            new_blocks = _extract_code_blocks(content)
            for blk in new_blocks:
                if blk.language == "python":
                    blk.execute_python(state.df)
                    state.metrics["code_runs"] += 1
                    if not blk.executed:
                        state.metrics["code_errors"] += 1
                        trace.append(AgentStep(
                            "error",
                            f"[Execute Error] {blk.error} — injecting into next attempt."
                        ))
                        prior_err = (
                            f"Code block failed:\n```python\n{blk.code}\n```\n"
                            f"Error: {blk.error}"
                        )
                    else:
                        trace.append(AgentStep(
                            "code",
                            f"[Execute OK] output={str(blk.output)[:120]}"
                        ))
                        prior_err = ""
                code_blks.append(blk)

        valid, parse_err = _PARSER.parse(content if ok else "", agent_key)

        if ok and valid and not prior_err:
            trace.append(AgentStep(
                "observation",
                f"[Parser ✅] {len(content)} chars accepted via {used}."
            ))
            output = content
            break

        err_detail = content if not ok else parse_err
        if not prior_err:
            prior_err = err_detail
        trace.append(AgentStep(
            "error",
            f"[Parser ❌ attempt {attempt}] {err_detail[:200]}"
        ))
        state.metrics["errors"]  += 1
        retries                  += 1
        time.sleep(min(0.3 * attempt, 1.0))

    if not output:
        output = _local_fallback(agent_key, state.df)
        trace.append(AgentStep(
            "observation", "All LLM attempts failed — local fallback used."
        ))

    return _clean(output), trace, code_blks, retries, model_used


# ============================================================
# LOCAL FALLBACKS
# ============================================================

def _local_fallback(key: str, df: pd.DataFrame) -> str:
    nums = _num(df)
    cats = _cat(df)
    fb: Dict[str, Callable] = {
        "quality": lambda: (
            f"**Data Quality (local fallback)**\n"
            f"- Rows: {len(df):,} | Cols: {len(df.columns)}\n"
            f"- Duplicates: {int(df.duplicated().sum()):,}\n"
            f"- Missing: {int(df.isna().sum().sum()):,}\n"
            f"- High-missing: "
            f"{', '.join(c for c in df.columns if df[c].isna().mean() > .3) or 'None'}"
        ),
        "stats": lambda: (
            "**Statistics (local fallback)**\n" +
            "\n".join(
                f"- {c}: mean={df[c].mean():.3f}, std={df[c].std():.3f}"
                for c in nums[:6]
            )
        ),
        "corr": lambda: (
            "**Correlations (local fallback)**\n" +
            ("\n".join(
                f"- {a} ↔ {b}: r={df[nums].corr().loc[a, b]:.3f}"
                for a, b in [
                    (nums[i], nums[j])
                    for i in range(min(len(nums), 4))
                    for j in range(i + 1, min(len(nums), 4))
                ][:5]
            ) if len(nums) >= 2 else "Not enough numeric cols.")
        ),
        "viz": lambda: (
            "**Viz Recommendations (local fallback)**\n"
            + (f"1. Histogram({nums[0]})\n"             if nums else "")
            + (f"2. Scatter({nums[0]}, {nums[1]})\n"    if len(nums) >= 2 else "")
            + (f"3. Heatmap(all numeric)\n"              if len(nums) >= 2 else "")
            + (f"4. Boxplot({nums[0]} by {cats[0]})\n"  if nums and cats else "")
            + (f"5. Bar({cats[0]})\n"                   if cats else "")
        ),
    }
    fn = fb.get(key)
    if fn:
        return fn()
    return "# EDA Report — Fallback\n\nSee individual agent sections above."


# ============================================================
# ❽  EVALUATION FRAMEWORK
# ============================================================

@dataclass
class EvalResult:
    dataset_name:          str
    task_completion_rate:  float
    agent_completion:      Dict[str, bool]
    total_retries:         int
    total_code_runs:       int
    total_code_errors:     int
    code_accuracy:         float
    memory_hit_rate:       float
    baseline_summary:      str
    vs_baseline_note:      str
    elapsed_s:             float


def run_evaluation(state: PipelineState) -> EvalResult:
    mem    = _get_mem()
    mstats = mem.stats
    total_m = mstats["hits"] + mstats["misses"]

    agent_done = {
        k: (k in state.agent_findings and
            len(state.agent_findings[k]) >= MIN_INSIGHT_CHARS)
        for k in ["quality", "stats", "corr", "viz"]
    }
    agent_done["orchestrator"] = len(state.final_summary.strip()) >= MIN_INSIGHT_CHARS
    tcr = sum(agent_done.values()) / max(len(agent_done), 1)

    runs = state.metrics.get("code_runs",   0)
    errs = state.metrics.get("code_errors", 0)
    acc  = 1.0 - (errs / max(runs, 1))

    hit_rate = mstats["hits"] / max(total_m, 1)

    nums = _num(state.df)
    cats = _cat(state.df)
    b_prompt = (
        f"Brief EDA: {state.df.shape[0]}r × {state.df.shape[1]}c. "
        f"Numeric: {', '.join(nums[:5])}. "
        f"Cat: {', '.join(cats[:3])}. "
        f"Missing: {int(state.df.isna().sum().sum())}. "
        "Cover quality, stats, correlations. Markdown bullets, be concise."
    )
    # Baseline uses the dedicated lightweight model
    baseline_model = AGENT_MODELS.get("baseline", "meta-llama/llama-3.1-8b-instruct:free")
    ok, baseline, _ = call_llm(
        "You are a single-prompt data scientist.",
        b_prompt, baseline_model, max_tok=400,
    )
    if not ok:
        baseline = "(Baseline call failed.)"

    retries = sum(state.retry_counts.values())
    note = (
        f"**Multi-agent (v5):** {state.metrics['llm_calls']} LLM calls, "
        f"{len(state.enabled_agents) + 1} specialised agents (each with own model), "
        f"{retries} ReAct retries, "
        f"{state.metrics.get('code_runs', 0)} code executions, "
        f"{mstats['hits']} memory cache hits.  \n"
        f"**Baseline:** 1 LLM call ({AGENT_MODEL_LABELS.get('baseline', 'Llama 3.1 8B')}), "
        f"no specialisation, no error-correction, no memory, no code execution."
    )

    return EvalResult(
        dataset_name         = state.ds_name,
        task_completion_rate = tcr,
        agent_completion     = agent_done,
        total_retries        = retries,
        total_code_runs      = runs,
        total_code_errors    = errs,
        code_accuracy        = acc,
        memory_hit_rate      = hit_rate,
        baseline_summary     = _clean(baseline),
        vs_baseline_note     = note,
        elapsed_s            = state.metrics["elapsed"],
    )


# ============================================================
# ❾  PIPELINE GRAPH
# ============================================================

def _make_graph(enabled: List[str]) -> StateGraph:
    g = StateGraph("EDA-v5")

    def _make_agent_node(key: str) -> Callable:
        STAT_FNS = {
            "quality": stat_quality,
            "stats":   stat_stats,
            "corr":    stat_corr,
            "viz":     stat_viz,
        }
        PLAN_FNS = {
            "quality": lambda s: s.plan_for_quality(),
            "stats":   lambda s: s.plan_for_stats(),
            "corr":    lambda s: s.plan_for_corr(),
            "viz":     lambda s: s.plan_for_viz(),
        }

        def _node(state: PipelineState) -> PipelineState:
            if key not in state.enabled_agents:
                state.node_states[key] = NodeState.SKIPPED
                return state

            state.node_states[key] = NodeState.RUNNING
            mem    = _get_mem()
            ds_sig = state.ds_sig
            df     = state.df
            sched  = ToolScheduler(df)

            stats_ctx = STAT_FNS[key](df, mem, ds_sig)

            if key == "quality":
                stat_describe(df, mem, ds_sig)

            tool_plan = PLAN_FNS[key](sched)

            if key == "viz" and state.agent_findings:
                prior = "\n".join(
                    f"{AGENTS[k]['name']}: {v[:120]}…"
                    for k, v in state.agent_findings.items()
                )
                stats_ctx = stats_ctx + "\nPrior findings:\n" + prior

            # ← Use the per-agent assigned model instead of the global model
            agent_model = AGENT_MODELS.get(key, state.model)

            output, trace, blks, retries, model_used = react_loop(
                key, stats_ctx, tool_plan, agent_model, state
            )

            state.agent_findings[key]    = output
            state.react_traces[key]      = trace
            state.code_blocks[key]       = blks
            state.retry_counts[key]      = retries
            state.agent_models_used[key] = model_used   # record actual model used
            state.metrics["agents_done"] += 1
            state.metrics["retries"]     += retries

            ms = _get_mem().stats
            state.metrics["memory_hits"]   = ms["hits"]
            state.metrics["memory_misses"] = ms["misses"]

            state.node_states[key] = NodeState.DONE
            return state

        return _node

    def _orchestrator(state: PipelineState) -> PipelineState:
        state.node_states["orchestrator"] = NodeState.RUNNING
        mem    = _get_mem()
        ds_sig = state.ds_sig
        df     = state.df

        desc_cache = stat_describe(df, mem, ds_sig)[:200]
        qual_cache = stat_quality(df, mem, ds_sig)[:200]

        LIMIT = 200
        findings_txt = "\n---\n".join(
            f"{AGENTS[k]['icon']} {AGENTS[k]['name']}:\n"
            f"{v[:LIMIT]}{'…' if len(v) > LIMIT else ''}"
            for k, v in state.agent_findings.items()
        )

        prompt = (
            f"Dataset: {state.ds_name} — "
            f"{df.shape[0]:,}r × {df.shape[1]}c. "
            f"Cols: {', '.join(df.columns[:10].tolist())}.\n"
            f"Stats: {desc_cache[:150]}\n"
            f"Quality: {qual_cache[:150]}\n"
            f"Findings:\n{findings_txt}\n\n"
            "Write a concise EDA report with: Executive Summary, "
            "Key Findings, Top 3 Recommendations, Next Steps. "
            "Use markdown. Be brief."
        )

        # ← Orchestrator always uses DeepSeek V3
        orch_model = AGENT_MODELS.get("orchestrator", "deepseek/deepseek-chat-v3-0324:free")

        state.final_summary = ""
        for attempt in range(1, MAX_REACT_CYCLES + 1):
            ok, content, used = call_llm(
                AGENTS["orchestrator"]["system"],
                prompt, orch_model,
                temp=0.1, max_tok=400,
            )
            state.metrics["llm_calls"] += 1
            if ok and len(content.strip()) > 20:
                state.final_summary = _clean(content)
                state.agent_models_used["orchestrator"] = used
                break
            state.metrics["errors"]  += 1
            state.metrics["retries"] += 1
            time.sleep(min(0.5 * attempt, 1.0))

        if not state.final_summary:
            sections = [
                f"### {AGENTS.get(k, {'icon': '•', 'name': k})['icon']} "
                f"{AGENTS.get(k, {'name': k})['name']}\n{v}"
                for k, v in state.agent_findings.items()
            ]
            state.final_summary = (
                "# EDA Report — Orchestrator Fallback\n\n"
                "> LLM unavailable — sub-agent findings compiled below.\n\n"
                + "\n\n---\n\n".join(sections)
            )

        mem.put(ds_sig, "orchestrator_report", state.final_summary)
        state.node_states["orchestrator"] = NodeState.DONE
        state.metrics["agents_done"] += 1
        return state

    def _eval(state: PipelineState) -> PipelineState:
        try:
            ev = run_evaluation(state)
            state.eval_results = {
                "task_completion_rate": ev.task_completion_rate,
                "agent_completion":     ev.agent_completion,
                "total_retries":        ev.total_retries,
                "total_code_runs":      ev.total_code_runs,
                "total_code_errors":    ev.total_code_errors,
                "code_accuracy":        ev.code_accuracy,
                "memory_hit_rate":      ev.memory_hit_rate,
                "baseline_summary":     ev.baseline_summary,
                "vs_baseline_note":     ev.vs_baseline_note,
            }
        except Exception:
            state.eval_results = {"error": traceback.format_exc(limit=4)}
        return state

    for k in ["quality", "stats", "corr", "viz"]:
        g.add_node(k, _make_agent_node(k))
    g.add_node("orchestrator", _orchestrator)
    g.add_node("eval",         _eval)

    g.add_edge("quality",      "stats")
    g.add_edge("stats",        "corr")
    g.add_edge("corr",         "viz")
    g.add_edge("viz",          "orchestrator")
    g.add_edge("orchestrator", "eval")
    g.set_entry("quality")
    return g


# ============================================================
# TOP-LEVEL RUNNER
# ============================================================

def run_pipeline(df: pd.DataFrame, model: str,
                 enabled: List[str]) -> PipelineState:
    t0     = time.time()
    ds_sig = _sig(df)
    mem    = _get_mem()

    if st.session_state.get("last_ds_sig") != ds_sig:
        mem.clear()
        st.session_state["last_ds_sig"] = ds_sig

    state = PipelineState(
        ds_sig         = ds_sig,
        ds_name        = st.session_state.get("last_ds_name", "dataset"),
        df             = df,
        model          = model,
        enabled_agents = enabled,
        node_states    = {k: NodeState.PENDING
                          for k in list(AGENTS.keys()) + ["eval"]},
    )

    graph   = _make_graph(enabled)
    prog    = st.progress(0, text="Initialising…")
    total   = len(enabled) + 2
    done_ct = [0]

    for nm, fn in list(graph._nodes.items()):
        def _wrap(fn=fn, name=nm):
            def _inner(s: PipelineState) -> PipelineState:
                lbl       = AGENTS.get(name, {}).get("name", name.title())
                mdl_label = AGENT_MODEL_LABELS.get(name, "")
                mdl_tag   = f" [{mdl_label}]" if mdl_label else ""
                prog.progress(
                    min(int(done_ct[0] / total * 100), 93),
                    text=(
                        f"{'🤖' if name not in ('orchestrator', 'eval') else '🧠'}"
                        f" Running {lbl}{mdl_tag}…"
                    ),
                )
                r = fn(s)
                done_ct[0] += 1
                return r
            return _inner
        graph._nodes[nm] = _wrap(fn, nm)

    state = graph.run(state)
    state.metrics["elapsed"] = round(time.time() - t0, 1)
    prog.progress(100, text="✅ Pipeline complete!")
    return state


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def _load_titanic() -> pd.DataFrame:
    return pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/"
        "master/titanic.csv"
    )


@st.cache_data(show_spinner=False)
def _load_diamonds() -> pd.DataFrame:
    try:
        import seaborn as sns
        return sns.load_dataset("diamonds")
    except Exception:
        rng = np.random.default_rng(42)
        n   = 1000
        return pd.DataFrame({
            "carat":   rng.uniform(0.2, 5.0, n).round(2),
            "cut":     rng.choice(["Fair", "Good", "Very Good", "Premium", "Ideal"], n),
            "color":   rng.choice(list("DEFGHIJ"), n),
            "clarity": rng.choice(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], n),
            "depth":   rng.uniform(55, 75, n).round(1),
            "table":   rng.uniform(50, 70, n).round(0),
            "price":   (rng.uniform(300, 18800, n)).astype(int),
            "x": rng.uniform(3.7, 10.7, n).round(2),
            "y": rng.uniform(3.7, 10.7, n).round(2),
            "z": rng.uniform(2.1, 6.6,  n).round(2),
        })


@st.cache_data(show_spinner=False)
def _load_builtin(name: str) -> pd.DataFrame:
    loaders = {
        "Iris":          lambda: load_iris(as_frame=True).frame,
        "Wine":          lambda: load_wine(as_frame=True).frame,
        "Breast Cancer": lambda: load_breast_cancer(as_frame=True).frame,
        "Diamonds":      _load_diamonds,
    }
    return loaders[name]() if name in loaders else _load_titanic()


def _read_upload(f) -> Optional[pd.DataFrame]:
    if f is None:
        return None
    n = f.name.lower()
    try:
        if n.endswith(".csv"):
            return pd.read_csv(f)
        if n.endswith((".xlsx", ".xls")):
            return pd.read_excel(f)
        if n.endswith(".json"):
            return pd.read_json(f)
    except Exception as e:
        st.error(f"Read error: {e}")
        return None
    st.error("Unsupported format.")
    return None


# ============================================================
# CHARTS
# ============================================================

def chart_hist(df: pd.DataFrame) -> None:
    nums = _num(df)
    if not nums:
        st.info("No numeric cols.")
        return
    col = st.selectbox("Column", nums, key="ch_col")
    st.plotly_chart(
        px.histogram(df, x=col, nbins=35, title=f"Distribution — {col}"),
        use_container_width=True,
    )


def chart_scatter(df: pd.DataFrame) -> None:
    nums = _num(df)
    cats = _cat(df)
    if len(nums) < 2:
        st.info("Need ≥2 numeric.")
        return
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", nums, 0, key="cs_x")
    y = c2.selectbox("Y", nums, 1, key="cs_y")
    h = c3.selectbox("Color", ["None"] + cats, key="cs_h")
    st.plotly_chart(
        px.scatter(df, x=x, y=y,
                   color=None if h == "None" else h,
                   title=f"{x} vs {y}"),
        use_container_width=True,
    )


def chart_heatmap(df: pd.DataFrame) -> None:
    nums = _num(df)
    if len(nums) < 2:
        st.info("Need ≥2 numeric.")
        return
    corr = df[nums].corr(numeric_only=True)
    st.plotly_chart(
        go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            zmid=0, text=corr.round(2).values, texttemplate="%{text}",
            colorscale="RdBu",
        )).update_layout(title="Correlation Heatmap", height=460),
        use_container_width=True,
    )


def chart_outliers(df: pd.DataFrame) -> None:
    nums = _num(df)
    if not nums:
        st.info("No numeric cols.")
        return
    rows = []
    for c in nums:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([.25, .75])
        iqr     = q3 - q1
        cnt     = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        rows.append({"Column": c, "Outliers": cnt,
                     "Pct": round(cnt / max(len(df), 1) * 100, 2)})
    if rows:
        st.plotly_chart(
            px.bar(
                pd.DataFrame(rows).sort_values("Outliers", ascending=False),
                x="Column", y="Outliers", color="Pct",
                title="Outliers per Column (IQR)",
            ),
            use_container_width=True,
        )


def chart_missing(df: pd.DataFrame) -> None:
    m = df.isna().sum().reset_index()
    m.columns = ["Column", "Missing"]
    m["Pct"]  = (m["Missing"] / max(len(df), 1) * 100).round(2)
    m = m[m["Missing"] > 0].sort_values("Missing", ascending=False)
    if m.empty:
        st.success("No missing values!")
        return
    st.plotly_chart(
        px.bar(m, x="Column", y="Pct",
               title="Missing Values (%)", color="Pct",
               color_continuous_scale="Reds"),
        use_container_width=True,
    )


# ============================================================
# REACT TRACE RENDERER
# ============================================================

def render_trace(trace: List[AgentStep]) -> None:
    icons = {"thought": "💭", "action": "⚡", "observation": "👁️",
             "error": "🔴", "code": "💻"}
    cols  = {"thought": "trace-thought", "action": "trace-action",
             "observation": "trace-observe", "error": "trace-error",
             "code": "trace-action"}
    lines = [
        f'<span class="{cols.get(s.step_type, "")}">'
        f'{icons.get(s.step_type, "•")} [{s.step_type.upper()}] '
        f'{s.content[:200]}{"…" if len(s.content) > 200 else ""}</span>'
        for s in trace
    ]
    st.markdown(
        '<div class="react-trace">' + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# EXPORT
# ============================================================

def _export(ps: PipelineState) -> str:
    ev = ps.eval_results
    # Build model assignment table for report
    model_rows = "\n".join(
        f"| {AGENTS.get(k, {'icon':'•','name':k})['icon']} "
        f"{AGENTS.get(k, {'name':k})['name']} "
        f"| {AGENT_MODEL_LABELS.get(k, '?')} "
        f"| {ps.agent_models_used.get(k, 'N/A')} |"
        for k in ["quality", "stats", "corr", "viz", "orchestrator"]
    )
    lines = [
        f"# EDA Report — {ps.ds_name}",
        f"_Generated: {datetime.now():%Y-%m-%d %H:%M} · v5 (Per-Agent Models + ReAct + Memory + Eval)_",
        "", "---", "",
        "## 🤖 Per-Agent Model Assignments",
        "| Agent | Assigned Model | Model Used |",
        "|---|---|---|",
        model_rows,
        "", "---", "",
        "## 🧠 Final Report", ps.final_summary, "", "---", "",
        "## Agent Findings",
    ]
    for k, v in ps.agent_findings.items():
        a = AGENTS.get(k, {"icon": "•", "name": k})
        lines += [
            f"### {a['icon']} {a['name']} "
            f"_(model: {AGENT_MODEL_LABELS.get(k, '?')})_",
            v, "",
        ]
    if ev and "error" not in ev:
        lines += [
            "---", "## 📊 Evaluation",
            f"- Task Completion Rate: **{ev.get('task_completion_rate', 0):.0%}**",
            f"- Error-Recovery Retries: {ev.get('total_retries', 0)}",
            f"- Code Execution Accuracy: {ev.get('code_accuracy', 1):.0%}",
            f"- Memory Hit Rate: {ev.get('memory_hit_rate', 0):.0%}",
            "", "### Baseline (single-prompt) EDA",
            ev.get("baseline_summary", "N/A"), "",
            f"### vs Baseline\n{ev.get('vs_baseline_note', '')}",
        ]
    lines += [
        "", "---",
        f"LLM calls: {ps.metrics['llm_calls']} | "
        f"Retries: {ps.metrics.get('retries', 0)} | "
        f"Memory hits: {ps.metrics.get('memory_hits', 0)} | "
        f"Elapsed: {ps.metrics['elapsed']}s",
    ]
    return "\n".join(lines)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Multi-Agent EDA v5",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{ font-family:'Syne',sans-serif; }

.hero{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
  border-radius:20px;padding:2rem;margin-bottom:1.3rem;color:white;}
.hero-title{font-size:2.1rem;font-weight:800;}
.hero-title span{color:#c4b5fd;}
.hero-sub{color:rgba(255,255,255,.7);margin-top:.4rem;font-size:.9rem;}

.model-table{width:100%;border-collapse:collapse;margin:.6rem 0;font-size:.82rem;}
.model-table th{background:#1e1b4b;color:white;padding:6px 10px;text-align:left;}
.model-table td{padding:5px 10px;border-bottom:1px solid #e5e7eb;}
.model-table tr:hover td{background:#f5f3ff;}
.model-badge{display:inline-block;border-radius:5px;padding:2px 8px;font-size:.7rem;
  font-weight:700;color:white;background:#6d28d9;}

.agent-pipeline{display:flex;gap:10px;flex-wrap:wrap;margin:1rem 0;}
.agent-card{flex:1;min-width:120px;border:1px solid #e5e7eb;border-radius:14px;
  padding:.8rem;text-align:center;background:white;}
.agent-card.done   {background:#f0fdf4;border-color:#10b981;}
.agent-card.running{background:#eff6ff;border-color:#3b82f6;animation:pulse 1s infinite;}
.agent-card.pending{opacity:.5;}
.agent-card.skipped{opacity:.3;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,.4);}
                 50%{box-shadow:0 0 0 6px rgba(59,130,246,0);}}
.agent-icon{font-size:1.5rem;}
.agent-name{font-size:.78rem;font-weight:700;display:block;margin-top:4px;}
.agent-model{font-size:.62rem;color:#7c3aed;font-weight:700;display:block;margin-top:2px;}
.agent-status{font-size:.68rem;color:#6b7280;}

.badge{display:inline-block;border-radius:6px;padding:2px 7px;font-size:.65rem;
  font-weight:700;margin:2px;vertical-align:middle;color:white;}
.badge-react  {background:#059669;}
.badge-memory {background:#b45309;}
.badge-code   {background:#7c3aed;}
.badge-retry  {background:#dc2626;}
.badge-model  {background:#1d4ed8;}
.badge-local  {background:#6b7280;}

.final-box{border:2px solid #a78bfa;border-radius:18px;padding:1.4rem 1.6rem;
  background:linear-gradient(135deg,#faf5ff,#eff6ff);margin:1rem 0;}
.eval-box {border:2px solid #fb923c;border-radius:18px;padding:1.2rem 1.4rem;
  background:linear-gradient(135deg,#fff7ed,#fef3c7);margin:1rem 0;}

.metric-card{border:1px solid #e5e7eb;border-radius:14px;background:white;
  padding:1rem;text-align:center;}
.metric-val{font-size:1.6rem;font-weight:800;color:#111827;}
.metric-lbl{font-size:.65rem;font-weight:700;color:#6b7280;text-transform:uppercase;}

.react-trace{background:#0f172a;color:#94a3b8;border-radius:10px;
  padding:.8rem 1rem;font-family:'JetBrains Mono',monospace;
  font-size:.7rem;line-height:1.7;margin:.4rem 0;}
.trace-thought {color:#7dd3fc;} .trace-action{color:#86efac;}
.trace-observe {color:#fde68a;} .trace-error {color:#f87171;}

.step-guide{background:#f8fafc;border-left:4px solid #6366f1;border-radius:0 12px 12px 0;
  padding:.7rem 1rem;margin:.4rem 0;font-size:.82rem;}
.step-label{font-weight:800;font-size:.7rem;text-transform:uppercase;color:#6366f1;}

.footer{text-align:center;color:#9ca3af;font-size:.75rem;
  padding:1.5rem 0;border-top:1px solid #f3f4f6;margin-top:2rem;}
</style>
""", unsafe_allow_html=True)


# ── session state ─────────────────────────────────────────────────────────────
def _init():
    defs = {
        "api_status":      "unknown",
        "api_last_error":  "",
        "ep_cache":        {},
        "pipeline_state":  None,
        "history":         [],
        "last_ds_name":    "",
        "last_ds_sig":     "",
        "show_clear":      False,
        "_llm_calls":      0,
    }
    for k, v in defs.items():
        st.session_state.setdefault(k, v)

_init()
st.session_state["api_key"] = _BACKEND_API_KEY

if st.session_state.pop("show_clear", False):
    st.toast("Cleared!", icon="🧹")


# ── header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Multi-Agent <span>EDA System</span>
    <small style="font-size:1rem;opacity:.5"> v5 · Per-Agent Models</small></div>
  <div class="hero-sub">
    ReAct Loop · Error-Correction Parser · LangGraph State Machine ·
    Memory Store · Dynamic Tool Scheduler · Pydantic Schemas ·
    Evaluation Framework · 5 Datasets ·
    <strong style="color:#c4b5fd">Each agent runs its optimal LLM</strong>
  </div>
</div>""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    s = st.session_state.api_status
    if   s == "ok":    st.success("🔑 API connected ✅")
    elif s == "error": st.error(f"🔑 API error: {st.session_state.get('api_last_error', '')}")
    else:              st.info("🔑 API key configured — click Test Connection")

    # Sidebar model selector is now only used for the Test Connection button
    # and as global fallback if an agent's assigned model fails
    model_lbl = st.selectbox(
        "Global Fallback Model",
        list(MODEL_CHOICES.keys()),
        help="Used only if an agent's assigned model is unavailable.",
    )
    sel_model = MODEL_CHOICES[model_lbl]

    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Testing…"):
            ok, msg = test_connection(sel_model)
        (st.success if ok else st.error)(msg)

    # ── Per-agent model assignment table ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Agent → Model Assignment")
    rows_html = "".join(
        f"<tr>"
        f"<td>{AGENTS[k]['icon']} {AGENTS[k]['name']}</td>"
        f"<td><span class='model-badge'>{AGENT_MODEL_LABELS[k]}</span></td>"
        f"</tr>"
        for k in ["quality", "stats", "corr", "viz", "orchestrator"]
    )
    rows_html += (
        f"<tr><td>📏 Eval Baseline</td>"
        f"<td><span class='model-badge'>{AGENT_MODEL_LABELS['baseline']}</span></td></tr>"
    )
    st.markdown(
        f"<table class='model-table'>"
        f"<thead><tr><th>Agent</th><th>Model</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each agent is assigned the model best suited to its task. "
        "Fallback chain activates if the assigned model is unavailable."
    )

    st.markdown("---")

    # ── dataset ───────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    source = st.radio(
        "Source",
        ["Titanic", "Iris", "Wine", "Breast Cancer", "Diamonds", "Upload"],
    )
    df: Optional[pd.DataFrame] = None

    if source == "Upload":
        up = st.file_uploader("CSV / Excel / JSON",
                               type=["csv", "xlsx", "xls", "json"])
        if up:
            df = _read_upload(up)
            if df is not None:
                st.session_state.last_ds_name = up.name
    else:
        df = _load_builtin(source)
        st.session_state.last_ds_name = source

    if df is not None:
        st.caption(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
        with st.expander("Preview (5 rows)"):
            st.dataframe(df.head(5), use_container_width=True)

    st.markdown("---")

    # ── agent toggles ─────────────────────────────────────────────────────
    st.markdown("### 🔧 Enable Agents")
    enabled: List[str] = []
    for k in ["quality", "stats", "corr", "viz"]:
        a = AGENTS[k]
        label = f"{a['icon']} {a['name']} · {AGENT_MODEL_LABELS[k]}"
        if st.checkbox(label, value=True, key=f"en_{k}"):
            enabled.append(k)
    st.caption("Orchestrator (DeepSeek V3) + Eval always run last.")
    st.markdown("---")

    # ── memory stats ──────────────────────────────────────────────────────
    mem   = _get_mem()
    ms    = mem.stats
    st.markdown("### 🧠 Memory Store")
    c1, c2, c3 = st.columns(3)
    c1.metric("Keys",   ms["keys"])
    c2.metric("Hits",   ms["hits"])
    c3.metric("Misses", ms["misses"])
    if ms["hits"] + ms["misses"] > 0:
        st.progress(
            ms["hits"] / (ms["hits"] + ms["misses"]),
            text=f"Hit rate: {ms['hits'] / (ms['hits'] + ms['misses']):.0%}",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear Results", use_container_width=True):
            mem.clear()
            st.session_state.pipeline_state = None
            st.session_state.show_clear     = True
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.toast("History cleared!", icon="🗑️")
            st.rerun()

    # ── history ───────────────────────────────────────────────────────────
    hist = st.session_state.history
    if hist:
        st.markdown("---")
        st.markdown(f"### 📚 History ({len(hist)})")
        for item in reversed(hist[-4:]):
            with st.expander(f"{item['ts']} · {item['name']}"):
                st.markdown(item["summary"])
    else:
        st.markdown("---")
        st.caption("No history yet — run the pipeline to record entries.")


# ── guard ─────────────────────────────────────────────────────────────────────
if df is None:
    st.info("Choose or upload a dataset from the sidebar to begin.")
    st.stop()

cur_sig = _sig(df)
ps: Optional[PipelineState] = st.session_state.pipeline_state
if ps is not None and ps.ds_sig != cur_sig:
    st.session_state.pipeline_state = None
    ps = None


# ── metrics bar ───────────────────────────────────────────────────────────────
st.markdown("#### 📊 Pipeline Metrics")
m = ps.metrics if ps else {}
metric_items = [
    (m.get("agents_done", 0),       "Agents Done"),
    (m.get("llm_calls",   0),       "LLM Calls"),
    (m.get("retries",     0),       "ReAct Retries"),
    (m.get("memory_hits", 0),       "Memory Hits"),
    (m.get("code_runs",   0),       "Code Runs"),
    (m.get("code_errors", 0),       "Code Errors"),
    (f'{m.get("elapsed",  0.0)}s',  "Elapsed"),
]
for col, (val, lbl) in zip(st.columns(len(metric_items)), metric_items):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-val">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )
st.markdown("---")


# ── main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

_STATE_CSS: Dict[str, str] = {
    "PENDING": "pending",
    "RUNNING": "running",
    "DONE":    "done",
    "FAILED":  "failed",
    "SKIPPED": "skipped",
}

def _node_css(state_val: Any) -> str:
    if isinstance(state_val, NodeState):
        return _STATE_CSS.get(state_val.name, "pending")
    return _STATE_CSS.get(str(state_val).upper().split(".")[-1], "pending")


with left:
    st.markdown("#### 🔄 ReAct Pipeline — Per-Agent Model View")

    cur_css = (
        {k: _node_css(ps.node_states.get(k, NodeState.PENDING))
         for k in AGENTS}
        if ps else {k: "pending" for k in AGENTS}
    )

    # Enhanced pipeline cards showing model per agent
    cards_html = ""
    for k in AGENTS:
        used_model = (ps.agent_models_used.get(k, "") if ps else "")
        display_model = (
            used_model.split("/")[-1].replace(":free", "")
            if used_model else AGENT_MODEL_LABELS.get(k, "")
        )
        cards_html += (
            f'<div class="agent-card {cur_css[k]}">'
            f'<div class="agent-icon">{AGENTS[k]["icon"]}</div>'
            f'<span class="agent-name">{AGENTS[k]["name"]}</span>'
            f'<span class="agent-model">{display_model}</span>'
            f'<span class="agent-status">{cur_css[k]}</span></div>'
        )
    st.markdown(
        f'<div class="agent-pipeline">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📅 Architecture Map — v5 Changes", expanded=False):
        steps = [
            ("NEW · Per-Agent Model Assignment",
             "Each agent runs its optimal LLM: "
             "Quality→Gemma 3 27B · Stats→DeepSeek R1 · Corr→DeepSeek R1 · "
             "Viz→Mistral 7B · Orchestrator→DeepSeek V3 · Baseline→Llama 3.1 8B. "
             "Fallback chain activates transparently if assigned model is unavailable."),
            ("Days 1-4 · Tool Environment",
             "Docker sandbox · Pydantic schemas (ToolInput/ToolOutput/AgentStep/CodeBlock) · "
             "Tool registry (describe_column, compute_correlation, detect_outliers, …)"),
            ("Days 5-9 · Agent Core (LangGraph)",
             "StateGraph: quality→stats→corr→viz→orchestrator→eval · "
             "ReAct cycle: Observe→Plan Hypothesis→Write Code→Execute→Observe Error→Rewrite"),
            ("Days 10-12 · Memory Module",
             "MemoryStore caches df.describe(), quality_stats, correlations, viz_context. "
             "Pre-warmed in the Quality node — all later agents get it free."),
            ("Days 13-15 · Evaluation",
             "5 datasets: Titanic, Iris, Wine, Breast Cancer, Diamonds. "
             "Metrics: Task Completion Rate, Error-Recovery Loops, Code Execution Accuracy, "
             "vs single-prompt Llama 3.1 8B baseline."),
        ]
        for title, desc in steps:
            st.markdown(
                f'<div class="step-guide">'
                f'<div class="step-label">{title}</div>'
                f'{desc}</div>',
                unsafe_allow_html=True,
            )

    if st.button("▶️ Run Full ReAct Pipeline",
                 type="primary", use_container_width=True):
        with st.spinner("Constructing state graph and executing…"):
            ps = run_pipeline(df, sel_model, enabled)
        st.session_state.pipeline_state = ps
        st.session_state.history.append({
            "ts":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name":    ps.ds_name,
            "summary": ps.final_summary[:500],
        })
        st.rerun()

    # ── results ───────────────────────────────────────────────────────────
    if ps is not None:
        st.markdown("---")
        st.markdown("#### 📋 Agent Findings · ReAct Traces · Code Blocks")

        for k in ["quality", "stats", "corr", "viz"]:
            if k not in ps.agent_findings:
                continue
            a          = AGENTS[k]
            retries    = ps.retry_counts.get(k, 0)
            blks       = ps.code_blocks.get(k, [])
            used_model = ps.agent_models_used.get(k, AGENT_MODEL_LABELS.get(k, ""))
            badge = (
                f'<span class="badge badge-react">ReAct</span>'
                f'<span class="badge badge-memory">Memory</span>'
                f'<span class="badge badge-model">🤖 {AGENT_MODEL_LABELS.get(k, "")}</span>'
                + (f'<span class="badge badge-code">{len(blks)} code block(s)</span>'
                   if blks else "")
                + (f'<span class="badge badge-retry">+{retries} retr.</span>'
                   if retries else "")
            )
            with st.expander(
                f"{a['icon']} {a['name']} · {AGENT_MODEL_LABELS.get(k, '')}",
                expanded=False,
            ):
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(ps.agent_findings[k])

                if blks:
                    with st.expander("💻 Code Blocks Executed", expanded=False):
                        for i, b in enumerate(blks, 1):
                            st.markdown(
                                f"**Block {i}** · lang=`{b.language}` · "
                                f"{'✅ OK' if b.executed else '❌ ' + b.error}"
                            )
                            st.code(b.code, language=b.language)
                            if b.output:
                                st.caption(f"Output: {b.output[:200]}")

                if k in ps.react_traces:
                    with st.expander(
                        "🔍 ReAct Trace (Observe→Plan→Code→Execute→Rewrite)",
                        expanded=False,
                    ):
                        render_trace(ps.react_traces[k])

        orch_model_used = ps.agent_models_used.get("orchestrator", "DeepSeek V3")
        st.markdown(
            f"#### 🧠 Orchestrator — Final EDA Report  "
            f"<small style='color:#7c3aed'>({orch_model_used.split('/')[-1].replace(':free','')})</small>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="final-box">', unsafe_allow_html=True)
        st.markdown(ps.final_summary)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 📊 Evaluation Results  _(Days 13-15 Metrics)_")
        ev = ps.eval_results
        if "error" in ev:
            st.error(f"Eval error:\n```\n{ev['error']}\n```")
        elif ev:
            tcr = ev.get("task_completion_rate", 0)
            ca  = ev.get("code_accuracy", 1.0)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Task Completion", f"{tcr:.0%}")
            c2.metric("Retries",         ev.get("total_retries", 0))
            c3.metric("Code Accuracy",   f"{ca:.0%}")
            c4.metric("Memory Hit Rate", f"{ev.get('memory_hit_rate', 0):.0%}")
            c5.metric("LLM Calls",       ps.metrics.get("llm_calls", 0))

            st.markdown('<div class="eval-box">', unsafe_allow_html=True)
            ac  = ev.get("agent_completion", {})
            tbl = "| Agent | Model | Task Completed |\n|---|---|---|\n" + "\n".join(
                f"| {AGENTS.get(k, {'icon': '•', 'name': k})['icon']} "
                f"{AGENTS.get(k, {'name': k})['name']} "
                f"| {AGENT_MODEL_LABELS.get(k, '?')} "
                f"| {'✅ Yes' if v else '❌ No'} |"
                for k, v in ac.items()
            )
            st.markdown(tbl)
            st.markdown(ev.get("vs_baseline_note", ""))

            with st.expander(
                f"📝 Baseline — Single-Prompt EDA "
                f"({AGENT_MODEL_LABELS.get('baseline', 'Llama 3.1 8B')})"
            ):
                st.markdown(ev.get("baseline_summary", "Not available."))

            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🗺️ LangGraph — State Transition Log"):
            nodes  = ["quality", "stats", "corr", "viz", "orchestrator", "eval"]
            active = [n for n in nodes if n in enabled + ["orchestrator", "eval"]]
            st.markdown(" → ".join(
                f"`{n}` [{AGENT_MODEL_LABELS.get(n, '?')}]" for n in active
            ))
            node_statuses = "\n".join(
                f"- `{n}` ({AGENT_MODEL_LABELS.get(n, '?')}): "
                f"**{_node_css(ps.node_states.get(n, NodeState.PENDING))}**"
                for n in active
            )
            st.markdown(node_statuses)

        st.markdown("---")
        st.download_button(
            "📥 Export Full Report (.md)",
            data      = _export(ps).encode("utf-8"),
            file_name = f"eda_v5_{ps.ds_name.replace(' ', '_')}.md",
            mime      = "text/markdown",
            use_container_width=True,
        )


with right:
    st.markdown("#### 🔬 Dataset Info")
    c1, c2 = st.columns(2)
    c1.metric("Rows",    f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3, c4 = st.columns(2)
    c3.metric("Numeric",     len(_num(df)))
    c4.metric("Categorical", len(_cat(df)))
    st.metric("Missing", int(df.isna().sum().sum()))

    # ── Per-agent model assignment display ────────────────────────────────
    st.markdown("#### 🤖 Active Model Assignments")
    rows_html2 = "".join(
        f"<tr>"
        f"<td>{AGENTS[k]['icon']} {AGENTS[k]['name']}</td>"
        f"<td><span class='model-badge'>{AGENT_MODEL_LABELS[k]}</span></td>"
        f"<td style='color:#6b7280;font-size:.7rem'>"
        f"{'✅ ' + ps.agent_models_used.get(k,'') if ps and k in ps.agent_models_used else '⏳ pending'}"
        f"</td>"
        f"</tr>"
        for k in ["quality", "stats", "corr", "viz", "orchestrator"]
    )
    st.markdown(
        f"<table class='model-table'>"
        f"<thead><tr><th>Agent</th><th>Assigned</th><th>Used</th></tr></thead>"
        f"<tbody>{rows_html2}</tbody></table>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🛠️ Tool Scheduler")
    sched = ToolScheduler(df)
    st.caption(sched.summary())
    with st.expander("Per-Agent Tool Plans (Python/SQL/sklearn)"):
        t1, t2, t3, t4 = st.tabs(["Quality", "Stats", "Corr", "Viz"])
        with t1: st.json(sched.plan_for_quality())
        with t2: st.json(sched.plan_for_stats())
        with t3: st.json(sched.plan_for_corr())
        with t4: st.json(sched.plan_for_viz())

    st.markdown("#### 📐 Schema Validator")
    with st.expander("Live ToolInput Validation"):
        first_num = _num(df)[0] if _num(df) else "x"
        sample = ToolInput(
            tool_name="describe_column",
            params={"column": first_num},
            agent_id="stats",
        )
        ok_s, msg_s = sample.validate()
        st.json(asdict(sample))
        (st.success if ok_s else st.error)(msg_s)

        bad = ToolInput(tool_name="group_stats",
                        params={"groupby": "col"},
                        agent_id="corr")
        ok_b, msg_b = bad.validate()
        st.caption("Invalid example (missing 'target'):")
        (st.success if ok_b else st.error)(msg_b)

    st.markdown("#### 🧠 Cached Memory Keys")
    mem2 = _get_mem()
    ms2  = mem2.stats
    if ms2["keys"] > 0:
        st.caption(
            f"{ms2['keys']} keys cached · "
            f"{ms2['hits']} hits · {ms2['misses']} misses"
        )
        cached_names = [
            "df_describe", "quality_stats", "stats",
            "correlations", "viz_ctx", "orchestrator_report",
        ]
        for nm in cached_names:
            val = mem2.get(cur_sig, nm)
            st.markdown(
                f"- `{nm}`: {'✅ cached' if val is not None else '○ not yet'}"
            )
    else:
        st.caption("Memory is empty — run the pipeline to populate.")

    if ps is not None:
        st.markdown("---")
        st.markdown("#### 📈 Data Explorer")
        tabs = st.tabs(["Histogram", "Scatter", "Heatmap", "Outliers", "Missing", "Raw"])
        with tabs[0]: chart_hist(df)
        with tabs[1]: chart_scatter(df)
        with tabs[2]: chart_heatmap(df)
        with tabs[3]: chart_outliers(df)
        with tabs[4]: chart_missing(df)
        with tabs[5]: st.dataframe(df, use_container_width=True, height=380)
    else:
        st.info("Run the pipeline to unlock charts.")


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
Multi-Agent EDA System v5 · Per-Agent Model Assignment ·
Quality→Gemma 3 27B · Stats→DeepSeek R1 · Corr→DeepSeek R1 ·
Viz→Mistral 7B · Orchestrator→DeepSeek V3 · Baseline→Llama 3.1 8B<br>
ReAct Loop · Error-Correction Parser · LangGraph State Machine ·
Memory Store · Dynamic Tool Scheduler (Python / SQL / sklearn) ·
Pydantic Schemas · Evaluation Framework (5 Datasets)<br>
Made By: Eng Kirollos Ashraf · Eng Hossam Abdelmoniem · Eng Abduallah Rashed
</div>
""", unsafe_allow_html=True)
