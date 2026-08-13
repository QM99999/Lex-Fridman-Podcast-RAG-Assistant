"""Streamlit helpers for managing API keys / model defaults from the UI.

Pages render a persistent "API keys" block (sidebar) via render_keys();
missing keys are filled in there and saved to the repo .env (bind-mounted
into the container), so testers configure keys once and CLI runs pick them up.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from retrieval import PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"
REQUIRED_KEYS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")
PLACEHOLDER_MARKERS = ("sk-REPLACE_WITH_YOUR", "REPLACE_WITH_YOUR")


def _key_is_set(key: str) -> bool:
    """A key counts as set only if non-empty and not a template placeholder."""
    val = os.environ.get(key, "").strip()
    if not val:
        return False
    low = val.lower()
    return not any(m.lower() in low for m in PLACEHOLDER_MARKERS)


def read_env() -> dict:
    out: dict = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def write_env(key: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    out, found = [], False
    for line in lines:
        if line.strip().startswith(key + "="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    os.environ[key] = value


def missing_keys() -> list[str]:
    """Only OpenAI is required for prompting; DeepSeek stays optional."""
    return ["OPENAI_API_KEY"] if not _key_is_set("OPENAI_API_KEY") else []


ADMIN_KEY = "ADMIN_PASSWORD"


def _admin_password() -> str:
    """Admin password: Streamlit secrets first (cloud), then local .env."""
    try:
        secret = st.secrets.get(ADMIN_KEY, "")
        if secret:
            return secret
    except Exception:  # noqa: BLE001 - secrets not available locally
        pass
    return read_env().get(ADMIN_KEY, "")


_ADMIN_LS = "lfqa_admin"


def _admin_ls_write(value: str) -> None:
    """Persist the admin flag in the browser's localStorage (survives refresh)."""
    try:
        components.html(
            f"<script>localStorage.setItem('{_ADMIN_LS}','{value}');</script>",
            height=0, key=f"admin_ls_write_{value}",
        )
    except Exception:  # noqa: BLE001 - best-effort
        pass


def _admin_ls_read() -> bool:
    """Read the admin flag from localStorage via a tiny component."""
    try:
        v = components.html(
            f"<script>const v=localStorage.getItem('{_ADMIN_LS}')||'';"
            "Streamlit.setComponentValue(v);</script>",
            height=0, key="admin_ls_probe",
        )
        return v == "1"
    except Exception:  # noqa: BLE001 - best-effort
        return False


def is_admin() -> bool:
    if st.session_state.get("admin_ok"):
        return True
    # Survive a page refresh: URL marker (same page) or localStorage (any page).
    if st.query_params.get("admin") == "1" or _admin_ls_read():
        st.session_state["admin_ok"] = True
        return True
    return False


def require_admin() -> bool:
    """Gate for admin-only pages. Returns True when unlocked, else renders the
    login form (and lets the owner set an initial password on first run)."""
    if is_admin():
        return True
    st.subheader("\U0001F512 Admin area")
    configured = bool(_admin_password())
    if not configured:
        st.warning(
            "No admin password configured yet. Set `ADMIN_PASSWORD` in `.env` "
            "(or Streamlit `secrets`) to enable the lock - or set one below."
        )
        new_pwd = st.text_input("Set admin password", type="password", key="admin_set_pwd")
        if st.button("Save admin password"):
            if len(new_pwd) >= 6:
                write_env(ADMIN_KEY, new_pwd)
                _persist_admin_unlock()
                st.rerun()
            else:
                st.error("Password must be at least 6 characters.")
        return False
    pwd = st.text_input("Admin password", type="password", key="admin_pwd")
    if st.button("Unlock"):
        if pwd == _admin_password():
            _persist_admin_unlock()
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


def _persist_admin_unlock() -> None:
    st.session_state["admin_ok"] = True
    st.query_params["admin"] = "1"
    _admin_ls_write("1")


def render_admin_sidebar() -> None:
    """Sidebar widget: lock status + login/logout for admin pages."""
    with st.sidebar.expander("\U0001F512 Admin", expanded=not is_admin()):
        if is_admin():
            st.success("Admin unlocked (Monitoring & Pipeline visible).")
            if st.button("Lock"):
                st.session_state.pop("admin_ok", None)
                st.query_params.pop("admin", None)
                _admin_ls_write("")
                st.rerun()
        else:
            st.caption("Unlock to view Monitoring / Pipeline pages.")
            if not _admin_password():
                st.info("No password set yet - open Monitoring or Pipeline "
                        "to configure the first admin password.")
            pwd = st.text_input("Password", type="password", key="admin_sidebar_pwd")
            if st.button("Unlock admin"):
                if pwd and pwd == _admin_password():
                    _persist_admin_unlock()
                    st.rerun()
                else:
                    st.error("Wrong password.")


def render_keys() -> None:
    """Persistent sidebar block: key status + edit inputs for both keys.

    OpenAI is required (warns when missing); DeepSeek is optional. Set keys
    show as "set" with an Edit button; editing replaces the stored value.
    """
    missing = missing_keys()
    with st.sidebar.expander("API keys", expanded=bool(missing)):
        editing = any(st.session_state.get(f"edit_{k}") for k in REQUIRED_KEYS)
        for k in REQUIRED_KEYS:
            has = _key_is_set(k)
            if has and not st.session_state.get(f"edit_{k}"):
                c1, c2 = st.columns([3, 1])
                c1.success(f"{k}: set")
                if c2.button("Edit", key=f"btn_{k}"):
                    st.session_state[f"edit_{k}"] = True
            else:
                st.text_input(
                    f"{k} (password)", type="password",
                    key=f"key_input_{k}",
                    placeholder="sk-..." if not has else "leave blank to keep current",
                )
        if missing:
            st.warning(
                "\U0001F512 **Security:** keys are saved only to your local "
                "`.env` file and never leave this machine. As a tester, please "
                "generate a **NEW key** for this test and **delete it once you "
                "are done**."
            )
        if missing or editing:
            if st.button("Save API keys"):
                saved = False
                for k in REQUIRED_KEYS:
                    v = st.session_state.get(f"key_input_{k}", "").strip()
                    if v:
                        write_env(k, v)
                        saved = True
                for k in REQUIRED_KEYS:
                    st.session_state.pop(f"edit_{k}", None)
                if saved:
                    st.rerun()
    if missing:
        st.warning(
            "API keys missing - fill them in the sidebar to enable LLM calls. "
            "OpenAI is required; DeepSeek is optional (for deepseek-* models)."
        )
