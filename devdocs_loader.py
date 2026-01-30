import asyncio
import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from helpers import clean_html, safe_print
from mcp_client import MCPServerClient
from rag_store import vs_add_documents


def load_devdocs_for_tech(mcp: MCPServerClient, tech: str, max_hits: Optional[int] = None) -> bool:
    try:
        search_res = mcp.call_tool("search_devdocs", {"doc_name": tech, "keyword": "introduction"})
    except Exception:
        search_res = None

    if not (search_res and isinstance(search_res, list) and len(search_res) > 0):
        try:
            search_res = mcp.call_tool("search_devdocs", {"doc_name": tech, "keyword": tech})
        except Exception:
            search_res = None

    if not (search_res and isinstance(search_res, list) and len(search_res) > 0):
        return False

    added = 0
    seen_paths = set()
    hits_iter = search_res if max_hits is None else search_res[:max_hits]

    for hit in hits_iter:
        slug = hit.get("doc_slug") or tech
        path = (hit.get("path") or "").split("#")[0]
        url = hit.get("url")
        if not path:
            continue
        key = f"{slug}::{path}"
        if key in seen_paths:
            continue
        seen_paths.add(key)

        html_content = mcp.call_tool("read_devdocs_page", {"doc_slug": slug, "path": path})
        if not html_content or not isinstance(html_content, str):
            continue
        if "не найдена" in html_content.lower():
            continue

        cleaned = clean_html(html_content)
        if len(cleaned) < 160:
            continue

        doc = Document(
            page_content=cleaned,
            metadata={"source": "devdocs", "tech": tech, "url": url, "path": path},
        )
        vs_add_documents([doc])
        added += 1

    return added > 0


def load_devdocs_for_tech_with_topics(
    mcp: MCPServerClient,
    tech: str,
    topics: List[str],
    max_hits: Optional[int] = None,
) -> bool:

    added = 0
    seen_paths = set()
    for topic in (topics or []):
        try:
            search_res = mcp.call_tool("search_devdocs", {"doc_name": tech, "keyword": topic})
        except Exception:
            search_res = None

        if not (search_res and isinstance(search_res, list) and len(search_res) > 0):
            try:
                search_res = mcp.call_tool("search_devdocs", {"doc_name": tech, "keyword": tech})
            except Exception:
                search_res = None

        if not (search_res and isinstance(search_res, list) and len(search_res) > 0):
            continue

        hits_iter = search_res if max_hits is None else search_res[:max_hits]
        for hit in hits_iter:
            slug = hit.get("doc_slug") or tech
            path = (hit.get("path") or "").split("#")[0]
            url = hit.get("url")
            if not path:
                continue
            key = f"{slug}::{path}"
            if key in seen_paths:
                continue
            seen_paths.add(key)

            html_content = mcp.call_tool("read_devdocs_page", {"doc_slug": slug, "path": path})
            if not html_content or not isinstance(html_content, str):
                continue
            if "не найдена" in html_content.lower():
                continue
            cleaned = clean_html(html_content)
            if len(cleaned) < 160:
                continue
            doc = Document(
                page_content=cleaned,
                metadata={"source": "devdocs", "tech": tech, "url": url, "path": path},
            )
            vs_add_documents([doc])
            added += 1
    return added > 0


async def background_load_other_techs(
    mcp: MCPServerClient,
    pending: List[str],
    loaded_techs: set,
    topics_map: Optional[Dict[str, List[str]]] = None,
):
    for tech in pending:
        if tech in loaded_techs:
            continue
        topics = (topics_map or {}).get(tech) if topics_map else None
        if topics:
            ok = await asyncio.to_thread(load_devdocs_for_tech_with_topics, mcp, tech, topics, 2)
        else:
            ok = await asyncio.to_thread(load_devdocs_for_tech, mcp, tech, 2)
        if ok:
            loaded_techs.add(tech)
            logging.info(f"[BG RAG] Loaded: {tech}")
        else:
            logging.info(f"[BG RAG] Not found / not loaded: {tech}")
