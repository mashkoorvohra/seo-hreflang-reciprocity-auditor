import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
import litellm
import re
import html as html_lib

st.set_page_config(page_title="Bulk Hreflang & Canonical Auditor", layout="wide")
st.title("Bulk Hreflang Reciprocity, Canonical & Sitemap Auditor")
st.markdown(
    "Validates **reciprocal return tags, self-references, ISO 639-1/3166-1 codes, canonical overrides, "
    "redirect/noindex conflicts, and XML `xhtml:link` sitemaps**—runs **100% free locally** with auto-generated fixes."
)

# --- ISO 639-1 Language & ISO 3166-1 Alpha-2 Region Reference Sets ---
ISO_639_1 = {
    "aa","ab","ae","af","ak","am","an","ar","as","av","ay","az","ba","be","bg","bh","bi","bm","bn","bo","br","bs",
    "ca","ce","ch","co","cr","cs","cu","cv","cy","da","de","dv","dz","ee","el","en","eo","es","et","eu","fa","ff",
    "fi","fj","fo","fr","fy","ga","gd","gl","gn","gu","gv","ha","he","hi","ho","hr","ht","hu","hy","hz","ia","id",
    "ie","ig","ii","ik","io","is","it","iu","ja","jv","ka","kg","ki","kj","kk","kl","km","kn","ko","kr","ks","ku",
    "kv","kw","ky","la","lb","lg","li","ln","lo","lt","lu","lv","mg","mh","mi","mk","ml","mn","mr","ms","mt","my",
    "na","nb","nd","ne","ng","nl","nn","no","nr","nv","ny","oc","oj","om","or","os","pa","pi","pl","ps","pt","qu",
    "rm","rn","ro","ru","rw","sa","sc","sd","se","sg","si","sk","sl","sm","sn","so","sq","sr","ss","st","su","sv",
    "sw","ta","te","tg","th","ti","tk","tl","tn","to","tr","ts","tt","tw","ty","ug","uk","ur","uz","ve","vi","vo",
    "wa","wo","xh","yi","yo","za","zh","zu"
}

ISO_3166_1 = {
    "AD","AE","AF","AG","AI","AL","AM","AO","AQ","AR","AS","AT","AU","AW","AX","AZ","BA","BB","BD","BE","BF","BG",
    "BH","BI","BJ","BL","BM","BN","BO","BQ","BR","BS","BT","BV","BW","BY","BZ","CA","CC","CD","CF","CG","CH","CI",
    "CK","CL","CM","CN","CO","CR","CU","CV","CW","CX","CY","CZ","DE","DJ","DK","DM","DO","DZ","EC","EE","EG","EH",
    "ER","ES","ET","FI","FJ","FK","FM","FO","FR","GA","GB","GD","GE","GF","GG","GH","GI","GL","GM","GN","GP","GQ",
    "GR","GS","GT","GU","GW","GY","HK","HM","HN","HR","HT","HU","ID","IE","IL","IM","IN","IO","IQ","IR","IS","IT",
    "JE","JM","JO","JP","KE","KG","KH","KI","KM","KN","KP","KR","KW","KY","KZ","LA","LB","LC","LI","LK","LR","LS",
    "LT","LU","LV","LY","MA","MC","MD","ME","MF","MG","MH","MK","ML","MM","MN","MO","MP","MQ","MR","MS","MT","MU",
    "MV","MW","MX","MY","MZ","NA","NC","NE","NF","NG","NI","NL","NO","NP","NR","NU","NZ","OM","PA","PE","PF","PG",
    "PH","PK","PL","PM","PN","PR","PS","PT","PW","PY","QA","RE","RO","RS","RU","RW","SA","SB","SC","SD","SE","SG",
    "SH","SI","SJ","SK","SL","SM","SN","SO","SR","SS","ST","SV","SX","SY","SZ","TC","TD","TF","TG","TH","TJ","TK",
    "TL","TM","TN","TO","TR","TT","TV","TW","TZ","UA","UG","UM","US","UY","UZ","VA","VC","VE","VG","VI","VN","VU",
    "WF","WS","YE","YT","ZA","ZM","ZW","419"
}

VALID_SCRIPTS = {"Hans", "Hant", "Latn", "Cyrl", "Arab"}

# --- Optional Live AI Model Fetcher ---
@st.cache_data(ttl=3600)
def fetch_live_models(provider_name: str, key: str) -> list:
    defaults = {
        "Google Gemini": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        "Anthropic Claude": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"],
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
        "Groq / OpenRouter": ["groq/llama-3.3-70b-versatile", "openrouter/anthropic/claude-3.5-sonnet"]
    }
    if not key:
        return defaults.get(provider_name, [])
    try:
        if provider_name == "Google Gemini":
            r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=6)
            models = [
                m["name"].replace("models/", "")
                for m in r.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            return models[::-1][:12] if models else defaults["Google Gemini"]
        elif provider_name == "Anthropic Claude":
            r = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                timeout=6
            )
            models = [m["id"] for m in r.json().get("data", [])]
            return models if models else defaults["Anthropic Claude"]
        elif provider_name == "OpenAI":
            r = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=6
            )
            models = [m["id"] for m in r.json().get("data", []) if any(v in m["id"] for v in ["gpt-", "o1", "o3", "o4"])]
            return sorted(models, reverse=True)[:15] if models else defaults["OpenAI"]
    except Exception:
        pass
    return defaults.get(provider_name, [])

# --- Sidebar Controls ---
st.sidebar.header("Audit & AI Settings")
provider = st.sidebar.selectbox(
    "AI Ticket Generator (Optional)",
    ["Free Local Audit Only (0 API Credits)", "Google Gemini", "Anthropic Claude", "OpenAI", "Groq / OpenRouter"]
)

api_key = ""
selected_model = ""
if provider != "Free Local Audit Only (0 API Credits)":
    api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")
    models = fetch_live_models(provider, api_key)
    if provider == "Groq / OpenRouter":
        selected_model = st.sidebar.text_input("Model ID", value=models[0])
    else:
        selected_model = st.sidebar.selectbox("Select Model (Auto-Updated)", models)
        prefix_map = {"Google Gemini": "gemini/", "Anthropic Claude": "anthropic/", "OpenAI": "openai/"}
        if not selected_model.startswith(prefix_map[provider]):
            selected_model = prefix_map[provider] + selected_model

max_live_urls = st.sidebar.slider(
    "Max Live Target URLs to Fetch (Live Mode)",
    min_value=5,
    max_value=80,
    value=25,
    step=5,
    help="Caps live HTTP concurrency to protect shared cloud IPs from target rate limits. XML Sitemap Graph mode handles large sitemaps in memory."
)
check_live_from_sitemap = st.sidebar.checkbox(
    "Also probe HTTP status & canonicals on Sitemap URLs (up to cap)",
    value=False
)

# --- Validation Helpers ---
def validate_hreflang_code(code: str) -> tuple[bool, str, str]:
    raw = (code or "").strip()
    if not raw:
        return False, "Empty hreflang attribute", raw
    if raw.lower() == "x-default":
        if raw != "x-default":
            return False, "Use lowercase 'x-default'", "x-default"
        return True, "Valid", "x-default"
    if "_" in raw:
        fixed = raw.replace("_", "-")
        parts = fixed.split("-")
        suggested = f"{parts[0].lower()}-{parts[1].upper()}" if len(parts) == 2 else fixed.lower()
        return False, f"Underscore invalid (use hyphen: '{suggested}')", suggested

    parts = raw.split("-")
    lang = parts[0]
    if lang.lower() not in ISO_639_1:
        if lang.upper() in ISO_3166_1:
            return False, f"Region-only code '{raw}' invalid (language required first)", raw
        return False, f"Invalid ISO 639-1 language code '{lang}'", raw

    if len(parts) == 1:
        if lang != lang.lower():
            return False, f"Language must be lowercase ('{lang.lower()}')", lang.lower()
        if lang == "uk":
            return True, "Valid (Note: 'uk' = Ukrainian; use 'en-GB' for UK English)", "uk"
        return True, "Valid", lang

    if len(parts) == 2:
        second = parts[1]
        if second in VALID_SCRIPTS or second.title() in VALID_SCRIPTS:
            suggested = f"{lang.lower()}-{second.title()}"
            if raw != suggested:
                return False, f"Script casing should be '{suggested}'", suggested
            return True, "Valid", suggested
        if second.upper() == "UK":
            return False, "Invalid region 'UK' (use 'GB' per ISO 3166-1)", f"{lang.lower()}-GB"
        if second.upper() not in ISO_3166_1:
            return False, f"Invalid ISO 3166-1 region code '{second}'", raw
        suggested = f"{lang.lower()}-{second.upper()}"
        if raw != suggested:
            return False, f"Case mismatch (use '{suggested}')", suggested
        return True, "Valid", suggested

    if len(parts) == 3:
        script, region = parts[1], parts[2]
        if script.title() in VALID_SCRIPTS and region.upper() in ISO_3166_1:
            suggested = f"{lang.lower()}-{script.title()}-{region.upper()}"
            if raw != suggested:
                return False, f"Casing mismatch (use '{suggested}')", suggested
            return True, "Valid", suggested

    return False, f"Malformed hreflang tag '{raw}'", raw


def parse_xml_sitemap(xml_text: str) -> tuple[dict, list]:
    """Parses XML sitemap xhtml:link annotations into {source_url: [(hreflang, target_url)]}."""
    graph = {}
    errors = []
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError as e:
        return {}, [f"XML Parse Error: {e}"]

    if "sitemapindex" in root.tag.lower():
        locs = [el.text.strip() for el in root.iter() if el.tag.lower().endswith("loc") and el.text]
        return {"__SITEMAP_INDEX__": locs}, []

    for url_el in root.iter():
        if url_el.tag.lower().endswith("}url") or url_el.tag == "url":
            loc = None
            links = []
            for child in url_el:
                cname = child.tag.lower()
                if cname.endswith("loc") and child.text:
                    loc = child.text.strip()
                elif cname.endswith("link") and child.attrib.get("rel", "").lower() == "alternate":
                    hl = child.attrib.get("hreflang", "").strip()
                    href = child.attrib.get("href", "").strip()
                    if hl or href:
                        links.append((hl, href))
            if loc:
                graph[loc] = links
    if not graph:
        errors.append("No <url> entries found in XML.")
    return graph, errors


def fetch_url_signals(url: str) -> dict:
    """Fetches one live URL with redirect inspection, header Link parsing, head/body hreflang, canonical, and robots."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HreflangAuditBot/1.0)"}
    result = {
        "url": url,
        "initial_status": None,
        "final_status": None,
        "final_url": url,
        "canonical": None,
        "noindex": False,
        "head_links": [],
        "body_links": [],
        "http_links": [],
        "error": None
    }
    try:
        resp = requests.get(url, headers=headers, timeout=9, allow_redirects=True)
        result["initial_status"] = resp.history[0].status_code if resp.history else resp.status_code
        result["final_status"] = resp.status_code
        result["final_url"] = resp.url

        if "noindex" in resp.headers.get("X-Robots-Tag", "").lower():
            result["noindex"] = True

        link_header = resp.headers.get("Link", "")
        if link_header:
            for part in link_header.split(","):
                if "alternate" in part.lower() and "hreflang=" in part.lower():
                    u_m = re.search(r"<([^>]+)>", part)
                    h_m = re.search(r'hreflang=["\']?([^"\';\s]+)', part, re.I)
                    if u_m and h_m:
                        result["http_links"].append((h_m.group(1).strip(), u_m.group(1).strip()))

        if "text/html" in resp.headers.get("Content-Type", "").lower():
            soup = BeautifulSoup(resp.text, "html.parser")
            can_tag = soup.find("link", rel=lambda r: r and "canonical" in [x.lower() for x in (r if isinstance(r, list) else [r])])
            if can_tag and can_tag.get("href"):
                result["canonical"] = urljoin(resp.url, can_tag["href"].strip())

            for meta in soup.find_all("meta", attrs={"name": re.compile(r"^(robots|googlebot)$", re.I)}):
                if "noindex" in (meta.get("content") or "").lower():
                    result["noindex"] = True

            head = soup.find("head")
            body = soup.find("body")
            if head:
                for lnk in head.find_all("link", attrs={"hreflang": True}):
                    rels = [x.lower() for x in (lnk.get("rel") or [])]
                    if "alternate" in rels:
                        result["head_links"].append((lnk.get("hreflang", "").strip(), lnk.get("href", "").strip()))
            if body:
                for lnk in body.find_all("link", attrs={"hreflang": True}):
                    rels = [x.lower() for x in (lnk.get("rel") or [])]
                    if "alternate" in rels:
                        result["body_links"].append((lnk.get("hreflang", "").strip(), lnk.get("href", "").strip()))
    except Exception as e:
        result["error"] = str(e)
    return result


def audit_cluster_graph(graph: dict, live_meta: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Audits full hreflang graph for return tags, self-reference, syntax, duplicates, canonicals, and generates fixed XML."""
    issue_rows = []
    summary_rows = []
    live_meta = live_meta or {}
    all_sources = set(graph.keys())
    healthy_clusters = {}

    for src, links in graph.items():
        issues = []
        seen_codes = {}
        x_default_count = 0
        has_self_ref = False
        parsed_targets = {}

        meta = live_meta.get(src, {})
        if meta.get("error"):
            issues.append(f"Fetch Error: {meta['error']}")
        else:
            if meta.get("initial_status") and meta["initial_status"] != 200:
                issues.append(f"Non-200 HTTP Status ({meta['initial_status']})")
            if meta.get("noindex"):
                issues.append("Page is noindex (invalidates hreflang)")
            can = meta.get("canonical")
            if can and can.rstrip("/") != src.rstrip("/"):
                issues.append(f"Non-self canonical -> {can} (overrides hreflang)")
            elif can and can != src:
                issues.append(f"Canonical trailing-slash/format mismatch ({can})")
            if meta.get("body_links"):
                issues.append(f"{len(meta['body_links'])} hreflang tag(s) placed in <body> (ignored by crawlers)")
            modes_used = sum([bool(meta.get("head_links")), bool(meta.get("http_links"))])
            if modes_used > 1:
                issues.append("Mixed HTML <head> and HTTP Link header declarations")

        if not links:
            issues.append("No hreflang annotations found")

        for hl_raw, href in links:
            if not href.startswith(("http://", "https://")):
                issues.append(f"Relative/invalid URL '{href}' for '{hl_raw}' (must be absolute)")

            is_valid_code, code_msg, suggested_code = validate_hreflang_code(hl_raw)
            if not is_valid_code:
                issues.append(f"Syntax error '{hl_raw}': {code_msg}")
            elif "Note:" in code_msg:
                issues.append(f"Warning '{hl_raw}': {code_msg}")

            norm_code = suggested_code if suggested_code else hl_raw
            if norm_code.lower() == "x-default":
                x_default_count += 1
                if x_default_count > 1:
                    issues.append("Multiple x-default tags on same page")
            if norm_code in seen_codes:
                issues.append(f"Duplicate hreflang declaration '{norm_code}'")
            seen_codes[norm_code] = href

            if href.rstrip("/") == src.rstrip("/"):
                has_self_ref = True
                if href != src:
                    issues.append(f"Self-ref trailing-slash mismatch ({href} vs {src})")

            if href != src:
                target_links = graph.get(href)
                if target_links is None:
                    alt_match = next((k for k in all_sources if k.rstrip("/") == href.rstrip("/")), None)
                    if alt_match:
                        issues.append(f"Target URL slash mismatch ({href} vs {alt_match})")
                        target_links = graph[alt_match]
                    else:
                        issues.append(f"Target '{href}' ({hl_raw}) was not crawled / missing from cluster")

                if target_links is not None:
                    target_hrefs = [t_href.rstrip("/") for _, t_href in target_links]
                    if src.rstrip("/") not in target_hrefs:
                        issues.append(f"Missing return tag from {href} ({hl_raw}) back to {src}")

                t_meta = live_meta.get(href, {})
                if t_meta.get("initial_status") and t_meta["initial_status"] != 200:
                    issues.append(f"Points to non-200 target ({t_meta['initial_status']}): {href}")
                if t_meta.get("noindex"):
                    issues.append(f"Points to noindex target: {href}")

            parsed_targets[norm_code] = href

        if links and not has_self_ref:
            issues.append("Missing self-referencing hreflang tag")
        if links and "x-default" not in {c.lower() for c in seen_codes}:
            issues.append("Missing x-default fallback (recommended)")

        severity = "🔴 Critical" if any(
            k in " | ".join(issues)
            for k in ["Missing return", "Missing self", "canonical", "noindex", "Syntax", "Non-200", "<body>", "Duplicate", "Multiple"]
        ) else ("🟠 Warning" if issues else "🟢 Valid")

        summary_rows.append({
            "URL": src,
            "Status": severity,
            "Alternates Declared": len(links),
            "Issue Count": len(issues),
            "Issues Detected": " ; ".join(issues) if issues else "All reciprocity, syntax, and indexability checks passed"
        })
        for iss in issues:
            issue_rows.append({"URL": src, "Severity": severity, "Issue": iss})

        if parsed_targets:
            clean_map = {k: v for k, v in parsed_targets.items() if v.startswith(("http://", "https://"))}
            if clean_map:
                healthy_clusters[src] = clean_map

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
        '        xmlns:xhtml="http://www.w3.org/1999/xhtml">'
    ]
    for src, cmap in healthy_clusters.items():
        merged = dict(cmap)
        for _, other_map in healthy_clusters.items():
            if src in other_map.values():
                for code, u in other_map.items():
                    if code not in merged:
                        merged[code] = u
        xml_lines.append("  <url>")
        xml_lines.append(f"    <loc>{html_lib.escape(src)}</loc>")
        for code, u in sorted(merged.items()):
            xml_lines.append(f'    <xhtml:link rel="alternate" hreflang="{html_lib.escape(code)}" href="{html_lib.escape(u)}" />')
        xml_lines.append("  </url>")
    xml_lines.append("</urlset>")

    return pd.DataFrame(summary_rows), pd.DataFrame(issue_rows), "\n".join(xml_lines)


# --- Neutral Synthetic Demo XML (With Realistic Planted Errors) ---
DEMO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
  <url>
    <loc>https://example.com/en-us/pricing/</loc>
    <xhtml:link rel="alternate" hreflang="en-US" href="https://example.com/en-us/pricing/" />
    <xhtml:link rel="alternate" hreflang="en-UK" href="https://example.com/en-gb/pricing/" />
    <xhtml:link rel="alternate" hreflang="fr-FR" href="https://example.com/fr/tarifs/" />
    <xhtml:link rel="alternate" hreflang="de-DE" href="https://example.com/de/preise/" />
    <xhtml:link rel="alternate" hreflang="x-default" href="https://example.com/en-us/pricing/" />
  </url>
  <url>
    <loc>https://example.com/en-gb/pricing/</loc>
    <xhtml:link rel="alternate" hreflang="en-US" href="https://example.com/en-us/pricing/" />
    <xhtml:link rel="alternate" hreflang="en-GB" href="https://example.com/en-gb/pricing/" />
    <xhtml:link rel="alternate" hreflang="fr-FR" href="https://example.com/fr/tarifs/" />
    <xhtml:link rel="alternate" hreflang="de-DE" href="https://example.com/de/preise/" />
    <xhtml:link rel="alternate" hreflang="x-default" href="https://example.com/en-us/pricing/" />
  </url>
  <url>
    <loc>https://example.com/fr/tarifs/</loc>
    <xhtml:link rel="alternate" hreflang="en-US" href="https://example.com/en-us/pricing/" />
    <xhtml:link rel="alternate" hreflang="en-GB" href="https://example.com/en-gb/pricing/" />
    <xhtml:link rel="alternate" hreflang="x-default" href="https://example.com/en-us/pricing/" />
  </url>
  <url>
    <loc>https://example.com/de/preise/</loc>
    <xhtml:link rel="alternate" hreflang="en-US" href="https://example.com/en-us/pricing/" />
    <xhtml:link rel="alternate" hreflang="en-GB" href="https://example.com/en-gb/pricing/" />
    <xhtml:link rel="alternate" hreflang="fr_FR" href="https://example.com/fr/tarifs/" />
    <xhtml:link rel="alternate" hreflang="de-DE" href="https://example.com/de/preise/" />
    <xhtml:link rel="alternate" hreflang="x-default" href="https://example.com/en-us/pricing/" />
  </url>
</urlset>"""

# --- Universal Input Modes ---
input_mode = st.radio(
    "Choose Audit Input Source:",
    [
        "Neutral Demo XML Sitemap (Instant Test)",
        "XML Sitemap URL / File Upload / Paste XML (Large-Scale Graph Audit)",
        "Bulk Live URL List / CSV / Excel (Crawls HTML Head, HTTP Headers & Canonicals)"
    ],
    horizontal=False
)

graph_to_audit = {}
live_metadata = {}

if input_mode == "Neutral Demo XML Sitemap (Instant Test)":
    with st.expander("Preview Demo XML Input (Contains Planted Bugs: en-UK, fr_FR, Missing Return & Self-Ref)"):
        st.code(DEMO_XML, language="xml")
    if st.button("Run Demo Hreflang Audit", type="primary"):
        graph_to_audit, _ = parse_xml_sitemap(DEMO_XML)

elif input_mode == "XML Sitemap URL / File Upload / Paste XML (Large-Scale Graph Audit)":
    sm_tab1, sm_tab2, sm_tab3 = st.tabs(["Sitemap URL", "Upload .xml File", "Paste Raw XML"])
    xml_source_text = ""
    with sm_tab1:
        sm_url = st.text_input("Enter XML Sitemap URL:", placeholder="https://example.com/sitemap.xml")
    with sm_tab2:
        sm_file = st.file_uploader("Upload XML Sitemap", type=["xml"])
    with sm_tab3:
        sm_paste = st.text_area("Paste XML Sitemap Content:", height=160)

    if st.button("Audit XML Sitemap Hreflang Graph", type="primary"):
        if sm_file:
            xml_source_text = sm_file.read().decode("utf-8", errors="ignore")
        elif sm_paste.strip():
            xml_source_text = sm_paste
        elif sm_url.strip():
            try:
                r = requests.get(sm_url.strip(), headers={"User-Agent": "Mozilla/5.0 (compatible; HreflangAuditBot/1.0)"}, timeout=12)
                r.raise_for_status()
                xml_source_text = r.text
            except Exception as e:
                st.error(f"Could not fetch sitemap URL: {e}")
                st.stop()
        else:
            st.warning("Provide a Sitemap URL, upload a .xml file, or paste XML text.")
            st.stop()

        graph_to_audit, errs = parse_xml_sitemap(xml_source_text)
        if "__SITEMAP_INDEX__" in graph_to_audit:
            child_sitemaps = graph_to_audit["__SITEMAP_INDEX__"][:5]
            st.info(f"Detected Sitemap Index—fetching first {len(child_sitemaps)} child sitemaps...")
            combined = {}
            for cs in child_sitemaps:
                try:
                    cr = requests.get(cs, timeout=10)
                    cg, _ = parse_xml_sitemap(cr.text)
                    combined.update(cg)
                except Exception:
                    continue
            graph_to_audit = combined
        elif errs:
            st.error(" | ".join(errs))
            st.stop()

        if check_live_from_sitemap and graph_to_audit:
            sample_urls = list(graph_to_audit.keys())[:max_live_urls]
            with st.spinner(f"Probing live HTTP status, canonicals, and noindex on {len(sample_urls)} URLs..."):
                with ThreadPoolExecutor(max_workers=8) as ex:
                    for res in ex.map(fetch_url_signals, sample_urls):
                        live_metadata[res["url"]] = res

else:
    url_source = st.radio("URL Input Method:", ["Paste Seed URLs (One per line)", "Upload CSV / Excel (.csv / .xlsx)"], horizontal=True)
    seed_urls = []
    if url_source == "Paste Seed URLs (One per line)":
        raw_urls = st.text_area(
            "Paste Target URLs (The tool automatically discovers & crawls declared hreflang alternates):",
            placeholder="https://example.com/en-us/\nhttps://example.com/en-gb/\nhttps://example.com/fr/",
            height=130
        )
        seed_urls = list(dict.fromkeys([u.strip() for u in raw_urls.splitlines() if u.strip().startswith("http")]))
    else:
        up = st.file_uploader("Upload CSV or Excel with a URL column", type=["csv", "xlsx"])
        if up:
            up_df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            cols = list(up_df.columns)
            guess_idx = next((i for i, c in enumerate(cols) if any(k in str(c).lower() for k in ["url", "address", "page", "loc"])), 0)
            ucol = st.selectbox("Select URL Column:", cols, index=guess_idx)
            seed_urls = list(dict.fromkeys([u.strip() for u in up_df[ucol].dropna().astype(str) if u.strip().startswith("http")]))

    if st.button("Crawl & Validate Live Hreflang Clusters", type="primary"):
        if not seed_urls:
            st.warning("Please enter at least one valid URL starting with http:// or https://")
            st.stop()

        seed_urls = seed_urls[:max_live_urls]
        with st.spinner(f"Step 1/2: Fetching {len(seed_urls)} seed URLs (HTML <head>, HTTP Link headers, Canonicals)..."):
            with ThreadPoolExecutor(max_workers=8) as ex:
                for res in ex.map(fetch_url_signals, seed_urls):
                    live_metadata[res["url"]] = res
                    combined_links = res["head_links"] if res["head_links"] else res["http_links"]
                    graph_to_audit[res["url"]] = combined_links

        discovered = set()
        for links in graph_to_audit.values():
            for _, href in links:
                if href.startswith("http") and href not in live_metadata:
                    discovered.add(href)

        remaining_budget = max(0, max_live_urls - len(live_metadata))
        to_fetch = list(discovered)[:remaining_budget]
        if to_fetch:
            with st.spinner(f"Step 2/2: Fetching {len(to_fetch)} discovered alternate URLs to verify reciprocal return tags..."):
                with ThreadPoolExecutor(max_workers=8) as ex:
                    for res in ex.map(fetch_url_signals, to_fetch):
                        live_metadata[res["url"]] = res
                        combined_links = res["head_links"] if res["head_links"] else res["http_links"]
                        graph_to_audit[res["url"]] = combined_links

# --- Render Audit Results & Exports ---
if graph_to_audit:
    summary_df, issues_df, fixed_xml = audit_cluster_graph(graph_to_audit, live_metadata)

    total_urls = len(summary_df)
    critical_urls = int((summary_df["Status"] == "🔴 Critical").sum())
    valid_urls = int((summary_df["Status"] == "🟢 Valid").sum())
    total_issues = len(issues_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("URLs Audited", total_urls)
    c2.metric("Critical Failures", critical_urls)
    c3.metric("Clean URLs", valid_urls)
    c4.metric("Total Signals Flagged", total_issues)

    tab1, tab2, tab3 = st.tabs(["URL Audit Matrix", "Detailed Error Log", "Auto-Generated Fix (XML Sitemap)"])

    with tab1:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Full Audit CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="hreflang_audit_report.csv",
            mime="text/csv"
        )

    with tab2:
        if issues_df.empty:
            st.success("Zero hreflang, reciprocity, or canonical issues detected!")
        else:
            st.dataframe(issues_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Issue-Level CSV",
                data=issues_df.to_csv(index=False).encode("utf-8"),
                file_name="hreflang_issues_breakdown.csv",
                mime="text/csv"
            )

    with tab3:
        st.markdown("Normalized ISO codes (`en-UK` -> `en-GB`, `fr_FR` -> `fr-FR`) and merged reciprocal cluster links ready for deployment:")
        st.code(fixed_xml[:4000] + ("\n<!-- Truncated preview; download full XML below -->" if len(fixed_xml) > 4000 else ""), language="xml")
        st.download_button(
            "Download Repaired Hreflang XML Sitemap",
            data=fixed_xml.encode("utf-8"),
            file_name="fixed_hreflang_sitemap.xml",
            mime="application/xml"
        )

    if provider != "Free Local Audit Only (0 API Credits)" and not issues_df.empty:
        st.subheader("AI Engineering Remediation Ticket")
        if not api_key:
            st.warning(f"Paste your {provider} API key in the sidebar to auto-draft a developer Jira ticket.")
        else:
            with st.spinner(f"Drafting technical ticket via {selected_model}..."):
                sample_errs = issues_df.head(20).to_dict(orient="records")
                prompt = (
                    f"Act as a Principal International SEO Architect. Summarize these hreflang/canonical audit errors "
                    f"into a concise Jira ticket for developers with Root Cause, Impact on Google Indexing, and Exact Fix Steps:\n{sample_errs}"
                )
                resp = litellm.completion(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=api_key
                )
                st.markdown(resp.choices[0].message.content)
