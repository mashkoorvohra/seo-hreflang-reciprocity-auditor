# 🌐 Bulk Hreflang Reciprocity, Canonical & XML Sitemap Auditor

**Live App:** [https://seo-hreflang-reciprocity-auditor.streamlit.app/](https://seo-hreflang-reciprocity-auditor.streamlit.app/)
> **Install as an AI Agent Skill (Claude Code / Cursor / MCP):** Get the packaged agent skill for automated multilingual hreflang audits and XML sitemap repairs on [Agensi](https://agensi.io/skills/seo-hreflang-reciprocity-auditor).
> 
A free, browser-based technical SEO auditor for international websites. Validates reciprocal return tags, self-references, ISO 639-1/3166-1 language-region codes, cross-locale canonical conflicts, HTTP status/noindex blocks, and XML `xhtml:link` sitemaps—and compiles a repaired XML sitemap automatically.

## Key Capabilities
- **Full Cluster Graph Validation:** Detects missing return links, omitted self-referencing tags, duplicate locale codes, multiple `x-default` tags, and invalid codes (`en-UK` vs `en-GB`, underscore formatting `fr_FR`, region-only tags).
- **Cross-Signal Conflict Detection:** Crawls live HTML `<head>`, `<body>` placement errors, HTTP `Link` headers, redirect hops, non-self canonicals, and `noindex` directives.
- **Large-Scale XML Sitemap Support:** Parses `<sitemapindex>` files, uploaded `.xml` sitemaps, or pasted XML graphs in memory.
- **Auto-Generated Fix Sitemap:** Outputs normalized, reciprocal `<xhtml:link>` XML ready for deployment, plus optional AI-drafted developer Jira tickets.

## Tech Stack
`Python` • `Streamlit` • `BeautifulSoup4` • `Requests` • `Pandas` • `LiteLLM`

## Run Locally
```bash
git clone [https://github.com/mashkoorvohra/seo-hreflang-reciprocity-auditor.git](https://github.com/mashkoorvohra/seo-hreflang-reciprocity-auditor.git)
cd seo-hreflang-reciprocity-auditor
pip install -r requirements.txt
streamlit run app.py
