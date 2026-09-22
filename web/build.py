"""Build the production site from the artifact source.

`index.html` is written for the artifact platform, which wraps it in its own HTML skeleton.
A web server needs the complete document plus the files crawlers, social networks and language
models look for. This script writes into `dist/`:

  index.html        the page with doctype, <head> metadata, icons, Open Graph and JSON-LD
  robots.txt        crawler access and the sitemap location
  sitemap.xml       the page URL with its last modification date
  llms.txt          an index for language models (llmstxt.org format)
  llms-full.txt     the full article as plain Markdown, for language models
  site.webmanifest  name, colours and icons for browsers and home screens

Images (favicons, share images) are produced by `make_images.py`; run it first.
Site identity (URL, site name, organisation, author, official profiles, analytics id) lives in
`site.json`, so the canonical, Open Graph, JSON-LD and sitemap always agree with each other and
with where the page is really served.

Usage:  python build.py            (values from site.json)
        python build.py --url https://leuner.es/research/lnr-2026-01/
"""
import argparse
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "index.html"
DIST = HERE / "dist"
CONFIG = HERE / "site.json"

TITLE_LONG = "Construir una dendrita dentro de un modelo de lenguaje"
SUBTITLE = "Leuner: una inteligencia emocional artificial hecha de partes diminutas, para acompañar"
DESCRIPTION = ("Leuner (GestasAI): una dendrita artificial dentro de Gemma 4. Primer hito medido y "
               "programa de investigación en inteligencia emocional para acompañar.")
REPORT_ID = "LNR-2026-01"
VERSION = "0.2"
PUBLISHED = "2026-09-22"
MODIFIED = "2026-09-22"
FAQ = [
    ("¿Qué es Leuner?",
     "Un programa de investigación de GestasAI que construye una dendrita artificial dentro de Gemma 4, "
     "un modelo de lenguaje abierto, siguiendo la organización de la dendrita biológica: recibir, decidir "
     "en local, mirarse en el resultado, moderar y consolidar."),
    ("¿Qué se ha demostrado?",
     "Que un vector de 6 KB extraído del propio modelo (una proteína) e inyectado en su capa 20 activa una "
     "capacidad sin ejemplos: la traducción pasa del 0 % al 90 % de acierto. Es la primera de las cinco "
     "fases de la dendrita."),
    ("¿Para qué servirá?",
     "Para una inteligencia emocional orientada a acompañar y, cuando el modelo esté completo, para apoyar "
     "a personas con problemas neurológicos en estadio uno y al inicio del estadio dos, sin salir de su "
     "dispositivo."),
]


# ─── Cabecera ───────────────────────────────────────────────────────────────
def analytics(measurement_id):
    """Google tag (gtag.js), only when a measurement id is configured."""
    if not measurement_id:
        return ""
    return f"""<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());

  gtag('config', '{measurement_id}');
</script>
"""


def structured_data(cfg, url):
    org, author, site = cfg["organization"], cfg["author"], cfg["site_name"]
    org_ld = {"@type": "Organization", "@id": org["url"] + "#organization", "name": org["name"],
              "legalName": org.get("legal_name", org["name"]), "url": org["url"], "email": org["email"],
              "description": org["description"], "logo": {"@type": "ImageObject", "url": url + "icon-512.png"},
              "founder": [{"@type": "Person", "name": n} for n in org.get("founders", [])],
              "sameAs": [p for p in org.get("profiles", []) if p]}
    author_ld = {"@type": "Person", "@id": author["url"] + "#person", "name": author["name"], "url": author["url"],
                 "jobTitle": author.get("job_title"), "description": author.get("description"),
                 "worksFor": {"@id": org["url"] + "#organization"},
                 "affiliation": {"@id": org["url"] + "#organization"},
                 "sameAs": [p for p in author.get("profiles", []) if p]}
    website_ld = {"@type": "WebSite", "@id": url + "#website", "name": site, "url": url,
                  "description": DESCRIPTION, "inLanguage": "es", "publisher": {"@id": org["url"] + "#organization"}}
    article_ld = {"@type": "ScholarlyArticle", "@id": url + "#article", "headline": TITLE_LONG,
                  "alternativeHeadline": SUBTITLE, "description": DESCRIPTION, "inLanguage": "es",
                  "url": url, "mainEntityOfPage": url, "isPartOf": {"@id": url + "#website"},
                  "datePublished": PUBLISHED, "dateModified": MODIFIED, "version": VERSION,
                  "identifier": REPORT_ID, "image": url + "og-image.png",
                  "author": {"@id": author["url"] + "#person"}, "publisher": {"@id": org["url"] + "#organization"},
                  "about": ["dendritas", "modelos de lenguaje", "Gemma 4", "inteligencia emocional",
                            "vectores de tarea", "acompañamiento", "deterioro cognitivo leve"],
                  "isAccessibleForFree": True}
    faq_ld = {"@type": "FAQPage", "mainEntity": [
        {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in FAQ]}
    graph = {"@context": "https://schema.org", "@graph": [org_ld, author_ld, website_ld, article_ld, faq_ld]}
    return f'<script type="application/ld+json">{json.dumps(graph, ensure_ascii=False)}</script>'


def head(cfg, url, styles):
    site, author = cfg["site_name"], cfg["author"]
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{analytics(cfg.get("analytics_id"))}<title>{TITLE_LONG} | {site}</title>
<meta name="description" content="{DESCRIPTION}">
<meta name="author" content="{author['name']}">
<meta name="application-name" content="{site}">
<meta name="apple-mobile-web-app-title" content="{site}">
<meta name="robots" content="index, follow, max-image-preview:large">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0c1014" media="(prefers-color-scheme: dark)">
<link rel="canonical" href="{url}">
<link rel="author" href="{author['url']}">
<link rel="icon" href="{url}favicon.ico" sizes="any">
<link rel="icon" href="{url}favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="{url}apple-touch-icon.png">
<link rel="manifest" href="{url}site.webmanifest">
<link rel="sitemap" type="application/xml" href="{url}sitemap.xml">
<meta property="og:type" content="article">
<meta property="og:site_name" content="{site}">
<meta property="og:locale" content="es_ES">
<meta property="og:url" content="{url}">
<meta property="og:title" content="{TITLE_LONG}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:image" content="{url}og-image.png">
<meta property="og:image:secure_url" content="{url}og-image.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{site}: Inteligencia artificial emocional: el programa de investigación Leuner, con un dibujo de dendrita a trazos">
<meta property="og:image" content="{url}og-image-square.png">
<meta property="og:image:secure_url" content="{url}og-image-square.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="1200">
<meta property="article:published_time" content="{PUBLISHED}">
<meta property="article:modified_time" content="{MODIFIED}">
<meta property="article:author" content="{author['url']}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{TITLE_LONG}">
<meta name="twitter:description" content="{DESCRIPTION}">
<meta name="twitter:image" content="{url}og-image.png">
{structured_data(cfg, url)}
{styles}
</head>
<body>
"""


# ─── Texto plano del artículo para llms-full.txt ────────────────────────────
def html_to_markdown(body):
    """Conversión sencilla y determinista del artículo a Markdown: títulos, párrafos, listas, tablas."""
    t = re.sub(r"<(svg|script|style|details|nav|aside|header)\b.*?</\1>", "", body, flags=re.S)
    t = re.sub(r"<div class=\"kicker\">.*?</div>", "", t, flags=re.S)
    t = re.sub(r"<sup class=\"ref\">(.*?)</sup>", lambda m: "[" + re.sub(r"<[^>]+>", "", m.group(1)) + "]", t, flags=re.S)
    t = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", t, flags=re.S)
    t = re.sub(r"<h2[^>]*>(?:<span class=\"n\">.*?</span>)?(.*?)</h2>", r"\n## \1\n", t, flags=re.S)
    t = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", t, flags=re.S)
    t = re.sub(r"<span class=\"tag\">(.*?)</span>", r"\n### \1\n", t, flags=re.S)
    t = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", t, flags=re.S)
    t = re.sub(r"<dt[^>]*>(.*?)</dt>", r"\n**\1**\n", t, flags=re.S)
    t = re.sub(r"<dd[^>]*>(.*?)</dd>", r"\1\n", t, flags=re.S)
    t = re.sub(r"<tr[^>]*>(.*?)</tr>", lambda m: "| " + " | ".join(
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()
        for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", m.group(1), flags=re.S)) + " |\n", t, flags=re.S)
    t = re.sub(r"<figcaption[^>]*>(.*?)</figcaption>", r"\n_\1_\n", t, flags=re.S)
    t = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", t, flags=re.S)
    t = re.sub(r"</p>|<br\s*/?>", "\n\n", t)
    t = re.sub(r"<a [^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", r"[\2](\1)", t, flags=re.S)
    t = re.sub(r"<(strong|b)>(.*?)</\1>", r"**\2**", t, flags=re.S)
    t = re.sub(r"<(em|i)>(.*?)</\1>", r"_\2_", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"


# ─── Archivos estáticos ─────────────────────────────────────────────────────
def write_static(cfg, url, body_markdown):
    org, author, site = cfg["organization"], cfg["author"], cfg["site_name"]
    (DIST / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {url}sitemap.xml\n", encoding="utf-8")
    (DIST / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url>\n    <loc>{url}</loc>\n    <lastmod>{MODIFIED}</lastmod>\n    <changefreq>monthly</changefreq>\n  </url>\n"
        "</urlset>\n", encoding="utf-8")

    profiles = [p for p in org.get("profiles", []) + author.get("profiles", []) if p]
    faq = "\n".join(f"- {q} {a}" for q, a in FAQ)
    (DIST / "llms.txt").write_text(
        f"# {site}\n\n"
        f"> {DESCRIPTION}\n\n"
        f"{site} es el programa de investigación de {org['name']}, {org['description'][0].lower()}{org['description'][1:]} "
        f"Lo dirige {author['name']} ({author.get('job_title', '')}). "
        f"Este sitio publica el informe técnico {REPORT_ID} (versión {VERSION}, publicado el {PUBLISHED}, "
        f"actualizado el {MODIFIED}), escrito en español, sobre el modelo abierto Gemma 4 E2B (Apache 2.0). "
        "Regla del proyecto: solo lo medido se comunica como resultado; el resto se etiqueta como diseño o hipótesis.\n\n"
        "## Investigación\n\n"
        f"- [{TITLE_LONG}]({url}): {SUBTITLE}. Qué es una dendrita, qué contendrá la artificial, el primer hito "
        "medido (una proteína de 6 KB lleva la traducción del 0 % al 90 % sin ejemplos), cómo seguirá el trabajo, "
        "el campo (inteligencia emocional para acompañar) y la aplicación futura (estadio uno y dos de enfermedad "
        "neurológica). 59 fuentes.\n"
        f"- [Texto completo del informe en Markdown]({url}llms-full.txt): el artículo entero, sin diseño, para leer o citar.\n\n"
        "## Respuestas breves\n\n"
        f"{faq}\n\n"
        "## Quién está detrás\n\n"
        f"- [{org['name']}]({org['url']}): {org['description']}\n"
        f"- [{author['name']}]({author['url']}): {author.get('description', '')}\n"
        + "".join(f"- [Perfil oficial]({p})\n" for p in profiles) +
        f"- Contacto: {org['email']}\n\n"
        "## Optional\n\n"
        f"- [Imagen para compartir]({url}og-image.png)\n"
        f"- [Sitemap]({url}sitemap.xml)\n",
        encoding="utf-8")

    (DIST / "llms-full.txt").write_text(
        f"Autor: {author['name']} ({author['url']}), {org['name']} ({org['url']}). "
        f"Informe {REPORT_ID}, versión {VERSION}. Publicado el {PUBLISHED}, actualizado el {MODIFIED}. "
        f"Fuente: {url}\n\n" + body_markdown, encoding="utf-8")

    (DIST / "site.webmanifest").write_text(json.dumps({
        "name": site, "short_name": site, "description": DESCRIPTION, "lang": "es",
        "start_url": url, "display": "browser", "background_color": "#ffffff", "theme_color": "#ffffff",
        "icons": [{"src": url + "icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": url + "icon-512.png", "sizes": "512x512", "type": "image/png"}]},
        indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=cfg.get("url", "https://leuner.es/"),
                    help="URL pública de la página, con barra final (por defecto, site.json)")
    ap.add_argument("--analytics", default=cfg.get("analytics_id"),
                    help="ID de medición de Google Analytics (por defecto, site.json); vacío para no incluirlo")
    a = ap.parse_args()
    url = a.url if a.url.endswith("/") else a.url + "/"
    cfg["analytics_id"] = a.analytics

    src = SOURCE.read_text(encoding="utf-8")
    title_tag = re.search(r"<title>.*?</title>", src, re.S).group(0)
    head_end = src.index("</style>") + len("</style>")
    styles = src[len(title_tag):head_end].strip()
    body = src[head_end:].strip()
    # La cita BibTeX apunta a la misma URL que la canónica.
    body = re.sub(r"url\s*=\s*\{https://leuner\.es[^}]*\}", f"url         = {{{url}}}", body)

    DIST.mkdir(exist_ok=True)
    (DIST / "index.html").write_text(head(cfg, url, styles) + body + "\n</body>\n</html>\n", encoding="utf-8")
    article = re.search(r"<article id=\"paper\">(.*?)</article>", body, re.S).group(1)
    write_static(cfg, url, html_to_markdown(article))
    for f in sorted(DIST.iterdir()):
        print(f"{f.name:<24}{f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
