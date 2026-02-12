"""
Encycloscope MCP Server - Streamable HTTP Implementation

MCP server using Streamable HTTP transport for Claude Web compatibility.
Based on the working wos-mcp implementation.
"""

import json
import logging
import re
import sys
from typing import Annotated
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware as CORSMiddlewareClass

from .core.config import get_settings
from .core.corpus import EncyclopediaCorpus
from . import __version__

# ==================== MCP Apps UI Resources ====================

NETWORK_VIEW_URI = "ui://encycloscope/author-network.html"

# Embedded HTML for author network visualization using D3.js
NETWORK_VIEW_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <title>Réseau d'auteur - Encyclopédie</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      width: 100%;
      height: 100%;
      min-height: 500px;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: transparent;
    }

    body {
      display: flex;
      flex-direction: column;
      height: 500px;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #1a1a1a;
        --text: #e0e0e0;
        --text-muted: #888;
        --border: #333;
        --node-main: #6366f1;
        --node-renvoi: #22c55e;
        --node-domain: #f59e0b;
        --link: #555;
      }
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #ffffff;
        --text: #1a1a1a;
        --text-muted: #666;
        --border: #e0e0e0;
        --node-main: #4f46e5;
        --node-renvoi: #16a34a;
        --node-domain: #d97706;
        --link: #ccc;
      }
    }

    #header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--bg);
    }

    #header h1 {
      font-size: 16px;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 4px;
    }

    #header .stats {
      font-size: 12px;
      color: var(--text-muted);
    }

    #graph-container {
      flex: 1;
      position: relative;
      overflow: hidden;
    }

    #graph {
      width: 100%;
      height: 100%;
    }

    #tooltip {
      position: absolute;
      padding: 8px 12px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 12px;
      color: var(--text);
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.15s;
      max-width: 250px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      z-index: 100;
    }

    #tooltip.visible { opacity: 1; }

    #tooltip .name {
      font-weight: 600;
      margin-bottom: 4px;
    }

    #tooltip .info {
      color: var(--text-muted);
      font-size: 11px;
    }

    #legend {
      position: absolute;
      bottom: 12px;
      left: 12px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 11px;
      color: var(--text-muted);
    }

    #legend .item {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }

    #legend .item:last-child { margin-bottom: 0; }

    #legend .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }

    .dot-main { background: var(--node-main); }
    .dot-renvoi { background: var(--node-renvoi); }
    .dot-domain { background: var(--node-domain); }

    #loading {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: var(--text-muted);
      font-size: 14px;
    }

    .node {
      cursor: pointer;
    }

    .node:hover {
      filter: brightness(1.2);
    }

    .link {
      stroke: var(--link);
      stroke-opacity: 0.6;
    }

    .label {
      font-size: 10px;
      fill: var(--text);
      pointer-events: none;
      text-anchor: middle;
    }
  </style>
</head>
<body>
  <div id="header">
    <h1 id="title">Chargement...</h1>
    <div class="stats" id="stats"></div>
  </div>

  <div id="graph-container">
    <div id="loading">Chargement du réseau...</div>
    <svg id="graph"></svg>
    <div id="tooltip">
      <div class="name"></div>
      <div class="info"></div>
    </div>
    <div id="legend">
      <div class="item"><span class="dot dot-main"></span> Auteur principal</div>
      <div class="item"><span class="dot dot-renvoi"></span> Connecté via renvois</div>
      <div class="item"><span class="dot dot-domain"></span> Domaine partagé</div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script type="module">
    import { App } from "https://unpkg.com/@modelcontextprotocol/ext-apps@0.4.0/app-with-deps";

    const app = new App({ name: "Author Network View", version: "1.0.0" });

    let simulation = null;

    function renderNetwork(data) {
      document.getElementById('loading').style.display = 'none';

      // Update header
      document.getElementById('title').textContent = `Réseau de ${data.author}`;
      document.getElementById('stats').textContent =
        `${data.articles_count} articles · ${Object.keys(data.connected_through_renvois || {}).length} connexions via renvois · ${data.domains?.length || 0} domaines`;

      // Prepare nodes and links
      const nodes = [];
      const links = [];
      const nodeMap = new Map();

      // Main author node
      const mainNode = {
        id: data.author,
        name: data.author,
        type: 'main',
        articles: data.articles_count,
        domains: data.domains || []
      };
      nodes.push(mainNode);
      nodeMap.set(data.author, mainNode);

      // Connected authors via renvois
      const renvoisAuthors = data.connected_through_renvois || {};
      Object.entries(renvoisAuthors).forEach(([author, count]) => {
        if (!nodeMap.has(author)) {
          const node = { id: author, name: author, type: 'renvoi', connections: count };
          nodes.push(node);
          nodeMap.set(author, node);
        }
        links.push({ source: data.author, target: author, type: 'renvoi', weight: count });
      });

      // Shared domain authors (limit to top 10 to avoid clutter)
      const domainAuthors = data.shared_domain_authors || {};
      const topDomainAuthors = Object.entries(domainAuthors)
        .sort((a, b) => b[1].length - a[1].length)
        .slice(0, 10);

      topDomainAuthors.forEach(([author, domains]) => {
        if (!nodeMap.has(author)) {
          const node = { id: author, name: author, type: 'domain', sharedDomains: domains };
          nodes.push(node);
          nodeMap.set(author, node);
        } else {
          // Already exists from renvois, add domain info
          nodeMap.get(author).sharedDomains = domains;
        }
        if (!links.find(l =>
          (l.source === data.author && l.target === author) ||
          (l.source === author && l.target === data.author)
        )) {
          links.push({ source: data.author, target: author, type: 'domain', weight: domains.length });
        }
      });

      // Setup SVG
      const container = document.getElementById('graph-container');
      const width = container.clientWidth;
      const height = container.clientHeight;

      const svg = d3.select('#graph')
        .attr('width', width)
        .attr('height', height);

      svg.selectAll('*').remove();

      // Create simulation with adjusted forces for smaller nodes
      simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(60))
        .force('charge', d3.forceManyBody().strength(-150))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => getNodeRadius(d) + 5));

      // Draw links
      const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('stroke-width', d => Math.sqrt(d.weight) * 1.5);

      // Draw nodes with logarithmic sizing to avoid giant circles
      function getNodeRadius(d) {
        if (d.type === 'main') return 18;
        const value = d.connections || d.sharedDomains?.length || 1;
        // Log scale: 6px base + log(value) * 4, capped at 25px
        return Math.min(25, 6 + Math.log(value + 1) * 4);
      }

      const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('class', 'node')
        .attr('r', getNodeRadius)
        .attr('fill', d => {
          if (d.type === 'main') return getComputedStyle(document.documentElement).getPropertyValue('--node-main');
          if (d.type === 'renvoi') return getComputedStyle(document.documentElement).getPropertyValue('--node-renvoi');
          return getComputedStyle(document.documentElement).getPropertyValue('--node-domain');
        })
        .call(drag(simulation));

      // Draw labels positioned below nodes
      const label = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .join('text')
        .attr('class', 'label')
        .attr('dy', d => getNodeRadius(d) + 12)
        .text(d => d.name.length > 12 ? d.name.slice(0, 10) + '...' : d.name);

      // Tooltip handling
      const tooltip = document.getElementById('tooltip');

      node.on('mouseenter', (event, d) => {
        tooltip.querySelector('.name').textContent = d.name;
        let info = '';
        if (d.type === 'main') {
          info = `${d.articles} articles\\nDomaines: ${d.domains.slice(0, 3).join(', ')}${d.domains.length > 3 ? '...' : ''}`;
        } else if (d.type === 'renvoi') {
          info = `${d.connections} renvois partagés`;
          if (d.sharedDomains) info += `\\n${d.sharedDomains.length} domaines communs`;
        } else {
          info = `${d.sharedDomains?.length || 0} domaines communs`;
        }
        tooltip.querySelector('.info').innerHTML = info.replace(/\\n/g, '<br>');
        tooltip.classList.add('visible');
      });

      node.on('mousemove', (event) => {
        tooltip.style.left = (event.offsetX + 15) + 'px';
        tooltip.style.top = (event.offsetY + 15) + 'px';
      });

      node.on('mouseleave', () => {
        tooltip.classList.remove('visible');
      });

      // Update positions on tick
      simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);

        node
          .attr('cx', d => d.x = Math.max(20, Math.min(width - 20, d.x)))
          .attr('cy', d => d.y = Math.max(20, Math.min(height - 20, d.y)));

        label
          .attr('x', d => d.x)
          .attr('y', d => d.y);
      });
    }

    // Drag behavior
    function drag(simulation) {
      function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }

      function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }

      function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }

      return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }

    // Handle tool result from MCP
    app.ontoolresult = ({ content, structuredContent }) => {
      let data;

      // Try structured content first
      if (structuredContent) {
        data = structuredContent;
      } else if (content) {
        // Parse from text content
        const textContent = content.find(c => c.type === 'text');
        if (textContent) {
          try {
            data = JSON.parse(textContent.text);
          } catch (e) {
            console.error('Failed to parse content:', e);
            document.getElementById('loading').textContent = 'Erreur de chargement';
            return;
          }
        }
      }

      if (data) {
        renderNetwork(data);
      }
    };

    // Handle resize
    window.addEventListener('resize', () => {
      if (simulation) {
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        d3.select('#graph')
          .attr('width', width)
          .attr('height', height);

        simulation.force('center', d3.forceCenter(width / 2, height / 2));
        simulation.alpha(0.3).restart();
      }
    });

    // Connect to MCP host and signal preferred size
    await app.connect();

    // Tell the host we want more height
    if (app.sendSizeChanged) {
      app.sendSizeChanged({ width: 600, height: 500 });
    }
  </script>
</body>
</html>"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("encycloscope-mcp")

# Load environment variables
load_dotenv()

# Get settings
settings = get_settings()

# Create FastMCP server with Streamable HTTP configuration (same as wos-mcp)
mcp = FastMCP(
    "encycloscope-mcp",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["localhost:*", "127.0.0.1:*"],
        allowed_origins=["https://claude.ai", "https://*.claude.ai"],
    ),
)

# Initialize corpus (lazy loading)
_corpus: EncyclopediaCorpus | None = None


def get_corpus() -> EncyclopediaCorpus:
    """Get or create the corpus instance."""
    global _corpus
    if _corpus is None:
        logger.info("Loading corpus...")

        # Prefer new FAISS index if available
        faiss_index = settings.faiss_index_path if settings.faiss_index_path.exists() else None
        chunks = settings.chunks_path if settings.chunks_path.exists() else None

        # Fall back to legacy paths
        legacy_index = settings.index_path if settings.index_path.exists() else None
        legacy_blocks = settings.blocks_path if settings.blocks_path.exists() else None
        legacy_model = settings.model_path if settings.model_path.exists() else None

        if faiss_index and chunks:
            logger.info(f"Using d'AlemBERT FAISS index: {faiss_index}")
        elif legacy_index:
            logger.info(f"Using legacy index: {legacy_index}")

        _corpus = EncyclopediaCorpus(
            articles_path=settings.articles_path,
            model_path=legacy_model,
            index_path=faiss_index or legacy_index,
            blocks_path=legacy_blocks,
            chunks_path=chunks,
        )
        logger.info(f"Corpus loaded: {len(_corpus.articles_df)} articles")
    return _corpus


# ==================== MCP Tools ====================


@mcp.tool()
async def encyclopedie_search(
    query: Annotated[str, "Question ou concept à rechercher"],
    limit: Annotated[int, "Nombre de résultats (1-50)"] = 10,
    author: Annotated[str | None, "Filtrer par auteur"] = None,
    domain: Annotated[str | None, "Filtrer par domaine"] = None,
) -> str:
    """Recherche dans l'Encyclopédie de Diderot et d'Alembert (sémantique si disponible, sinon textuelle)."""
    corpus = get_corpus()

    # Try semantic search first
    try:
        results = corpus.search_semantic(query=query, limit=limit, author=author, domain=domain, threshold=0.2)
        return json.dumps({"query": query, "search_type": "semantic", "total_found": len(results), "results": results}, indent=2, ensure_ascii=False)
    except RuntimeError:
        pass  # Fall back to text search

    # Fallback: text-based search in article content
    df = corpus.articles_df
    query_terms = query.lower().split()

    # Filter by author/domain first if specified
    filtered = df.copy()
    if author:
        filtered = filtered[filtered['auteurs'].str.lower().str.contains(author.lower(), na=False)]
    if domain:
        filtered = filtered[filtered['domainesEnccre'].str.lower().str.contains(domain.lower(), na=False)]

    # Score articles by number of query terms found
    def score_article(text):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(1 for term in query_terms if term in text_lower)

    filtered['_score'] = filtered['contenu_clean'].apply(score_article)
    matches = filtered[filtered['_score'] > 0].nlargest(limit, '_score')

    results = []
    for _, row in matches.iterrows():
        text = str(row.get('contenu_clean', ''))
        # Extract context around first matching term
        context = ""
        for term in query_terms:
            idx = text.lower().find(term)
            if idx >= 0:
                start = max(0, idx - 100)
                end = min(len(text), idx + 150)
                context = ('...' if start > 0 else '') + text[start:end].strip() + ('...' if end < len(text) else '')
                break

        results.append({
            'articleID': row.get('articleID', ''),
            'vedette': row.get('vedette', ''),
            'auteur': row.get('auteurs', 'Anonyme'),
            'domaine': row.get('domainesEnccre', ''),
            'text': context or text[:250] + '...',
            'url': f"https://enccre.academie-sciences.fr/encyclopedie/article/{row.get('articleID', '')}/",
        })

    return json.dumps({
        "query": query,
        "search_type": "text",
        "note": "Recherche textuelle (modèle sémantique non disponible)",
        "total_found": len(results),
        "results": results
    }, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_search_by_vedette(
    vedette: Annotated[str, "Vedette à rechercher"],
    exact: Annotated[bool, "Recherche exacte"] = False,
) -> str:
    """Recherche par vedette (mot-entrée) dans l'Encyclopédie."""
    corpus = get_corpus()
    results = corpus.search_by_vedette(vedette, exact=exact)
    return json.dumps({"vedette_query": vedette, "total_found": len(results), "articles": results}, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_search_by_author(
    author: Annotated[str, "Nom de l'auteur"],
    limit: Annotated[int, "Nombre maximum"] = 20,
) -> str:
    """Liste les articles d'un auteur dans l'Encyclopédie."""
    corpus = get_corpus()
    articles = corpus.get_articles_by_author(author)
    return json.dumps({"author_query": author, "total_found": len(articles), "articles": articles[:limit]}, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_search_by_domain(
    domain: Annotated[str, "Domaine"],
    limit: Annotated[int, "Nombre maximum"] = 20,
) -> str:
    """Liste les articles d'un domaine dans l'Encyclopédie."""
    corpus = get_corpus()
    articles = corpus.get_articles_by_domain(domain)
    return json.dumps({"domain_query": domain, "total_found": len(articles), "articles": articles[:limit]}, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_get_article(
    article_id: Annotated[str, "Identifiant de l'article (ex: v10-1234-0)"],
) -> str:
    """Récupère un article complet de l'Encyclopédie."""
    corpus = get_corpus()
    article = corpus.get_article(article_id)
    if article is None:
        return json.dumps({"error": f"Article non trouvé: {article_id}"}, ensure_ascii=False)
    return json.dumps(article, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_get_definition(
    article_id: Annotated[str, "Identifiant de l'article"],
) -> str:
    """Extrait la définition d'un article de l'Encyclopédie."""
    corpus = get_corpus()
    result = corpus.extract_definition(article_id)
    if result is None:
        return json.dumps({"error": f"Article non trouvé: {article_id}"}, ensure_ascii=False)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_count_term(
    term: Annotated[str, "Terme à compter"],
    author: Annotated[str | None, "Filtrer par auteur"] = None,
    domain: Annotated[str | None, "Filtrer par domaine"] = None,
) -> str:
    """Compte le nombre d'occurrences d'un terme dans le corpus.

    Utile pour avoir une vue d'ensemble avant d'explorer les contextes
    avec encyclopedie_get_term_contexts.
    """
    corpus = get_corpus()
    df = corpus.articles_df
    term_lower = term.lower()
    pattern = re.escape(term_lower)

    # Filter by author/domain if specified
    filtered_df = df[df['contenu_clean'].str.lower().str.contains(term_lower, na=False)]

    if author:
        filtered_df = filtered_df[
            filtered_df['auteurs'].fillna('').str.lower().str.contains(author.lower())
        ]
    if domain:
        filtered_df = filtered_df[
            filtered_df['domainesEnccre'].fillna('').str.lower().str.contains(domain.lower())
        ]

    total_articles = len(filtered_df)
    total_occurrences = 0

    for _, row in filtered_df.iterrows():
        text = str(row.get('contenu_clean', '')).lower()
        total_occurrences += len(re.findall(pattern, text))

    return json.dumps({
        "term": term,
        "filters": {"author": author, "domain": domain},
        "total_articles": total_articles,
        "total_occurrences": total_occurrences,
    }, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_compare_term_usage(
    term: Annotated[str, "Terme à analyser"],
    limit: Annotated[int, "Nombre d'articles"] = 10,
) -> str:
    """Compare l'usage d'un terme par différents auteurs de l'Encyclopédie."""
    corpus = get_corpus()
    result = corpus.compare_term_usage(term, limit=limit)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_get_term_contexts(
    term: Annotated[str, "Terme à rechercher"],
    context_size: Annotated[int, "Caractères de contexte (mode chars)"] = 150,
    limit: Annotated[int, "Nombre max d'occurrences à retourner"] = 50,
    offset: Annotated[int, "Décalage pour pagination"] = 0,
    all_occurrences: Annotated[bool, "Extraire toutes les occurrences par article"] = False,
    context_mode: Annotated[str, "Mode: 'chars' ou 'sentence'"] = "chars",
    author: Annotated[str | None, "Filtrer par auteur"] = None,
    domain: Annotated[str | None, "Filtrer par domaine"] = None,
    exclude_patterns: Annotated[list[str] | None, "Exclure les occurrences contenant ces patterns (ex: ['luxemb', 'luxeuil'])"] = None,
) -> str:
    """Récupère les contextes d'usage d'un terme avec pagination.

    Modes de contexte:
    - 'chars': nombre fixe de caractères autour du terme (défaut)
    - 'sentence': phrase complète contenant le terme

    Si all_occurrences=False (défaut), retourne une seule occurrence par article.
    Si all_occurrences=True, retourne toutes les occurrences de chaque article.

    exclude_patterns permet d'exclure les faux positifs (ex: 'luxe' dans 'Luxembourg').
    """
    corpus = get_corpus()
    df = corpus.articles_df
    term_lower = term.lower()
    pattern = re.escape(term_lower)

    # Regex for sentence boundaries
    SENTENCE_DELIMITERS = r'(?<=[.!?;])\s+'

    # Filter articles containing the term
    filtered_df = df[df['contenu_clean'].str.lower().str.contains(term_lower, na=False)]

    if author:
        filtered_df = filtered_df[
            filtered_df['auteurs'].fillna('').str.lower().str.contains(author.lower())
        ]
    if domain:
        filtered_df = filtered_df[
            filtered_df['domainesEnccre'].fillna('').str.lower().str.contains(domain.lower())
        ]

    # Normalize exclude patterns
    exclude_lower = [p.lower() for p in exclude_patterns] if exclude_patterns else []

    # Pre-compute all matches for accurate pagination
    all_matches = []
    for _, row in filtered_df.iterrows():
        text = str(row.get('contenu_clean', ''))
        text_lower = text.lower()

        matches_in_article = list(re.finditer(pattern, text_lower))

        # Filter out matches containing exclude patterns
        if exclude_lower:
            filtered_matches = []
            for match in matches_in_article:
                # Check surrounding context for exclude patterns
                start = max(0, match.start() - 20)
                end = min(len(text_lower), match.end() + 20)
                surrounding = text_lower[start:end]
                if not any(excl in surrounding for excl in exclude_lower):
                    filtered_matches.append(match)
            matches_in_article = filtered_matches

        occurrences_in_article = len(matches_in_article)

        if not all_occurrences:
            # Only first occurrence per article
            matches_in_article = matches_in_article[:1]

        for occ_idx, match in enumerate(matches_in_article, start=1):
            all_matches.append({
                'row': row,
                'match': match,
                'occurrence_index': occ_idx,
                'occurrences_in_article': occurrences_in_article,
            })

    total_occurrences = len(all_matches)
    total_articles = len(filtered_df)

    # Apply pagination
    paginated = all_matches[offset:offset + limit]
    has_more = (offset + limit) < total_occurrences

    # Extract contexts
    contexts = []
    for item in paginated:
        row = item['row']
        match = item['match']
        text = str(row.get('contenu_clean', ''))
        idx = match.start()

        if context_mode == "sentence":
            # Split text into sentences
            sentences = re.split(SENTENCE_DELIMITERS, text)
            # Find which sentence contains our match
            current_pos = 0
            context = ""
            for sentence in sentences:
                sentence_end = current_pos + len(sentence)
                if current_pos <= idx < sentence_end:
                    context = sentence.strip()
                    break
                current_pos = sentence_end + 1  # +1 for the split character
            if not context:
                # Fallback to chars mode if sentence detection fails
                start = max(0, idx - context_size)
                end = min(len(text), idx + len(term) + context_size)
                context = text[start:end].strip()
                if start > 0:
                    context = '...' + context
                if end < len(text):
                    context = context + '...'
        else:
            # chars mode (default)
            start = max(0, idx - context_size)
            end = min(len(text), idx + len(term) + context_size)
            context = text[start:end].strip()
            if start > 0:
                context = '...' + context
            if end < len(text):
                context = context + '...'

        row_author = row.get('auteurs', 'Anonyme')
        row_domain = row.get('domainesEnccre', '')

        contexts.append({
            'articleID': row.get('articleID', ''),
            'vedette': row.get('vedette', ''),
            'auteur': row_author if row_author else 'Anonyme',
            'domaine': row_domain,
            'occurrence_index': item['occurrence_index'],
            'occurrences_in_article': item['occurrences_in_article'],
            'char_position': idx,
            'context': context,
            'url': f"https://enccre.academie-sciences.fr/encyclopedie/article/{row.get('articleID', '')}/",
        })

    return json.dumps({
        'term': term,
        'filters': {'author': author, 'domain': domain, 'exclude_patterns': exclude_patterns},
        'total_articles': total_articles,
        'total_occurrences': total_occurrences,
        'returned': len(contexts),
        'offset': offset,
        'has_more': has_more,
        'contexts': contexts,
    }, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_get_renvois(
    article_id: Annotated[str, "Identifiant de l'article"],
) -> str:
    """Récupère les renvois (références croisées) d'un article."""
    corpus = get_corpus()
    renvois = corpus.get_renvois(article_id)
    return json.dumps(renvois, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_get_author_network(
    author: Annotated[str, "Nom de l'auteur"],
) -> str:
    """Analyse le réseau d'un auteur via les renvois."""
    corpus = get_corpus()
    network = corpus.get_author_network(author)
    return json.dumps(network, indent=2, ensure_ascii=False)


# ==================== MCP Apps Tools ====================


@mcp.tool(
    meta={
        "ui": {"resourceUri": NETWORK_VIEW_URI},
        "ui/resourceUri": NETWORK_VIEW_URI,  # legacy support
    }
)
async def encyclopedie_visualize_author_network(
    author: Annotated[str, "Nom de l'auteur à visualiser"],
) -> dict:
    """Visualise le réseau d'un auteur dans l'Encyclopédie sous forme de graphe interactif.

    Affiche les connexions via les renvois et les domaines partagés.
    Le graphe permet de voir les collaborations et influences entre encyclopédistes.
    """
    corpus = get_corpus()
    network = corpus.get_author_network(author)

    # Limit data to avoid browser overload
    if 'connected_through_renvois' in network:
        # Keep only top 20 connections
        renvois = network['connected_through_renvois']
        if len(renvois) > 20:
            sorted_renvois = dict(sorted(renvois.items(), key=lambda x: x[1], reverse=True)[:20])
            network['connected_through_renvois'] = sorted_renvois

    if 'shared_domain_authors' in network:
        # Keep only top 15 shared domain authors
        domain_authors = network['shared_domain_authors']
        if len(domain_authors) > 15:
            sorted_da = dict(sorted(domain_authors.items(), key=lambda x: len(x[1]), reverse=True)[:15])
            network['shared_domain_authors'] = sorted_da

    if 'domains' in network:
        # Limit domains list
        network['domains'] = network['domains'][:20]

    return network


# ==================== MCP Apps Resources ====================


@mcp.resource(
    NETWORK_VIEW_URI,
    mime_type="text/html;profile=mcp-app",
    meta={
        "ui": {
            "csp": {
                "resourceDomains": [
                    "https://cdn.jsdelivr.net",
                    "https://unpkg.com"
                ]
            }
        }
    },
)
def author_network_view() -> str:
    """HTML resource for author network visualization."""
    return NETWORK_VIEW_HTML


@mcp.tool()
async def encyclopedie_list_authors(
    prefix: Annotated[str | None, "Préfixe pour filtrer"] = None,
) -> str:
    """Liste les auteurs de l'Encyclopédie."""
    corpus = get_corpus()
    authors = corpus.get_authors()
    if prefix:
        authors = [a for a in authors if a.lower().startswith(prefix.lower())]
    return json.dumps({"total_authors": len(authors), "authors": authors[:100]}, indent=2, ensure_ascii=False)


@mcp.tool()
async def encyclopedie_list_domains(
    prefix: Annotated[str | None, "Préfixe pour filtrer"] = None,
) -> str:
    """Liste les domaines de l'Encyclopédie."""
    corpus = get_corpus()
    domains = corpus.get_domains()
    if prefix:
        domains = [d for d in domains if d.lower().startswith(prefix.lower())]
    return json.dumps({"total_domains": len(domains), "domains": domains[:100]}, indent=2, ensure_ascii=False)


# ==================== Authentication ====================


def validate_token(request: Request) -> bool:
    """Validate token from query params or Authorization header."""
    if not settings.mcp_token:
        return True

    query_token = request.query_params.get("token")
    auth_header = request.headers.get("authorization", "")
    bearer_token = auth_header[7:] if auth_header.startswith("Bearer ") else None

    token = query_token or bearer_token
    return token == settings.mcp_token


# ==================== HTTP Routes ====================


async def health_check(request: Request):
    """Health check endpoint."""
    try:
        corpus = get_corpus()
        articles_count = len(corpus.articles_df)
        return JSONResponse({
            "status": "healthy",
            "version": __version__,
            "transport": "streamable-http",
            "corpus_loaded": True,
            "articles_count": articles_count,
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
        }, status_code=503)


async def info(request: Request):
    """Server info endpoint."""
    return JSONResponse({
        "name": "Encycloscope MCP Server",
        "version": __version__,
        "description": "MCP server for exploring the Encyclopédie de Diderot et d'Alembert",
        "transports": {
            "streamable-http": "/mcp",
            "sse": "/sse"
        },
        "tools": [
            "encyclopedie_search",
            "encyclopedie_search_by_vedette",
            "encyclopedie_search_by_author",
            "encyclopedie_search_by_domain",
            "encyclopedie_get_article",
            "encyclopedie_get_definition",
            "encyclopedie_count_term",
            "encyclopedie_compare_term_usage",
            "encyclopedie_get_term_contexts",
            "encyclopedie_get_renvois",
            "encyclopedie_get_author_network",
            "encyclopedie_list_authors",
            "encyclopedie_list_domains",
            "encyclopedie_visualize_author_network",  # MCP App with UI
        ],
    })


# ==================== App Setup ====================


# Get the ASGI apps from FastMCP
raw_mcp_app = mcp.streamable_http_app()  # Streamable HTTP transport (/mcp)
raw_sse_app = mcp.sse_app()  # SSE transport (/sse + /messages)


# Combined ASGI application with auth and custom routes (same pattern as wos-mcp)
class CombinedApp:
    """ASGI app that combines MCP with custom routes and authentication."""

    def __init__(self, mcp_app, sse_app):
        self.mcp_app = mcp_app
        self.sse_app = sse_app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.mcp_app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Handle custom routes
        if path == "/":
            request = Request(scope, receive)
            response = await info(request)
            await response(scope, receive, send)
            return

        if path == "/health":
            request = Request(scope, receive)
            response = await health_check(request)
            await response(scope, receive, send)
            return

        # For MCP routes (Streamable HTTP), check authentication
        if path.startswith("/mcp"):
            request = Request(scope, receive)
            if not validate_token(request):
                response = JSONResponse(
                    {"error": "Invalid or missing token"},
                    status_code=401
                )
                await response(scope, receive, send)
                return
            # Forward to Streamable HTTP MCP app
            await self.mcp_app(scope, receive, send)
            return

        # For SSE transport routes (/sse and /messages), check authentication
        if path.startswith("/sse") or path.startswith("/messages"):
            request = Request(scope, receive)
            if not validate_token(request):
                response = JSONResponse(
                    {"error": "Invalid or missing token"},
                    status_code=401
                )
                await response(scope, receive, send)
                return
            # Forward to SSE MCP app
            await self.sse_app(scope, receive, send)
            return

        # Default: forward to MCP app
        await self.mcp_app(scope, receive, send)


# Wrap with CORS middleware
class AppWithCORS:
    def __init__(self, app):
        self.app = CORSMiddlewareClass(
            app,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


# Final app
app = AppWithCORS(CombinedApp(raw_mcp_app, raw_sse_app))


def main():
    """Entry point for the Streamable HTTP server."""
    import uvicorn

    print(f"Encycloscope MCP Server v{__version__}", file=sys.stderr)
    print(f"  Transports:", file=sys.stderr)
    print(f"    - Streamable HTTP: http://{settings.host}:{settings.port}/mcp", file=sys.stderr)
    print(f"    - SSE: http://{settings.host}:{settings.port}/sse", file=sys.stderr)
    print(f"  Articles: {settings.articles_path}", file=sys.stderr)
    print(f"  Auth required: {'Yes' if settings.mcp_token else 'No'}", file=sys.stderr)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
