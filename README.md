# Encycloscope MCP Server

Serveur [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) pour explorer l'*Encyclopédie* de Diderot et d'Alembert.

Permet à un LLM (Claude, etc.) d'interroger les 74 000+ articles de l'Encyclopédie via des outils structurés : recherche sémantique, analyse d'idiomes, réseaux d'auteurs et de renvois.

## Fonctionnalités

### Recherche et consultation

- **encyclopedie_search** : Recherche sémantique (via d'AlemBERT + FAISS) ou textuelle
- **encyclopedie_search_by_vedette** : Recherche par mot-entrée (vedette)
- **encyclopedie_get_article** : Récupère un article complet
- **encyclopedie_get_definition** : Extrait la définition d'un terme

### Analyse d'idiomes

- **encyclopedie_count_term** : Compte les occurrences d'un terme dans le corpus
- **encyclopedie_compare_term_usage** : Compare l'usage d'un terme par différents auteurs
- **encyclopedie_get_term_contexts** : Récupère les contextes d'usage avec pagination et filtres

### Réseaux des textes et auteurs

- **encyclopedie_get_renvois** : Récupère les références croisées d'un article
- **encyclopedie_get_author_network** : Analyse le réseau d'un auteur
- **encyclopedie_visualize_author_network** : Graphe interactif D3.js (MCP Apps)
- **encyclopedie_list_authors** / **encyclopedie_list_domains** : Listes des auteurs et domaines
- **encyclopedie_search_by_author** / **encyclopedie_search_by_domain** : Articles par auteur ou domaine

## Installation

```bash
git clone <repo-url>
cd Encycloscope-mcp

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# ou : .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -e .
```

## Données

Les données ne sont pas incluses dans le dépôt (trop volumineuses). Deux niveaux d'utilisation :

### Niveau 1 : Recherche textuelle (minimum)

Placer le fichier CSV des articles dans `data/articles.csv`. Ce fichier doit contenir les colonnes :

| Colonne | Description |
|---------|-------------|
| `articleID` | Identifiant unique (ex : `v8-2410-1`) |
| `entreeID` | Identifiant de l'entrée |
| `vedette` | Mot-entrée de l'article |
| `contenu_clean` | Texte nettoyé |
| `auteurs` | Auteur(s) de l'article |
| `domainesEnccre` | Domaine(s) de l'article |
| `idRenvois_art` | IDs des articles référencés (séparés par `;`) |

Le corpus est issu du projet [ENCCRE](https://enccre.academie-sciences.fr/) (Édition Numérique Collaborative et CRitique de l'*Encyclopédie*).

### Niveau 2 : Recherche sémantique (recommandé)

Pour activer la recherche sémantique, construire l'index d'AlemBERT + FAISS :

```bash
python scripts/build_dalembert_index.py --articles data/articles.csv
```

Cela génère dans `data/` :
- `dalembert_index.faiss` : Index FAISS pour la recherche par similarité
- `dalembert_index_chunks.pkl` : Métadonnées des chunks d'articles

Requiert un GPU (ou beaucoup de patience en CPU). Le modèle [d'AlemBERT](https://huggingface.co/pjox/dalembert) est téléchargé automatiquement depuis HuggingFace.

## Configuration

Copier `.env.example` vers `.env` et ajuster si nécessaire :

```env
ARTICLES_PATH=./data/articles.csv
HOST=0.0.0.0
PORT=8001
```

## Utilisation

### Avec Claude Desktop (stdio)

Ajouter dans `claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "encycloscope": {
      "command": "python",
      "args": ["-m", "encycloscope_mcp.server"],
      "cwd": "/chemin/vers/Encycloscope-mcp/src",
      "env": {
        "ARTICLES_PATH": "/chemin/vers/data/articles.csv"
      }
    }
  }
}
```

### Avec Claude Web (HTTP Streamable)

```bash
encycloscope-mcp
# ou
python -m encycloscope_mcp.http_streamable
```

Le serveur démarre sur `http://localhost:8001/mcp`.

### Docker

```bash
docker build -t encycloscope-mcp .
docker run -p 8001:8001 -v ./data:/app/data encycloscope-mcp
```

## Exemples de requêtes

### Recherche sémantique

```
"Qu'est-ce que le luxe selon l'Encyclopédie ?"
"Comment fonctionne la circulation du sang ?"
"La liberté naturelle de l'homme"
```

### Analyse d'idiomes

```python
# Compare comment Saint-Lambert, Jaucourt et d'autres utilisent "luxe"
encyclopedie_compare_term_usage(term="luxe")

# Tous les contextes où Diderot emploie "liberté"
encyclopedie_get_term_contexts(term="liberté", author="Diderot")

# Compter les occurrences, filtrer les faux positifs
encyclopedie_count_term(term="luxe")
encyclopedie_get_term_contexts(term="luxe", exclude_patterns=["luxemb", "luxeuil"])
```

### Exploration des réseaux

```python
# Renvois d'un article
encyclopedie_get_renvois(article_id="v10-1234-1")

# Réseau d'un auteur (connexions via renvois et domaines partagés)
encyclopedie_get_author_network(author="Diderot")
```

## Architecture

```
Encycloscope-mcp/
├── src/
│   └── encycloscope_mcp/
│       ├── __init__.py            # Version
│       ├── http_streamable.py     # Serveur HTTP + tous les outils MCP
│       └── core/
│           ├── config.py          # Configuration (pydantic-settings)
│           └── corpus.py          # Gestion du corpus, recherche, réseaux
├── scripts/
│   └── build_dalembert_index.py   # Construction de l'index sémantique
├── data/                          # Données (non versionnées, voir ci-dessus)
│   └── .gitkeep
├── Dockerfile
├── pyproject.toml
├── .env.example
├── LICENSE
└── README.md
```

## Citation

Si vous utilisez cet outil dans un travail de recherche, merci de le citer :

```bibtex
@software{blangeois2026encycloscope,
  author       = {Blangeois, Morgan},
  title        = {Encycloscope: MCP Server for Exploring the Encyclopédie of Diderot and d'Alembert},
  year         = {2026},
  url          = {https://github.com/MSH-UCA/Encycloscope-mcp},
  version      = {1.0.0},
  note         = {Uses d'AlemBERT (pjox/dalembert) for semantic search over 18th-century French}
}
```

## Crédits

- **[d'AlemBERT](https://huggingface.co/pjox/dalembert)** : Modèle RoBERTa pré-entraîné sur le français historique (XVIe-XXe siècle), par Pedro Ortiz Suarez (Inria)
- **[ENCCRE](https://enccre.academie-sciences.fr/)** : Édition Numérique Collaborative et CRitique de l'*Encyclopédie*, Académie des sciences
- **[Model Context Protocol](https://modelcontextprotocol.io/)** : Protocole ouvert pour connecter les LLM à des sources de données, par Anthropic
- **MSH Clermont-Ferrand** / **Université Clermont Auvergne** : Hébergement et infrastructure

## Licence

MIT. Voir [LICENSE](LICENSE).
