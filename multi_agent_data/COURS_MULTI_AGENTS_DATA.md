# **Cours Multi-agents Data** {#cours-multi-agents-data}

[Cours Multi-agents Data](#cours-multi-agents-data)

[1\) Qu’est-ce qu’un Data Agent ? (What is a Data Agent?)](#1\)-qu’est-ce-qu’un-data-agent-?-\(what-is-a-data-agent?\))

[2\) Construire un workflow multi-agents (Construct a Multi-Agent Workflow)](#2\)-construire-un-workflow-multi-agents-\(construct-a-multi-agent-workflow\))

[3\) Étendre les capacités (Expand Data Agent Capabilities)](#3\)-étendre-les-capacités-\(expand-data-agent-capabilities\))

[4\) Observer la performance et améliorer (Observe Agent Performance & Improve)](#4\)-observer-la-performance-et-améliorer-\(observe-agent-performance-&-improve\))

[5\) Mesurer le GPA et améliorer (Measure Agent’s GPA & Improve)](#5\)-mesurer-le-gpa-et-améliorer-\(measure-agent’s-gpa-&-improve\))

[Atelier technique (exemples guidés)](#atelier-technique-\(exemples-guidés\))

[A) Modifier cortex\_researcher : passer à un RAG local \+ un agent structuré DuckDB](#a\)-modifier-cortex_researcher-:-passer-à-un-rag-local-+-un-agent-structuré-duckdb)

[B) OPTION: remplacer Tavily : DuckDuckGo, SearXNG, Wikipedia](#b\)-option:-remplacer-tavily-:-duckduckgo,-searxng,-wikipedia)

[C) Ajuster les “observations/feedback” et stabiliser](#c\)-ajuster-les-“observations/feedback”-et-stabiliser)

[Tableau de maîtrise — être capable de modifier les nœuds (archi, code, prompts, observation)](#tableau-de-maîtrise-—-être-capable-de-modifier-les-nœuds-\(archi,-code,-prompts,-observation\))

[Récapitulatif (à retenir)](#récapitulatif-\(à-retenir\))

**Objectif du cours**  
**source: [https://www.deeplearning.ai/short-courses/building-and-evaluating-data-agents/](https://www.deeplearning.ai/short-courses/building-and-evaluating-data-agents/)**

À la fin, vous savez :

* **Construire** un agent “data” multi-étapes (planification → orchestration → sous-agents) avec **LangGraph**.  
* **Étendre** ses capacités (web \+ données internes structurées / non structurées).  
* **Observer** ce qui se passe (traces OpenTelemetry, instrumentation) et **évaluer** la qualité (RAG triade) puis la **fiabilité** (GPA).  
* **Itérer** (prompts \+ inline evals) pour “stabiliser” l’agent et améliorer ses scores.

Livrables “cibles” côté code : logique des prompts (planner/executor) \+ nœuds \+ instrumentation \+ feedbacks/évals, tels qu’on les retrouve dans `prompts.py` et `helper.py`.

---

## **1\) Qu’est-ce qu’un Data Agent ? (What is a Data Agent?)** {#1)-qu’est-ce-qu’un-data-agent-?-(what-is-a-data-agent?)}

### **1.1 Définition opérationnelle**

Un **Data Agent** est un système (autonome ou semi-autonome) piloté par LLM qui :

1. comprend une demande en langage naturel,  
2. **décompose** le problème,  
3. **récupère** des données (web, bases, docs, APIs…),  
4. **analyse / visualise**,  
5. **synthétise** une réponse (ou déclenche une action).

### **1.2 Pourquoi “trustworthy” \= aligner Goal / Plan / Action**

Votre agent n’est fiable que si :

* le **Goal** (ce que l’utilisateur veut),  
* le **Plan** (les étapes décidées),  
* les **Actions** (ce qui est réellement exécuté),  
  sont cohérents et alignés (GPA).

---

## **2\) Construire un workflow multi-agents (Construct a Multi-Agent Workflow)** {#2)-construire-un-workflow-multi-agents-(construct-a-multi-agent-workflow)}

### **2.1 Architecture cible (niveau conceptuel)**

L’architecture hiérarchique du cours :

* **Planner** : écrit un plan en étapes (JSON).  
* **Executor** : choisit le prochain agent à exécuter, gère replan / progression.  
* **Sous-agents** :  
  * `web_researcher` (recherche web),  
  * `cortex_researcher` (données internes structurées \+ non structurées),  
  * `chart_generator` (Python tool → graphe),  
  * `chart_summarizer` (résume le graphe),  
  * `synthesizer` (réponse finale texte).

### **2.2 LangGraph : le graphe, l’état, les commandes**

LangGraph fonctionne ici avec 3 idées simples :

1. **Un état partagé** (`State`) : mémoire commune de l’agent (messages, plan, step courant, etc.).  
2. **Des nœuds** (fonctions) : chaque nœud lit l’état, produit un résultat, et renvoie une **Command**.  
3. **Une Command** \=  
   * `update={...}` : mise à jour de l’état  
   * `goto="nom_du_noeud"` : routage vers le prochain nœud.

#### **Extrait (structure de l’état)**

Dans `helper.py`, `State` hérite de `MessagesState` et ajoute les clés utiles : plan, enabled agents, replan flags, agent\_query, etc.

class State(MessagesState):  
    enabled\_agents: Optional\[List\[str\]\]  
    plan: Optional\[Dict\[str, Dict\[str, Any\]\]\]  
    user\_query: Optional\[str\]  
    current\_step: int  
    replan\_flag: Optional\[bool\]  
    last\_reason: Optional\[str\]  
    replan\_attempts: Optional\[Dict\[int, int\]\]  
    agent\_query: Optional\[str\]

### **2.3 Le couple Planner / Executor (et pourquoi le JSON est critique)**

#### **Planner : génère un plan JSON**

* LLM “reasoning” configuré pour **répondre en JSON** (`response_format`).  
* Le plan est stocké dans `state["plan"]` et on repart vers `executor`.

#### **Executor : choisit la prochaine action**

L’executor :

* lit `current_step` et la step correspondante dans `plan`,  
* décide de **replan** si bloqué (avec limite `MAX_REPLANS`),  
* sinon choisit `goto` \+ rédige la **sub-question** (`agent_query`).

Point important de “stabilisation” déjà intégré : **si on vient de replan**, on force l’exécution d’au moins 1 étape du nouveau plan avant de re-replan.

### **2.4 Les prompts de pilotage (prompts.py)**

Le pilotage Planner/Executor dépend énormément des prompts :

* `plan_prompt(state)` : demande au planner de produire des étapes avec un agent par step, format JSON, règles de replan.  
* `executor_prompt(state)` : impose un JSON minimal `{replan,goto,reason,query}` et insiste sur “forward progress”.

**Où modifier le comportement “méta” de l’agent ?**  
Dans `prompts.py`, via la description structurée des agents (`get_agent_descriptions`) et les règles `format_*_guidelines_*`.

---

## **3\) Étendre les capacités (Expand Data Agent Capabilities)** {#3)-étendre-les-capacités-(expand-data-agent-capabilities)}

### **3.1 Du web-only à l’hybride (web \+ interne)**

On ajoute un agent capable de :

* **Text-to-SQL** sur des tables (données structurées),  
* **search** sur des notes/documents (données non structurées),  
  puis on combine ça avec le web.

### **3.2 `cortex_researcher` : outil \+ agent \+ nœud**

Dans `helper.py` :

* `CortexAgentTool` construit une requête Cortex Agents avec 2 tools (`cortex_analyst_text_to_sql` \+ `cortex_search`), puis consomme le stream, récupère `text`, `sql`, `citations`, exécute le SQL si présent et renvoie aussi un `results_str`.  
* le nœud `cortex_agents_research_node` appelle l’agent ReAct outillé, puis pousse un message `name="cortex_researcher"` dans l’état.

### **3.3 `web_researcher` : Tavily \+ ReAct**

Même pattern :

* `TavilySearch(max_results=5)` comme outil.  
* agent ReAct \+ nœud `web_research_node` qui écrit un message `name="web_researcher"` dans l’état.

### **3.4 `chart_generator` \+ `chart_summarizer` \+ `synthesizer`**

* `chart_generator` : ReAct \+ `python_repl_tool` (exécute du Python pour générer un graphique). Attention : exécution de code arbitraire si non sandboxé.  
* `chart_summarizer` : résume en 2–3 phrases et met `final_answer`.  
* `synthesizer` : agrège uniquement les messages “informatifs” (researchers \+ chart\*), puis répond.

---

## **4\) Observer la performance et améliorer (Observe Agent Performance & Improve)** {#4)-observer-la-performance-et-améliorer-(observe-agent-performance-&-improve)}

### **4.1 Tracing : pourquoi et quoi tracer**

Pour diagnostiquer *où* ça casse (plan, exécution, retrieval, synthèse), on trace via OpenTelemetry \+ TruLens.

**Point clé** : instrumenter spécialement les étapes de retrieval pour exposer :

* la **sub-question** (`QUERY_TEXT`),  
* le **contexte récupéré** (`RETRIEVED_CONTEXTS`).

Dans `helper.py`, l’instrumentation est posée sur `cortex_agents_research_node` (et le même pattern s’applique au web researcher).

### **4.2 La RAG triade (même logique appliquée aux agents)**

Même si votre système n’est pas un “RAG pur”, la triade s’applique très bien à un data agent :

1. **Context relevance** : le contexte récupéré est-il pertinent pour la sub-question ?  
2. **Groundedness** : la réponse finale est-elle supportée par les contextes récupérés ?  
3. **Answer relevance** : la réponse finale répond-elle à la question utilisateur ?

Dans `helper.py`, c’est implémenté via `Feedback(...)` TruLens \+ selectors de spans `SpanType.RETRIEVAL`.

**Lecture de symptômes typiques :**

* Context relevance faible → le planner découpe mal, ou l’executor envoie une mauvaise sub-question, ou le retriever est faible.  
* Groundedness faible → hallucination / synthèse non contrainte.  
* Answer relevance faible → mauvaise route (ex: chart sans résumé, ou synthèse qui ignore la question).

---

## **5\) Mesurer le GPA et améliorer (Measure Agent’s GPA & Improve)** {#5)-mesurer-le-gpa-et-améliorer-(measure-agent’s-gpa-&-improve)}

### **5.1 GPA \= Goal / Plan / Action alignment**

On évalue :

* **Plan quality** : le plan est-il bon pour atteindre le goal ?  
* **Plan adherence** : les actions suivent-elles le plan ?  
* **Execution efficiency** : pas de détours, redondances, appels inutiles ?  
* **Logical consistency** : pas de contradictions (entre plan / actions / résultats) ?

Dans `helper.py`, ces feedbacks existent et utilisent un provider “long context” (ex: GPT-4.1) via TruLens.

### **5.2 Pourquoi le GPA “attrape” des pannes que la RAG triade ne voit pas**

Exemples fréquents :

* L’agent récupère du bon contexte (RAG ok) mais **n’exécute pas** les étapes prévues (plan adherence faible).  
* L’agent fait 5 recherches équivalentes, “au cas où” (execution efficiency faible).  
* L’agent change de stratégie en cours de route sans justification (logical consistency faible).

### **5.3 Deux leviers majeurs d’amélioration (L6)**

1. **Améliorer les prompts** : demander explicitement des **préconditions / postconditions / goals** par step, pour aider l’executor à “coller” au plan.  
2. **Inline evals** : évaluer juste après un retrieval et injecter score \+ explication dans la mémoire pour guider replan / next action.

Effet attendu : souvent \+plan adherence et \+groundedness, parfois au prix de \-execution efficiency (plus de recherches), ce qui est un compromis assumé.

---

# **Atelier technique (exemples guidés)** {#atelier-technique-(exemples-guidés)}

## **A) Modifier `cortex_researcher` : passer à un RAG local \+ un agent structuré DuckDB** {#a)-modifier-cortex_researcher-:-passer-à-un-rag-local-+-un-agent-structuré-duckdb}

Vous avez deux approches.

### **Approche A1 — Remplacer le backend de `cortex_researcher` (sans changer le graphe)**

Objectif : garder le même nœud `cortex_agents_research_node` (et le même nom `cortex_researcher`), mais remplacer le “tool” Snowflake par 2 tools locaux.

**1\) Un tool “RAG local” (non structuré)**  
Exemple (squelette) :

from langchain\_core.tools import tool

@tool  
def local\_rag\_search(query: str) \-\> str:  
    """  
    Retourne des extraits pertinents depuis une base locale (vector store).  
    """  
    \# 1\) récupérer top-k chunks via retriever  
    \# docs \= retriever.get\_relevant\_documents(query)  
    \# 2\) formater (source \+ extrait)  
    \# return format\_docs(docs)  
    return "TODO: implémenter retriever local \+ formatage"

**2\) Un tool DuckDB (structuré)**  
Exemple simple (SQL direct) :

import duckdb  
from langchain\_core.tools import tool

@tool  
def duckdb\_sql(query\_sql: str) \-\> str:  
    """  
    Exécute du SQL sur une base DuckDB locale et renvoie un tableau texte.  
    """  
    con \= duckdb.connect("sales.duckdb")  
    df \= con.execute(query\_sql).df()  
    return df.to\_string(index=False)

**3\) Envelopper en ReAct agent \+ node**  
Même pattern que `cortex_agent = create_react_agent(...)` dans `helper.py`.

from langgraph.prebuilt import create\_react\_agent  
from langchain\_openai import ChatOpenAI

llm \= ChatOpenAI(model="gpt-4o")

local\_cortex\_agent \= create\_react\_agent(  
    llm,  
    tools=\[local\_rag\_search, duckdb\_sql\],  
    prompt="Tu es un researcher. Utilise les tools locaux. Ne fais rien d'autre."  
)

def cortex\_agents\_research\_node(state):  
    query \= state.get("agent\_query", state.get("user\_query", ""))  
    agent\_response \= local\_cortex\_agent.invoke({"messages": query})  
    \# pousser message name="cortex\_researcher" comme avant

**Stabilisation recommandée :**

* imposer un format de sortie du tool (ex: “SOURCES: … EXCERPTS: …”),  
* limiter top-k et la taille totale,  
* tracer/instrumenter comme avant (QUERY\_TEXT \+ RETRIEVED\_CONTEXTS).

---

### **Approche A2 — Ajouter deux nouveaux nœuds (facilite le contrôle de fin dans le planner)**

Objectif : séparer clairement :

* `local_rag_researcher` (docs)  
* `duckdb_researcher` (SQL local)

**Ce que ça implique côté prompts :**  
Dans `prompts.py`, la liste “allowed agents” est actuellement fermée (`web_researcher`, `cortex_researcher`, etc.). Pour ajouter un agent, il faut :

1. l’ajouter dans `get_agent_descriptions()`,  
2. l’autoriser dans `_get_enabled_agents()` (set `allowed`),  
3. l’inclure dans les enums du planner/executor (`planner_agent_enum`, JSON schema).

---

## **B) OPTION: remplacer Tavily : DuckDuckGo, SearXNG, Wikipedia** {#b)-option:-remplacer-tavily-:-duckduckgo,-searxng,-wikipedia}

Dans `helper.py`, le web tool est `TavilySearch(max_results=5)`.

### **B1 — DuckDuckGo (simple)**

(Exemple typique LangChain Community)

\# pip install langchain-community duckduckgo-search  
from langchain\_community.tools import DuckDuckGoSearchRun

ddg\_tool \= DuckDuckGoSearchRun()  
web\_search\_agent \= create\_react\_agent(llm, tools=\[ddg\_tool\], prompt=...)

### **B2 — SearXNG (contrôle infra / on-prem)**

Vous remplacez le tool par un wrapper HTTP interne vers votre instance SearXNG (format JSON), puis vous renvoyez un texte normalisé (title/url/snippet).

### **B3 — Wikipedia (quand on veut une source encyclopédique)**

from langchain\_community.utilities import WikipediaAPIWrapper  
from langchain\_community.tools import WikipediaQueryRun

wiki \= WikipediaQueryRun(api\_wrapper=WikipediaAPIWrapper())  
web\_search\_agent \= create\_react\_agent(llm, tools=\[wiki\], prompt=...)

**Conseil stabilisation** : quel que soit le provider, normalisez la sortie tool (mêmes champs, même structure) pour rendre la groundedness plus stable.

---

## C) Ajuster les “observations/feedback” et stabiliser {#c)-ajuster-les-“observations/feedback”-et-stabiliser}

### C1 — Observations minimales (ce que vous devez voir pour comprendre ce qui se passe)

Objectif : quand un run est “mauvais”, pouvoir répondre en 2 minutes à :

* **Quel nœud** a fait quoi ?  
* **Quelle requête** a été envoyée aux tools ?  
* **Quel contexte** a été ramené ?  
* Est-ce que la réponse est **ancrée** dans ce contexte ou inventée ?

### **1\) Toujours tracer les nœuds de retrieval (web \+ cortex)**

Dans votre `helper.py`, `cortex_agents_research_node` est déjà instrumenté en span RETRIEVAL via `@instrument(...)` et remplit `QUERY_TEXT` et `RETRIEVED_CONTEXTS`.  
**Mais `web_research_node` ne l’est pas** : c’est la modification la plus simple et la plus rentable.

**Modification légère** : instrumenter `web_research_node` avec le même décorateur (même type de span, mêmes attributs).  
➡️ Résultat : la triade RAG verra **aussi** les retrievals web (au lieu d’avoir une observabilité partielle).

### **2\) Garantir que `agent_query` ne soit jamais vide**

Aujourd’hui, `web_research_node` lit `state.get("agent_query")` et l’envoie tel quel au tool.  
**Modification légère** : fallback systématique sur `user_query` (ou le dernier message utilisateur) si `agent_query` est `None`.  
➡️ Résultat : moins de runs “bizarres” (tool appelé avec `None`, spans vides, evals incohérents).

### **3\) Normaliser le “contexte récupéré” (format stable)**

Vous voulez que l’élève puisse lire un run et comprendre vite. Le minimum :

* chaque chunk/extrait doit avoir : `source`, `id`, `texte`.  
* si cortex renvoie SQL \+ results, gardez un format stable (ex: bloc “SQL:” puis “RESULTS:”).

Ce n’est pas “optimisation chunking”, c’est juste : **rendre lisible ce que l’agent a réellement vu**.

---

### C2 — Feedbacks minimalistes (ce qu’on mesure, et pourquoi)

Vous avez déjà le bon “set minimal” dans `helper.py` :

* **RAG triade**  
  * Context relevance (query ↔ retrieved\_contexts)  
  * Groundedness (réponse ↔ contexts)  
  * Answer relevance (question ↔ réponse)  
* **GPA**  
  * plan quality, plan adherence, execution efficiency, logical consistency

### **Lecture simple (pour ne pas perdre les élèves)**

* **Context relevance bas** : mauvais retrieval *ou* mauvaise sub-question envoyée (souvent un problème executor/prompt, pas “chunking”).  
* **Groundedness bas** : la synthèse invente / ne cite pas / n’utilise pas le contexte → durcir le prompt du synthesizer ou la contrainte “use only context”.  
* **Answer relevance bas** : réponse à côté → route/nœud choisi ou synthèse qui ignore le besoin utilisateur.  
* **GPA bas** : problème d’orchestration (replan loop, étapes non suivies, appels inutiles), donc à corriger côté planner/executor.

---

### C3 — Comment ça aide à “stabiliser” (sans partir dans un cours d’optimisation)

Le but n’est pas d’optimiser 50 paramètres. Le but est de corriger **les trois causes dominantes** d’instabilité :

### **1\) “Le mauvais nœud est appelé” / “le plan part en vrille”**

→ Regardez GPA (plan adherence / efficiency / consistency)  
→ Ajustez surtout :

* les règles de replan (limites déjà présentes : `MAX_REPLANS`)  
* le prompt executor (raison \+ goto \+ query).

### **2\) “Le retrieval est bon mais la réponse invente”**

→ Groundedness bas  
→ Ajustez :

* prompt du synthesizer (obligation de s’appuyer sur messages `web_researcher`/`cortex_researcher`)  
* format de sortie (citations / références internes, “si info absente → je ne sais pas”).

### **3\) “Le retrieval est mauvais”**

→ Context relevance bas  
→ Ajustez **d’abord** :

* la sub-question `agent_query` produite par l’executor (c’est souvent la vraie source du problème)  
* la source (web vs cortex) / top\_k (si vous en avez)  
  Et **seulement ensuite** (si nécessaire) les paramètres de chunking.

---

### C4 — Inline eval (optionnel, mais utile et simple)

Vous avez déjà une logique de replan dans `executor_node` avec garde-fous.  
La version “légère” d’inline eval, c’est juste :

* Après un nœud retrieval, évaluer rapidement **context relevance**,  
* si le score est trop bas : ajouter un message “feedback” dans l’état (ou lever un flag) pour forcer :  
  * reformulation de la query,  
  * ou changement de source,  
  * ou replan.

Ça sert à **éviter** d’aller en synthèse avec un mauvais contexte (stabilisation), sans transformer le cours en tuning de chunking.

# 

# **Tableau de maîtrise — être capable de modifier les nœuds (archi, code, prompts, observation)** {#tableau-de-maîtrise-—-être-capable-de-modifier-les-nœuds-(archi,-code,-prompts,-observation)}

Les points ci-dessous s’appuient sur l’architecture/nœuds/patterns présents dans `helper.py` (nodes, tools, instrumentation, evals) et `prompts.py` (governance planner/executor).

| Compétence à maîtriser | Où intervenir | Ce que l’élève doit savoir faire (checklist) | Test de validation | Pièges fréquents |
| ----- | ----- | ----- | ----- | ----- |
| Ajouter/remplacer un **retriever** (web) | `helper.py` | 1\) remplacer le tool (Tavily → DDG/SearXNG/Wiki) 2\) garder la sortie normalisée 3\) conserver `name="web_researcher"` dans le message | la RAG triade: context relevance stable, groundedness ↑ | tool renvoie trop de texte / format instable |
| Remplacer `cortex_researcher` par **RAG local** | `helper.py` (tool \+ node), éventuellement `prompts.py` | 1\) créer un tool `local_rag_search` 2\) wrapper ReAct 3\) node écrit message `name="cortex_researcher"` 4\) instrumenter retrieval spans | même requête → réponse grounded \+ citations internes | oublier instrumentation → evals inutilisables |
| Ajouter un agent **DuckDB structuré** | `helper.py` \+ `prompts.py` si nouveau nœud | 1\) tool `duckdb_sql` 2\) option text-to-sql (prompt) 3\) soit remplacer cortex, soit nouveau nœud \+ ajout dans prompts | requête SQL → tableau \+ synthèse correcte | SQL dangereux / pas de guardrails / schéma inconnu |
| Modifier le **planner** | `prompts.py` | 1\) changer gabarit JSON 2\) ajouter pre/post-conditions, goal par step 3\) maintenir JSON valide | plan quality ↑ ; plan adherence ↑ | casser l’enum agent / JSON invalide |
| Modifier l’**executor** | `prompts.py` \+ `helper.py` | 1\) ajuster critères replan 2\) exploiter scores inline eval 3\) limiter replans | plan adherence ↑ ; efficiency contrôlée | replan en boucle / step index incohérent |
| Ajouter des **observations** (traces) | `helper.py` | 1\) instrumenter chaque retrieval node 2\) exposer QUERY\_TEXT \+ RETRIEVED\_CONTEXTS 3\) log erreurs | dashboard exploitable, debug rapide | spans non typés retrieval → selectors ne matchent pas |
| Ajouter un **feedback** (offline) | `helper.py` | 1\) définir Feedback 2\) selectors corrects (trace\_level vs retrieval spans) 3\) agrégation | leaderboard montre le signal attendu | sélectionner la mauvaise source (input/output) |
| Ajouter un **feedback inline** (runtime) | notebooks L6 / code équivalent | 1\) décorer nœud retrieval 2\) injecter score+reason dans state messages 3\) l’executor le lit | hausse groundedness \+ adherence | seuil mal réglé → sur-retrieval (efficiency ↓) |

# **Récapitulatif (à retenir)** {#récapitulatif-(à-retenir)}

* Un data agent fiable \= **récupération \+ analyse \+ synthèse**, mais surtout **Goal/Plan/Action alignés** (GPA).  
* LangGraph vous donne un cadre simple : **State \+ Nodes \+ Command(update/goto)** pour orchestrer un workflow multi-agents.  
* La **RAG triade** sert à mesurer *qualité de récupération \+ ancrage \+ pertinence* (context relevance, groundedness, answer relevance).  
* Le **GPA** mesure *qualité du plan et de l’exécution* (plan quality, adherence, efficiency, consistency).  
* Les deux leviers d’amélioration les plus rentables ici :  
  1. **prompts** plus “contractuels” (pré/post conditions),  
  2. **inline evals** qui pilotent replanning et stratégie en temps réel.

Si vous voulez, je peux aussi fournir une version “fiche de TP” (exercices pas-à-pas) structurée exactement comme L2→L6 : baseline → ajout cortex → instrumentation → RAG triade → GPA → amélioration (prompts \+ inline evals).