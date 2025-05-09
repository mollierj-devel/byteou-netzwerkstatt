L'effet papillon de la connaissance, génère des concepts Zettlecasten à partir d'une simple vidéo Youtube

# Pourquoi **byteou-netzwerkstatt** ?

Quel nom étrange pour un projet. C'est un anagramme de **youtube-zettlecasten**.

**byteou** : Ce mot pourrait suggérer "l'information numérique (byte) sous toutes ses formes (ou)" 😉
**netzwerkstatt** : Celui-ci est un mot allemand existant ! Il signifie "atelier de réseau" ou "atelier de travail en réseau". C'est une coïncidence intéressante étant donné que le Zettelkasten est une forme de réseau de connaissances

# A quoi ça sert ?

L'idée est de ne plus perdre son temps lorsque l'on regarde des vidéos youtube intéressantes qui nous apprennent quelque chose.
Quelques notes dans un fichier avec l'ID de la vidéo et cette application s'occupe de tout. 
Elle génère une synthèse, complète vos notes avec le contenu et extrait différent concepts selon la méthode Zettlecasten.
Pour que cela fonctionne la vidéo doit avoir la transcription disponible sur la plateforme et il vous faut une clé API type OpenAI et un modèle LLM performant.

Tout est structuré en Markdown pour améliorer la lisibilité, un fichier pour la note de synthèse et un fichier par concept, l'ensemble est lié et intègre des hashtags intelligents. De quoi nourrir votre vault Obsidian avec de la bonne donnée qui vous correspond.

Tout cela pour quoi faire ? **"L'effet papillon de la connaissance" - Observez comment une simple note peut influencer tout un réseau d'idées grâce à la visualisation dynamique**.

Concretement cela permet de bénéficier d'une cartographie dynamique sans effort qui correspond à votre interprétation.

La vue graphique d'Obsidian n'est pas simplement un gadget visuel attrayant - c'est un outil puissant qui transforme la façon dont nous organisons, visualisons et utilisons nos connaissances. 
Ce n'est pas une simple fonction esthétique - c'est un levier cognitif puissant pour structurer votre réflexion et repérer les angles morts.

"Voyagez dans vos idées comme jamais auparavant" - En quelques clics, explorez les connexions entre vos notes et découvrez de nouvelles pistes de réflexion"
"Pensez en réseau, pas en silos" - La vue graphique ne remplace pas le contenu, elle révèle le sens derrière les liens.
"Des liens plus parlants que des dossiers" - Si chaque note a au moins deux liens, la vue graphique devient plus utile que les dossiers traditionnels.
"Un saut quantique dans la navigation cognitive" - Passez instantanément d'une idée à l'autre en suivant le réseau visuel de vos pensées.

Une extention permettant d'exloiter l'inférence, les similarité cosinus sont à portée de main et vous pourrez exploiter nos réflexions et mémos comme jamais.

# Techniquement comment on s'en sert

## Configuration des paramètres :
config.yaml

Plusieurs provideurs d'API sont supportés : tous les providers compatibles OpenAI (testé via un proxy type LLM et via l'API perplexity) ainsi que google Gemini.

!  La puissance et la performance du modèle affecte le process de l'outil car certain marqueurs sont générés par les LLM via les prompts. Si le modèle est trop "faible" il ne respectera pas les consignes. En règle générale la qualité de l'extraction des concepts Zettlecasten dépend de la "profondeur" du modèle.
L'utilisation nécessite de traiter de grand contexte si vous traiter des videéo longue, le modèle doit aussi être en capacité de le supporter la largeur du contexte est donc un critère a également prendre en compte. Les modèles dits de "raisonnement" n'apportent aucune plus value à byteou-netzwerkstatt
En date de Mai 2025 les deux modèles recommandés sont Claude3.7 ou Gemini2.0 Flash.  

## Installation

```bash
git clone https://github.com/mollierj-devel/byteou-netzwerkstatt.git
cd byteou-netzwerkstatt
```

### Chargement d'un environnement uv avec les dépendances python

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
chmod +x ./byteou-netzwerkstatt.py
```

### Utilisation

```bash
python -m byteou-netzwerkstatt -vvv -help
#ou
python ./byteou-netzwerkstatt.py -vvv -help
#ou
./byteou-netzwerkstatt.py -vvv -help
```