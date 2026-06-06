#  H2 Morocco — Plateforme d'Aide à la Décision pour l'Hydrogène Vert

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/PostgreSQL-15-316192?logo=postgresql" />
  <img src="https://img.shields.io/badge/PyPSA-Energy%20Modelling-green" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/Status-Research%20%2F%20PFE-orange" />
</p>

> **Plateforme complète d'analyse techno-économique pour la planification de la chaîne de valeur hydrogène vert au Maroc (2024–2050).**  
> Couvre 12 sites de production, 4 technologies d'électrolyseurs, 7 vecteurs de stockage et l'optimisation du réseau logistique international.

---

##  Table des Matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Architecture du projet](#-architecture-du-projet)
3. [Modules & fichiers](#-modules--fichiers)
4. [Base de données](#-base-de-données-postgresql)
5. [Installation](#-installation)
6. [Configuration](#️-configuration)
7. [Utilisation](#-utilisation)
8. [Méthodologie scientifique](#-méthodologie-scientifique)
9. [Sites couverts](#-sites-couverts)
10. [Sources & références](#-sources--références)
11. [Contribuer](#-contribuer)
12. [Licence](#-licence)

---

##  Vue d'ensemble

Ce projet constitue un outil d'aide à la décision pour l'évaluation du potentiel de production d'**hydrogène vert au Maroc**. Il intègre :

- **Analyse multi-critères MACBETH** (Step 1) — sélection et scoring des filières H₂ par région
- **Optimisation énergétique PyPSA + NSGA-II** (Step 2) — dimensionnement optimal PV/éolien/électrolyseur/batterie sur 8 760 h
- **Modèle de stockage LCOS** (Step 3) — évaluation du coût de stockage (GH2, LH2, NH3, LOHC, Caverne…)
- **Planification logistique MILP** (Step 4) — optimisation du réseau de transport multi-modal 2024–2050
- **Dashboard Streamlit interactif** — visualisation temps réel de tous les résultats

Le tout est alimenté par une base PostgreSQL construite à partir de **données réelles** (CDER, MASEN, NASA POWER, Solargis, IEA, IRENA) et intégrée via SQLAlchemy.

---

## Architecture du Projet

```
h2morocco/
│
├──  BASEDEDONNEES.py          # Construction de la base de données (Monte Carlo + validation)
├──  DATABASEBUILDER.py        # Création des tables PostgreSQL manquantes
├──  LOADTOPOST.py             # Chargement des CSV dans PostgreSQL
├──  db_connector.py           # Connecteur unifié (PostgreSQL ↔ CSV fallback)
│
├──  ETAPE1MACBETH.py          # Étape 1 — Analyse MACBETH (scoring multi-critères)
├──  ETAPE2.py                  # Étape 2 — Optimisation production (PyPSA + NSGA-II)
├──  MODELESTOCKAGE.py          # Étape 3 — Modèle LCOS stockage H₂
├──  ETAPE4TRANSPORT.py         # Étape 4 — MILP transport & logistique
│
├──  engine.py                  # Moteur de calcul central (toutes formules physiques/éco)
├──  app.py                     # Dashboard Streamlit (interface utilisateur)
│
├── requirements.txt              # Dépendances Python
├── .env.example                  # Modèle de configuration (à copier en .env)
└── README.md
```

**Flux de données :**

```
BASEDEDONNEES.py
      ↓ génère CSVs
LOADTOPOST.py / DATABASEBUILDER.py
      ↓ charge dans PostgreSQL
db_connector.py ←──────────────────────────────────────┐
      ↓ fournit les données                             │
ETAPE1MACBETH.py → ETAPE2.py → MODELESTOCKAGE.py → ETAPE4TRANSPORT.py
      ↓ résultats consolidés dans engine.py
app.py (Streamlit Dashboard)
```

---

##  Modules & Fichiers

### `BASEDEDONNEES.py` — Construction de la Base de Données
**Rôle :** Génère l'ensemble des tables de données brutes avec ancrage littérature et correction Maroc.

Contenu :
- Ressources énergétiques par région (GHI, DNI, vitesse vent, CF hybride)
- Technologies de production (CAPEX/OPEX PEM, AEL, SOEC par année 2024–2050)
- Technologies de stockage (densités, TRL, coûts)
- Corridors de transport (distances, CAPEX pipelines, LCOT)
- Scénarios de demande nationale et benchmark compétiteurs
- Émissions CO₂ par filière + certifications (RFNBO, CertifHy)
- **10 000 simulations Monte Carlo** pour intervalles de confiance

Outputs : CSV nettoyés dans `~/Downloads/H2Morocco222_Outputs/csv/`

---

### `DATABASEBUILDER.py` — Initialisation PostgreSQL
**Rôle :** Vérifie et crée les tables manquantes dans le schéma `h2morocco`.

Tables créées :
| Table | Description |
|-------|-------------|
| `t5_parametres_economiques` | Paramètres économiques (WACC, inflation, taux) |
| `t7a_emissions_co2` | Émissions CO₂ par filière (min/mode/max) |
| `t7b_certifications` | Seuils RFNBO, CertifHy, primes prix |
| `t8_projets_reference_maroc` | Projets de référence (Noor, Tarfaya, Dakhla…) |
| `t10_profils_horaires` | Profils 8 760 h (CF PV, CF éolien, GHI, T°C) |

---

### `LOADTOPOST.py` — Chargement CSV → PostgreSQL
**Rôle :** Pipeline de chargement batch depuis les CSV vers PostgreSQL.

Fonctionnalités :
- Détection automatique du dossier CSV (principal + fallback)
- Insertion par batch de 500 lignes (`method="multi"`)
- Gestion `NaN → NULL` PostgreSQL
- Log complet des tables chargées et des erreurs

---

### `db_connector.py` — Connecteur Unifié
**Rôle :** Interface unique entre tous les modules et la base de données.

Modes d'utilisation :
```bash
python db_connector.py --load      # Charge les CSV dans PostgreSQL
python db_connector.py --calculs   # Lance tous les calculs et stocke les résultats
python db_connector.py --check     # Vérifie l'état de la base
python db_connector.py --all       # load + calculs en séquence
```

Fonctionnalité clé : **fallback CSV automatique** si PostgreSQL est indisponible.

---

### `ETAPE1MACBETH.py` — Analyse MACBETH
**Rôle :** Scoring et classement multi-critères des filières H₂ par région.

Méthode :
- **MACBETH** (Measuring Attractiveness by a Categorical Based Evaluation Technique)
- Critères : LCOH, émissions CO₂, fiabilité, TRL, accessibilité eau
- Résolution par **programmation linéaire** (scipy.optimize.linprog)
- Vérification et correction de transitivité avant LP
- Ancrage s_best=1 / s_worst=0 pour faisabilité garantie

Corrections physiques (v4) :
- Électrolyse → EnR pure (CI_enr = 20 gCO₂/kWh)
- Compression/liquéfaction → réseau hybride (CI_grid par région)
- NH3/e-méthanol → mix industriel (CI_ind = CI_grid × 1.15)
- OPEX tech-spécifique : NH3 (catalyseur Haber-Bosch 3 ans), LH2 (boil-off 0.2%/j), LOHC (dégradation huile DBT)

---

### `ETAPE2.py` — Optimisation Production H₂ Vert
**Rôle :** Dimensionnement optimal du système de production EnR + électrolyseur.

Variables décisionnelles :
| Variable | Description |
|----------|-------------|
| x₁ | Capacité PV (MW) |
| x₂ | Capacité éolienne (MW) |
| x₃ | Capacité électrolyseur (MW) |
| x₄ | Capacité batterie (MWh) |

Objectifs bi-critères :
- **f₁** : Minimiser le LCOH (€/kgH₂)
- **f₂** : Maximiser la fiabilité (taux de couverture demande)

Modèles d'intermittence (5 modes) :
- **Mode 1** : Profils déterministes
- **Mode 2** : AR(1) — corrélation temporelle, rafales & calmes
- **Mode 3** : HMM — 3 régimes météo (Ensoleillé / Nuageux / Couvert)
- **Mode 4** : AR(1) + HMM combinés
- **Mode 5** : Monte Carlo N tirages → IC 90% sur LCOH et fiabilité

Simulation énergétique : **PyPSA** sur 8 760 heures (annuelle horaire)  
Optimisation multi-objectifs : **NSGA-II** (algorithme génétique)

---

### `MODELESTOCKAGE.py` — Modèle de Stockage H₂
**Rôle :** Calcul du LCOS (Levelized Cost of Storage) pour 7 vecteurs de stockage.

Technologies couvertes :
| Code | Technologie | Densité (kg/m³) | TRL |
|------|------------|-----------------|-----|
| GH2_350bar | H₂ gazeux 350 bar | 23.5 | 9 |
| GH2_700bar | H₂ gazeux 700 bar | 40.2 | 9 |
| LH2 | H₂ liquide | 70.8 | 7 |
| NH3 | Ammoniac | 121 | 9 |
| LOHC | LOHC (DBT) | 57 | 7 |
| Caverne | Caverne saline | 120 | 8 |
| eMethanol | e-Méthanol | 140 | 7 |

Nouveautés v2.2 :
- Jours de stockage dynamiques calculés depuis profil 8 760 h
- Analyse de sensibilité **Sobol** (SALib)
- Argument CLI `--jours` pour forçage manuel

---

### `ETAPE4TRANSPORT.py` — MILP Transport Multi-Période
**Rôle :** Optimisation du réseau logistique H₂ (2024–2050) par MILP.

Formulation mathématique :
- **Minimisation** de la VAN des coûts sur l'horizon 2024–2050
- Variables binaires : construction d'arcs et de hubs
- Variables continues : flux H₂ sur chaque arc [ktH₂/an]
- 8 contraintes (bilan flux, capacités, irréversibilité, budget…)

Périodes : 2024, 2030, 2035, 2040, 2050  
Scénarios : optimiste / central / pessimiste  
Modes de transport : pipeline, camion, ship (NH3, LH2, LOHC)

---

### `engine.py` — Moteur de Calcul Central
**Rôle :** Bibliothèque de toutes les formules physiques et économiques, partagée par `app.py` et tous les modules.

Contient :
- CRF (Capital Recovery Factor), Haversine, LCOH, LCOS, LCOT
- Modèles éoliens (Weibull, courbe de puissance)
- Physique de compression et de liquéfaction H₂
- Calcul émissions CO₂ par filière et par région
- Scoring MACBETH simplifié pour le dashboard

---

### `app.py` — Dashboard Streamlit
**Rôle :** Interface web interactive pour visualiser tous les résultats.

Onglets :
1. **Carte & Sites** — Localisation des 12 sites, ressources solaires/éoliennes
2. **MACBETH** — Scoring interactif par région et par filière
3. **Production** — Dimensionnement optimal, courbes LCOH, fiabilité
4. **Stockage** — Comparaison LCOS par technologie, analyse de sensibilité
5. **Transport** — Réseau logistique, LCOT, carte flux
6. **Scénarios 2050** — Projections à long terme, comparaison internationale

---

## 🗄 Base de Données PostgreSQL

### Schéma `h2morocco`

| Table | Lignes (typ.) | Description |
|-------|---------------|-------------|
| `t1_ressources` | 12 | Ressources énergétiques par région |
| `t2_technologies_production` | ~50 | CAPEX/OPEX électrolyseurs × années |
| `t3_technologies_stockage` | ~35 | Coûts stockage × vecteurs × années |
| `t4_corridors_resume` | ~40 | Corridors de transport + LCOT |
| `t5_parametres_economiques` | ~20 | WACC, taux d'actualisation, inflation |
| `t6a_demande_nationale` | ~25 | Demande H₂ nationale 2024–2050 |
| `t6b_benchmark_competiteurs` | ~30 | Comparatif international (LCOH) |
| `t7a_emissions_co2` | ~7 | Émissions CO₂ par filière |
| `t7b_certifications` | ~5 | Seuils RFNBO, CertifHy |
| `t8_projets_reference_maroc` | ~15 | Projets solaires/éoliens réels |
| `t9_scenarios_long` | ~150 | Scénarios 2024–2050 (long format) |
| `t10_profils_horaires` | ~105K | Profils horaires 8 760 h × régions |
| `dim_region` | 12 | Dimensions régions |
| `dim_technologie` | ~10 | Dimensions technologies |

---

##  Installation

### Prérequis

- Python 3.10 ou supérieur
- PostgreSQL 14 ou supérieur (optionnel — fallback CSV disponible)
- Git

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/h2-morocco.git
cd h2-morocco
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales :**

```
streamlit>=1.32
plotly>=5.20
pandas>=2.0
numpy>=1.26
sqlalchemy>=2.0
psycopg2-binary>=2.9
pypsa>=0.26
scipy>=1.12
SALib>=1.4
matplotlib>=3.8
requests>=2.31
```

### 4. Configurer PostgreSQL (optionnel)

```bash
# Créer la base de données
psql -U postgres -c "CREATE DATABASE h2morocco_db;"
psql -U postgres -c "CREATE SCHEMA h2morocco;"
```

Copier le fichier de configuration :

```bash
cp .env.example .env
```

Éditer `.env` avec vos identifiants :

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=h2morocco_db
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe
```

---

## 🛠️ Configuration

### Sans PostgreSQL (mode CSV)

Le projet fonctionne **entièrement en mode CSV** si PostgreSQL n'est pas disponible. Le connecteur `db_connector.py` bascule automatiquement.

Aucune configuration supplémentaire n'est nécessaire dans ce cas.

### Avec PostgreSQL

Modifier la variable `DB_URL` dans les fichiers concernés, ou mieux, utiliser le fichier `.env` :

```python
# db_connector.py et autres modules
DB_URL = "postgresql+psycopg2://user:password@localhost:5432/h2morocco_db"
```

> ⚠️ **Important :** Ne jamais committer de mots de passe dans le code. Utilisez toujours des variables d'environnement ou un fichier `.env` (listé dans `.gitignore`).

---

##  Utilisation

### Étape 0 — Construire la base de données

```bash
python BASEDEDONNEES.py
```

Génère tous les CSV dans `~/Downloads/H2Morocco222_Outputs/csv/`.

### Étape 1 — Initialiser PostgreSQL (si utilisé)

```bash
# Créer les tables
python DATABASEBUILDER.py

# Charger les données
python LOADTOPOST.py
```

Ou avec le connecteur unifié :

```bash
python db_connector.py --all
```

### Étape 2 — Lancer les analyses scientifiques

```bash
# Analyse MACBETH (scoring multi-critères)
python ETAPE1MACBETH.py

# Optimisation production (PyPSA + NSGA-II) — peut prendre 5–30 min
python ETAPE2.py

# Modèle de stockage LCOS
python MODELESTOCKAGE.py

# Planification logistique MILP
python ETAPE4TRANSPORT.py
```

### Étape 3 — Lancer le Dashboard

```bash
streamlit run app.py
```

Ouvre automatiquement `http://localhost:8501` dans le navigateur.

---

##  Méthodologie Scientifique

### LCOH — Levelized Cost of Hydrogen

```
LCOH = (CAPEX × CRF + OPEX_fixe) / H2_annuel + OPEX_variable
```

Avec `CRF(r, n) = r(1+r)ⁿ / ((1+r)ⁿ - 1)`

### Émissions CO₂

Les intensités carbone sont calculées en distinguant trois sources d'énergie :
- **EnR pure** : CI_enr = 20 gCO₂/kWh (électrolyse)
- **Réseau hybride** : CI_grid = f(CF région) (compression, liquéfaction)
- **Mix industriel** : CI_ind = CI_grid × 1.15 (synthèse NH₃, e-méthanol)

### Critères de certification
- **H₂ vert UE (RFNBO)** : < 3.38 kgCO₂/kgH₂ (Règlement délégué UE 2023/1184)
- **H₂ vert premium** : < 1.0 kgCO₂/kgH₂ (IEA Hydrogen 2023)
- **Référence H₂ gris (SMR)** : 10.0 kgCO₂/kgH₂

### Monte Carlo
10 000 tirages par simulation. Distributions utilisées :
- CAPEX : triangulaire (min / mode / max) via `scipy.stats.triang`
- LCOH : log-normale
- Facteurs de charge : normale tronquée

---

##  Sites Couverts

| Site | Latitude | GHI (kWh/m²/an) | Vitesse Vent (m/s) | Accès Port (km) |
|------|----------|-----------------|-------------------|-----------------|
| Laayoune | 27.13°N | 2 160 | 7.8 | 20 |
| Dakhla | 23.68°N | 2 155 | 9.0 | 8 |
| Boujdour | 26.10°N | 2 175 | 8.5 | 12 |
| Guelmim | 28.99°N | 1 940 | 5.5 | 55 |
| Jorf Lasfar | 33.11°N | 1 900 | 5.0 | 2 |
| Ouarzazate | 30.92°N | 2 180 | 5.5 | 350 |
| Agadir | 30.43°N | 2 095 | 5.5 | 5 |
| Tanger | 35.76°N | 1 840 | 9.5 | 15 |
| Casablanca | 33.57°N | 1 875 | 4.5 | 40 |
| Nador | 35.17°N | 1 785 | 5.8 | 8 |
| Marrakech | 31.63°N | 2 085 | 4.0 | 230 |
| Midelt | 32.68°N | 2 200 | 5.5 | 380 |

Sources : CDER/Solargis, NASA POWER v8.2, GSA 2.0, MASEN NOOR mesures terrain.

---

##  Sources & Références

| Domaine | Source |
|---------|--------|
| Ressources solaires | NASA POWER v8.2, Solargis, GSA 2.0 |
| Ressources éoliennes | CDER, Global Wind Atlas |
| CAPEX électrolyseurs | IEA Hydrogen 2023, IRENA 2024 |
| Scénarios demande | MASEN, OCP Group, HyDeal Maroc |
| Émissions CO₂ | Règlement UE 2023/1184 (RFNBO), IPCC AR6 |
| Projets de référence | NOOR Ouarzazate, Tarfaya, Dakhla Éole |
| Taux de change | BCE Jan 2024, BAM 2024, IMF WEO 2024 |
| Optimisation | PyPSA v0.26, NSGA-II (pymoo), SALib |

---

##  Contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. **Forker** le dépôt
2. Créer une branche : `git checkout -b feature/ma-contribution`
3. Committer vos changements : `git commit -m "feat: description claire"`
4. Pousser : `git push origin feature/ma-contribution`
5. Ouvrir une **Pull Request**

### Conventions de commit

```
feat:     nouvelle fonctionnalité
fix:      correction de bug
docs:     documentation uniquement
refactor: refactorisation sans changement fonctionnel
data:     ajout ou mise à jour de données
```

### Domaines prioritaires pour contribution

- [ ] Intégration données NASA POWER API en temps réel
- [ ] Machine Learning pour prédiction LCOH (Random Forest / XGBoost)
- [ ] Extension à d'autres pays africains (Mauritanie, Sénégal)
- [ ] Interface multilingue (arabe, français, anglais)
- [ ] Export PDF des rapports d'analyse

---

##  Licence

Ce projet est distribué sous licence **MIT**.

```
MIT License — Copyright (c) 2026 — H2 Morocco Project
```

Voir le fichier `LICENSE` pour le texte complet.

---

##  Auteurs

Développé dans le cadre d'un Projet de Fin d'Études (PFE) — Par Maroua Larhni et Nisrine Moujane sous encadrement de M. Meryeme AZAROULE

---

##  Citation

Si vous utilisez ce travail dans vos recherches, merci de citer :

```bibtex
@misc{h2morocco2026,
  title  = {H2 Morocco — Plateforme d'Aide à la Décision pour l'Hydrogène Vert},
  author = {[Auteur(s)]},
  year   = {2026},
  institut = {IRESEN},
  note   = {Projet de Fin d'Études}
}
```
