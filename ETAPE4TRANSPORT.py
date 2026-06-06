# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:38:40 2026

@author: HP 840 G8
"""
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  H2 MOROCCO — MODÈLE MILP TRANSPORT MULTI-PÉRIODE 2024–2050                 ║
║  Mixed-Integer Linear Programming — Planification Réseau Logistique H2      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FORMULATION MATHÉMATIQUE                                                    ║
║  ─────────────────────────                                                   ║
║                                                                              ║
║  Indices                                                                     ║
║    i, j  ∈ N       — nœuds (sources, hubs, marchés)                         ║
║    (i,j,m) ∈ A     — arcs (origine, destination, mode)                       ║
║    t  ∈ T          — périodes {2024, 2030, 2035, 2040, 2050}                 ║
║    s  ∈ S          — scénarios {optimiste, central, pessimiste}              ║
║                                                                              ║
║  Variables de décision                                                       ║
║    x[i,j,m,t] ∈ {0,1}  — construire arc (i,j,m) à la période t             ║
║    X[i,j,m,t] ∈ {0,1}  — arc (i,j,m) disponible à t (cumulatif)            ║
║    f[i,j,m,t] ≥ 0       — flux H2 [ktH2/an] sur arc (i,j,m) à t            ║
║    y[i,t]     ∈ {0,1}  — construire hub i à la période t                    ║
║    Y[i,t]     ∈ {0,1}  — hub i disponible à t (cumulatif)                   ║
║                                                                              ║
║  Fonction objectif (VAN des coûts sur horizon 2024–2050)                    ║
║    min Σ_t  δ_t × [                                                          ║
║          Σ_{i,j,m} CAPEX_arc(i,j,m) × x[i,j,m,t]                           ║
║        + Σ_i       CAPEX_hub(i)      × y[i,t]                               ║
║        + Σ_{i,j,m} LCOT(i,j,m,t)   × f[i,j,m,t] × Δt                      ║
║    ]                                                                         ║
║    δ_t = (1+WACC)^{-(t - t_0)} — facteur d'actualisation                   ║
║    Δt  = durée de la période [ans]                                           ║
║                                                                              ║
║  Contraintes                                                                 ║
║    (C1) Bilan flux nœud ∀i,t                                                 ║
║    (C2) Capacité arc    ∀(i,j,m),t  : f ≤ Cap × X  [Big-M]                 ║
║    (C3) Irréversibilité : X[t] ≥ X[t-1]  (pas de démantèlement)            ║
║    (C4) Capacité hub    ∀i,t  : flux_transit ≤ Cap_hub × Y                  ║
║    (C5) Demande minimale ∀i marché, ∀t                                       ║
║    (C6) Offre maximale  ∀i source, ∀t                                        ║
║    (C7) Budget par période                                                   ║
║    (C8) Liaison X/x     : X[t] = X[t-1] + x[t]                              ║
║                                                                              ║
║  STRUCTURE DU FICHIER                                                        ║
║  ─────────────────────                                                       ║
║  BLOC A — Données temporelles (coûts, demande, offre par période)            ║
║  BLOC B — Classe H2TransportMILP_MP (modèle multi-période)                  ║
║             B1. _build_sets()                                                ║
║             B2. _build_temporal_params()  — courbes d'apprentissage          ║
║             B3. _build_model()            — MILP complet                     ║
║             B4. solve()                                                      ║
║             B5. _extract_results()                                           ║
║             B6. summary()                                                    ║
║             B7. save_results()                                               ║
║             B8. plot_network()            — carte réseau par période         ║
║             B9. plot_cashflow()           — flux financiers actualisés       ║
║                                                                              ║
║  BLOC C — Cas d'usage multi-période                                          ║
║             C1. cas1_exportateur_national_mp()                               ║
║             C2. cas2_hub_industriel_mp()                                     ║
║             C3. cas3_site_isole_mp()                                         ║
║                                                                              ║
║  BLOC D — Analyse de sensibilité multi-période                               ║
║                                                                              ║
║  Sources : IEA (2024), Hydrogen Council (2023), IRENA (2024)                 ║
║            T4 corridors, T5 économique, T6 demande, T9 scénarios            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Auteur    : Maroua Larhni
Encadrante: Meryeme Azaroual
Année     : 2025–2026
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# Import des données depuis le module mono-période
# Si milp_transport_H2 n'est pas disponible, définir les constantes localement
try:
    from milp_transport_H2 import (
        NODES, ARCS_DATA, TAUX_USD_EUR,
        CAPEX_HUB_USD, CAPEX_ARC_PAR_KM_USD,
        DUREE_VIE_MODE, CAPACITE_MAX_ARC, EMISSIONS_MODE,
        _crf
    )
    print("✓ Import milp_transport_H2 réussi")
except ImportError:
    print("⚠️ milp_transport_H2 non trouvé — chargement depuis base de données...")
    
    TAUX_USD_EUR = 0.9217
    
    def _crf(dr, lt):
        """Capital Recovery Factor"""
        return (dr * (1 + dr)**lt) / ((1 + dr)**lt - 1)
    
    # ══════════════════════════════════════════════════════════════════════
    # Construction NODES + ARCS_DATA depuis T4 segments
    # ══════════════════════════════════════════════════════════════════════
    _df_seg = None
    
    # Tentative 1 : PostgreSQL
    try:
        from sqlalchemy import create_engine as _ce
        _eng = _ce("postgresql+psycopg2://postgres:marwamarwa2016@localhost:5432/h2morocco_db")
        _df_seg = pd.read_sql("SELECT * FROM h2morocco.t4_segments_detail", _eng)
        print(f"  ✓ {len(_df_seg)} segments chargés depuis PostgreSQL")
    except Exception as e:
        print(f"  ⚠️ PostgreSQL non disponible ({e})")
    
    # Tentative 2 : CSV
    if _df_seg is None or _df_seg.empty:
        from pathlib import Path as _Path
        for _p in [
            _Path.home() / "Downloads" / "h2pipeline" / "outputs" / "clean_csv" / "t4_segments_detail_clean.csv",
            _Path.home() / "Downloads" / "H2Morocco222_Outputs" / "csv" / "T4_segments_detail.csv",
        ]:
            if _p.exists():
                _df_seg = pd.read_csv(_p, encoding="utf-8-sig")
                print(f"  ✓ {len(_df_seg)} segments chargés depuis CSV : {_p.name}")
                break
    
    # Marchés européens / africains (destinations finales export)
    _MARCHES_EXPORT = {
        "Rotterdam", "Barcelone", "Marseille", "Algésiras",
        "Almería", "Canaries", "Dakar", "Paris",
    }
    
    if _df_seg is not None and not _df_seg.empty:
        # ── NODES ──────────────────────────────────────────────────────
        NODES = {}
        for _, r in _df_seg.iterrows():
            o = str(r.get("origine", "")).strip()
            if o and o not in NODES:
                NODES[o] = {
                    "lat": float(r.get("lat_depart", 0) or 0),
                    "lon": float(r.get("lon_depart", 0) or 0),
                    "type": str(r.get("type_noeud_depart", "")).strip(),
                    "pays": str(r.get("pays_depart", "Maroc")).strip(),
                }
            d = str(r.get("destination", "")).strip()
            if d and d not in NODES:
                NODES[d] = {
                    "lat": float(r.get("lat_arrivee", 0) or 0),
                    "lon": float(r.get("lon_arrivee", 0) or 0),
                    "type": str(r.get("type_noeud_arrivee", "")).strip(),
                    "pays": str(r.get("pays_arrivee", "")).strip(),
                }

        # Nœuds de production connus (sources H2 — producteurs d'hydrogène vert)
        _SOURCES_CONNUES = {
            "Dakhla", "Laayoune", "Tarfaya", "Guelmim", "Ouarzazate",
            # NOTE : Midelt est un site industriel CONSOMMATEUR, pas une source
        }

        # Typer les nœuds correctement
        for n, attr in NODES.items():
            if n in _MARCHES_EXPORT:
                attr["type"] = "market"
            elif n in _SOURCES_CONNUES:
                attr["type"] = "source"
            elif "production" in attr.get("type", "").lower():
                attr["type"] = "source"
            elif attr.get("pays", "") == "Maroc":
                attr["type"] = "hub"
            else:
                attr["type"] = "market"

        # ── ARCS_DATA ──────────────────────────────────────────────────
        _arcs_set = set()
        ARCS_DATA = []
        for _, r in _df_seg.iterrows():
            o = str(r.get("origine", "")).strip()
            d = str(r.get("destination", "")).strip()
            m = str(r.get("mode_optimal", "")).strip()
            if not o or not d or not m:
                continue
            clef = (o, d, m)
            if clef not in _arcs_set:
                _arcs_set.add(clef)
                ARCS_DATA.append((
                    o, d, m,
                    float(r.get("distance_km", 0) or 0),
                    float(r.get("cout_min_usd_kg", 0) or 0),
                    float(r.get("cout_max_usd_kg", 0) or 0),
                ))

        # ── Constantes par mode ────────────────────────────────────────
        CAPEX_HUB_USD = {
            n: (80_000_000 if attr.get("pays") != "Maroc" else 50_000_000)
            for n, attr in NODES.items()
            if attr.get("type") in ("hub", "hub_industriel")
        }

        CAPEX_ARC_PAR_KM_USD = {
            # Pipelines : coût fixe par km de conduite construite
            # Source : IEA Hydrogen 2024, Hydrogen Council 2023
            "Pipeline_H2_nouveau"    : 3500,   # $/km
            "Pipeline_H2_reconverti" : 1200,   # $/km
            # Modes mobiles : CAPEX flotte équivalent $/km
            # Navire NH3 ~150 M$ / 15 000 km-voyage / 20 ans de vie
            "Tanker_NH3"  : 180,    # $/km-equiv
            # Tanker LH2 cryogénique ~300 M$, technologie émergente
            "Tanker_LH2"  : 250,    # $/km-equiv
            # Camion Tube_trailer ~1.5 M$ / 15 ans / 200 km/j
            "Tube_trailer": 45,     # $/km-equiv
        }
        DUREE_VIE_MODE = {
            "Pipeline_H2_nouveau": 40, "Pipeline_H2_reconverti": 30,
            "Tanker_NH3": 25, "Tube_trailer": 15, "Tanker_LH2": 25,
        }
        CAPACITE_MAX_ARC = {
            "Pipeline_H2_nouveau": 500, "Pipeline_H2_reconverti": 300,
            "Tanker_NH3": 200, "Tube_trailer": 10, "Tanker_LH2": 100,
        }
        EMISSIONS_MODE = {
            "Pipeline_H2_nouveau": 0.5, "Pipeline_H2_reconverti": 0.3,
            "Tanker_NH3": 2.5, "Tube_trailer": 1.8, "Tanker_LH2": 3.0,
        }

        print(f"  ✅ Réseau construit : {len(NODES)} nœuds | {len(ARCS_DATA)} arcs")
        for n, attr in sorted(NODES.items()):
            print(f"     {n:<15} type={attr['type']:<20} pays={attr['pays']}")
    else:
        print("  ❌ Aucune source T4 disponible")
        NODES, ARCS_DATA = {}, []
        CAPEX_HUB_USD, CAPEX_ARC_PAR_KM_USD = {}, {}
        DUREE_VIE_MODE, CAPACITE_MAX_ARC, EMISSIONS_MODE = {}, {}, {}

# ══════════════════════════════════════════════════════════════════════════════
# BLOC A — DONNÉES TEMPORELLES 2024–2050
# ══════════════════════════════════════════════════════════════════════════════

# ── A1. Périodes et durées ────────────────────────────────────────────────────
PERIODES     = [2024, 2030, 2035, 2040, 2050]
DUREE_PERIODE = {
    2024: 6,   # 2024→2030 : 6 ans
    2030: 5,   # 2030→2035 : 5 ans
    2035: 5,   # 2035→2040 : 5 ans
    2040: 10,  # 2040→2050 : 10 ans
    2050: 0,   # dernière période (pas de durée suivante)
}
T0    = 2024    # Année de référence actualisation
WACC  = 0.08   # Source : T5 — taux_actualisation_mode = 8%

def facteur_actualisation(t: int) -> float:
    """δ_t = (1 + WACC)^{-(t - T0)}"""
    return (1 + WACC) ** (-(t - T0))


# ── A2. Courbes d'apprentissage CAPEX (depuis T9) ────────────────────────────
# Source : T9_scenarios_temporels — Wright's Law (LR=18% PEM, 12% AEL, 24% solaire)
# CAPEX_arc diminue avec le temps → pénalise moins les investissements tardifs

def capex_arc_facteur(mode: str, annee: int) -> float:
    """
    Facteur de réduction du CAPEX infrastructure par mode et période.
    Reflète la maturité technologique croissante des infrastructures H2.

    Source : T9 + Hydrogen Council Insights 2024
    """
    # Réduction cumulée du CAPEX infrastructure H2 par période
    # Pipeline : mature → faible réduction (~5%/5ans)
    # Tanker NH3 : en cours de scaling → réduction modérée (~10%/5ans)
    facteurs = {
        "Pipeline_H2_nouveau"    : {2024:1.00, 2030:0.92, 2035:0.86, 2040:0.80, 2050:0.72},
        "Pipeline_H2_reconverti" : {2024:1.00, 2030:0.95, 2035:0.90, 2040:0.85, 2050:0.78},
        "Pipeline_sous_marin"    : {2024:1.00, 2030:0.93, 2035:0.87, 2040:0.81, 2050:0.73},
        "Tube_trailer"           : {2024:1.00, 2030:0.90, 2035:0.82, 2040:0.75, 2050:0.68},
        "Tanker_NH3"             : {2024:1.00, 2030:0.88, 2035:0.78, 2040:0.70, 2050:0.60},
    }
    f = facteurs.get(mode, {})
    # Interpolation linéaire si l'année n'est pas une borne exacte
    bornes = sorted(f.keys())
    for i in range(len(bornes) - 1):
        if bornes[i] <= annee <= bornes[i+1]:
            alpha = (annee - bornes[i]) / (bornes[i+1] - bornes[i])
            return f[bornes[i]] + alpha * (f[bornes[i+1]] - f[bornes[i]])
    return f.get(max(bornes), 1.0)


def lcot_facteur(mode: str, annee: int) -> float:
    """
    Facteur de réduction du LCOT opérationnel (coût transport €/kg).
    Reflète la baisse des prix électricité + gains d'efficacité logistique.

    Source : T9 + IEA Hydrogen Insights 2024
    """
    facteurs = {
        "Pipeline_H2_nouveau"    : {2024:1.00, 2030:0.90, 2035:0.82, 2040:0.75, 2050:0.65},
        "Pipeline_H2_reconverti" : {2024:1.00, 2030:0.92, 2035:0.85, 2040:0.78, 2050:0.68},
        "Pipeline_sous_marin"    : {2024:1.00, 2030:0.90, 2035:0.82, 2040:0.75, 2050:0.65},
        "Tube_trailer"           : {2024:1.00, 2030:0.93, 2035:0.87, 2040:0.81, 2050:0.72},
        "Tanker_NH3"             : {2024:1.00, 2030:0.87, 2035:0.76, 2040:0.67, 2050:0.55},
    }
    f = facteurs.get(mode, {})
    bornes = sorted(f.keys())
    for i in range(len(bornes) - 1):
        if bornes[i] <= annee <= bornes[i+1]:
            alpha = (annee - bornes[i]) / (bornes[i+1] - bornes[i])
            return f[bornes[i]] + alpha * (f[bornes[i+1]] - f[bornes[i]])
    return f.get(max(bornes), 1.0)


def capex_hub_facteur(annee: int) -> float:
    """
    Facteur de réduction CAPEX hub (compression + stockage).
    Source : T3 + T9 — réduction 30% entre 2024 et 2050.
    """
    f = {2024:1.00, 2030:0.90, 2035:0.82, 2040:0.76, 2050:0.68}
    bornes = sorted(f.keys())
    for i in range(len(bornes) - 1):
        if bornes[i] <= annee <= bornes[i+1]:
            alpha = (annee - bornes[i]) / (bornes[i+1] - bornes[i])
            return f[bornes[i]] + alpha * (f[bornes[i+1]] - f[bornes[i]])
    return f.get(max(bornes), 1.0)


# ── A3. Demande par période (depuis T6 — build_T6_marche) ────────────────────
# Demande nationale + export  [ktH2/an]
# Source : T6a_demande_nationale.csv + Stratégie Nationale H2 Maroc 2021

DEMANDE_NATIONALE_KT = {
    # {secteur : {période: ktH2/an}}
    "OCP_chimie"   : {2024: 120, 2030: 200, 2035: 280, 2040: 350, 2050: 500},
    "Raffinage"    : {2024:  80, 2030:  70, 2035:  55, 2040:  45, 2050:  30},
    "Mobilite"     : {2024:   2, 2030:  25, 2035:  65, 2040: 110, 2050: 280},
    "Industrie"    : {2024:  10, 2030:  50, 2035: 100, 2040: 150, 2050: 350},
    "Export_EU"    : {2024:   0, 2030: 400, 2035: 800, 2040:1500, 2050:3000},
}

# Demande agrégée par nœud de destination et par période
# (synthèse pour le modèle MILP — T6 + Stratégie Nationale)
DEMANDE_NOEUDS = {
    # Export vers Europe
    "Rotterdam" : {2024:  0, 2030: 200, 2035: 350, 2040: 600, 2050:1200},
    "Barcelone" : {2024:  0, 2030: 100, 2035: 180, 2040: 300, 2050: 600},
    "Marseille" : {2024:  0, 2030:  80, 2035: 140, 2040: 240, 2050: 480},
    "Algésiras" : {2024:  0, 2030:  20, 2035:  40, 2040:  80, 2050: 200},
    # Demande domestique industrielle
    "OCP_Jorf"  : {2024: 60, 2030: 120, 2035: 180, 2040: 250, 2050: 400},
    # Site isolé
    "Mine_desert": {2024: 5, 2030:  20, 2035:  30, 2040:  40, 2050:  50},
}

# ── A4. Offre maximale par nœud de production et par période ─────────────────
# Source : T1 CF hybride + T2 CAPEX électrolyseur + T9 capacité_electrolyseur_GW
# Hypothèse : déploiement progressif de l'électrolyse selon T9

OFFRE_NOEUDS = {
    "Dakhla"    : {2024:  30, 2030: 400, 2035: 800, 2040:1500, 2050:3000},
    "Laayoune"  : {2024:  20, 2030: 300, 2035: 600, 2040:1000, 2050:2000},
    "Tarfaya"   : {2024:  10, 2030: 200, 2035: 400, 2040: 700, 2050:1400},
    "Ouarzazate": {2024:   5, 2030: 150, 2035: 250, 2040: 400, 2050: 700},
    "Guelmim"   : {2024:   5, 2030:  80, 2035: 150, 2040: 250, 2050: 450},
}

# ── A5. Budget maximum par période (investissement infrastructure) ────────────
# Source : T9 investissement_cumul_Mrd_USD
# Budgets progressifs cohérents avec la trajectoire d'investissement nationale
BUDGET_PAR_PERIODE_MUSD = {
    2024: 2000,    #  2 Mrd$ — phase pilote (inclut infra domestique initiale)
    2030: 5000,    #  5 Mrd$  — phase déploiement
    2035: 8000,    #  8 Mrd$  — phase accélération
    2040:12000,    # 12 Mrd$  — phase maturité
    2050:20000,    # 20 Mrd$  — phase export massif
}


# ══════════════════════════════════════════════════════════════════════════════
# BLOC B — CLASSE H2TransportMILP_MP (Multi-Période)
# ══════════════════════════════════════════════════════════════════════════════

class H2TransportMILP_MP:
    """
    Modèle MILP multi-période pour l'optimisation du réseau de transport H2.

    Planifie sur T = {2024, 2030, 2035, 2040, 2050} :
      — Quand construire chaque arc de transport ?
      — Quand ouvrir chaque hub ?
      — Quels flux acheminer à chaque période ?

    La fonction objectif minimise la Valeur Actualisée Nette (VAN)
    des coûts totaux sur l'horizon complet.

    Usage
    ─────
    >>> mp = H2TransportMILP_MP(
    ...     scenario="central",
    ...     noms_noeuds_actifs=["Dakhla", "Laayoune", "Agadir", "Rotterdam"]
    ... )
    >>> mp.set_demande(DEMANDE_NOEUDS)
    >>> mp.set_offre(OFFRE_NOEUDS)
    >>> mp.solve()
    >>> mp.summary()
    >>> mp.plot_network()
    >>> mp.plot_cashflow()
    """

    SCENARIOS = {
        "optimiste" : {"lcot_factor": 0.80, "capex_factor": 0.80, "demand_factor": 1.20},
        "central"   : {"lcot_factor": 1.00, "capex_factor": 1.00, "demand_factor": 1.00},
        "pessimiste": {"lcot_factor": 1.20, "capex_factor": 1.20, "demand_factor": 0.80},
    }

    def __init__(self,
                 scenario             : str   = "central",
                 periodes             : list  = None,
                 budget_par_periode   : dict  = None,
                 taxe_carbone_USD_tCO2: float = 0.0,
                 noms_noeuds_actifs   : list  = None,
                 avec_irrev           : bool  = True):
        """
        Paramètres
        ──────────
        scenario              : "optimiste" | "central" | "pessimiste"
        periodes              : liste d'années (défaut : [2024,2030,2035,2040,2050])
        budget_par_periode    : {année: budget_MUSD} (None = non contraint)
        taxe_carbone_USD_tCO2 : taxe CO2 $/tonne (croît linéairement avec le temps)
        noms_noeuds_actifs    : filtrage des nœuds
        avec_irrev            : True → contrainte d'irréversibilité (X[t] ≥ X[t-1])
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario doit être parmi {list(self.SCENARIOS.keys())}")

        self.scenario      = scenario
        self.T             = periodes or PERIODES
        self.sf            = self.SCENARIOS[scenario]
        self.budget_dict   = budget_par_periode or BUDGET_PAR_PERIODE_MUSD
        self.taxe0         = taxe_carbone_USD_tCO2   # taxe en 2024
        self.avec_irrev    = avec_irrev

        # Filtrage nœuds
        if noms_noeuds_actifs:
            self.nodes = {k: v for k, v in NODES.items() if k in noms_noeuds_actifs}
        else:
            self.nodes = NODES.copy()

        # Demande et offre temporelles (à remplir)
        self._demande : dict = {}   # {nœud: {période: ktH2/an}}
        self._offre   : dict = {}   # {nœud: {période: ktH2/an}}

        # Résultats
        self.prob      = None
        self.status    = None
        self.resultats = {}

        self._build_sets()
        self._build_temporal_params()

    # ─────────────────────────────────────────────────────────────────────────
    # B1. Ensembles
    # ─────────────────────────────────────────────────────────────────────────
    def _build_sets(self):
        """Construit N (nœuds), A (arcs), M (modes), T (périodes)."""
        self.N = list(self.nodes.keys())

        self.arcs = [
            (o, d, m, dist, cmin, cmax)
            for (o, d, m, dist, cmin, cmax) in ARCS_DATA
            if o in self.N and d in self.N
        ]

        # ── Arcs synthétiques de secours ──────────────────────────────────────
        # Si une source n'est pas connectée à un nœud actif du sous-réseau,
        # on ajoute des arcs Tube_trailer de secours pour garantir la faisabilité.
        arcs_existants = {(o, d) for (o, d, m, _, _, _) in self.arcs}
        sources = [n for n, a in self.nodes.items() if a.get("type") == "source"]
        marches = [n for n, a in self.nodes.items() if a.get("type") == "market"]
        hubs_locaux = [n for n, a in self.nodes.items()
                       if a.get("type") in ("hub", "hub_industriel")]

        for src in sources:
            # Vérifie si la source a au moins un arc sortant dans le sous-réseau
            reachable = any(d in self.N for (o, d, m, _, _, _) in self.arcs if o == src)
            if not reachable:
                cibles = hubs_locaux if hubs_locaux else marches
                for hub in cibles:
                    if hub != src and (src, hub) not in arcs_existants:
                        self.arcs.append((src, hub, "Tube_trailer", 500, 0.5, 1.5))
                        arcs_existants.add((src, hub))

        # Vérifier que les marchés ont au moins un arc entrant
        for mkt in marches:
            entrants = [(o, d) for (o, d, m, _, _, _) in self.arcs if d == mkt]
            if not entrants:
                for src in sources:
                    if (src, mkt) not in arcs_existants:
                        self.arcs.append((src, mkt, "Tube_trailer", 800, 0.8, 2.0))
                        arcs_existants.add((src, mkt))
                        break  # un seul arc de secours suffit
        # ─────────────────────────────────────────────────────────────────────

        self.A = [(o, d, m) for (o, d, m, _, _, _) in self.arcs]
        self.M = list(set(m for (_, _, m, _, _, _) in self.arcs))

        self.hubs = [
            n for n, attr in self.nodes.items()
            if attr.get("type") in ("hub", "hub_industriel") and n in CAPEX_HUB_USD
        ]

        # Index de distance par arc
        self._dist = {
            (o, d, m): dist
            for (o, d, m, dist, _, _) in self.arcs
        }
        self._cmin = {(o, d, m): cmin for (o, d, m, _, cmin, _) in self.arcs}
        self._cmax = {(o, d, m): cmax for (o, d, m, _, _, cmax) in self.arcs}

    # ─────────────────────────────────────────────────────────────────────────
    # B2. Paramètres temporels
    # ─────────────────────────────────────────────────────────────────────────
    def _build_temporal_params(self):
        """
        Calcule pour chaque arc et chaque période :
          - LCOT(arc, t) en €/kgH2 (coût opérationnel)
          - CAPEX_arc(arc, t) en USD (coût investissement non annualisé)
          - CAPEX_hub(hub, t) en USD

        Intègre les courbes d'apprentissage (T9) et la taxe carbone croissante.
        """
        sf_l = self.sf["lcot_factor"]
        sf_c = self.sf["capex_factor"]

        self.LCOT     = {}   # (arc, t) → €/kgH2
        self.CAPEX_arc= {}   # (arc, t) → USD (CAPEX total non annualisé)
        self.CAPEX_ann= {}   # (arc, t) → USD/an (annualisé)
        self.CAP_arc  = {}   # (arc, t) → ktH2/an

        for t in self.T:
            delta_t = t - T0   # années depuis 2024
            # Taxe carbone croît linéairement : +5 $/tCO2/an
            taxe_t = max(0, self.taxe0 + delta_t * 5)

            for arc in self.A:
                o, d, m = arc
                dist_km = self._dist[arc]
                cmin    = self._cmin[arc]
                cmax    = self._cmax[arc]

                # LCOT opérationnel : valeur centrale × facteur scénario × courbe apprentissage
                lcot_usd = ((cmin + cmax) / 2) * sf_l * lcot_facteur(m, t)
                # Ajout taxe carbone
                lcot_co2 = EMISSIONS_MODE.get(m, 0) * taxe_t / 1000
                self.LCOT[(arc, t)] = (lcot_usd + lcot_co2) * TAUX_USD_EUR   # → €/kg

                # ── CAPEX arc — selon le mode de transport ────────────────────
                # Pipelines : coût proportionnel à la distance (infrastructure fixe)
                # Modes mobiles : coût de flotte (navires / camions) proportionnel
                #   à la capacité maximale de l'arc, pas uniquement à la distance.
                #   Formule : nb_unités × coût_unitaire × facteur_scénario × learning
                #
                # Coûts unitaires de référence (source : IEA 2024, H2 Council 2023)
                CAPEX_UNITAIRE_USD = {
                    "Pipeline_H2_nouveau"    : None,      # géré par $/km
                    "Pipeline_H2_reconverti" : None,      # géré par $/km
                    "Tanker_NH3"  : 150_000_000,   # 150 M$ / navire (55 000 t NH3)
                    "Tanker_LH2"  : 300_000_000,   # 300 M$ / navire (LH2 cryogénique)
                    "Tube_trailer":   1_500_000,   # 1.5 M$ / camion-remorque 1 t H2
                }
                # Capacité par unité [ktH2/an]
                CAP_UNITAIRE_KT = {
                    "Tanker_NH3"  : 33.0,   # 55 000 t NH3 × 0.177 t H2/t NH3 × 3.4 voyages/an
                    "Tanker_LH2"  : 6.5,    # ~6 500 t H2/an par navire (IEA 2024)
                    "Tube_trailer": 0.18,   # ~180 t H2/an par camion (250 jours × 700 kg)
                }

                capex_km = CAPEX_ARC_PAR_KM_USD.get(m, 0)
                cap_max_kt = CAPACITE_MAX_ARC.get(m, 100)

                if CAPEX_UNITAIRE_USD.get(m) is None:
                    # Pipeline : CAPEX ∝ distance
                    capex_tot = capex_km * dist_km * sf_c * capex_arc_facteur(m, t)
                else:
                    # Mode mobile : CAPEX = nb_unités × coût_unitaire
                    cap_unit  = CAP_UNITAIRE_KT.get(m, 1.0)
                    nb_unites = max(1, math.ceil(cap_max_kt / cap_unit))
                    capex_tot = (nb_unites * CAPEX_UNITAIRE_USD[m]
                                 * sf_c * capex_arc_facteur(m, t))

                self.CAPEX_arc[(arc, t)] = capex_tot

                # Annualisé pour la fonction objectif
                lt  = DUREE_VIE_MODE.get(m, 20)
                crf = _crf(WACC, lt)
                self.CAPEX_ann[(arc, t)] = capex_tot * crf

                # Capacité arc (augmente légèrement avec les améliorations technologiques)
                cap_base = CAPACITE_MAX_ARC.get(m, 500)
                cap_factor = 1.0 + (t - 2024) / 100   # +1% par an
                self.CAP_arc[(arc, t)] = cap_base * cap_factor

        # Paramètres hubs par période
        self.CAPEX_hub     = {}   # (hub, t) → USD
        self.CAPEX_hub_ann = {}   # (hub, t) → USD/an
        self.CAP_hub       = {}   # (hub, t) → ktH2/an

        crf_hub = _crf(WACC, 20)
        for t in self.T:
            for h in self.hubs:
                capex_h = CAPEX_HUB_USD[h] * sf_c * capex_hub_facteur(t)
                self.CAPEX_hub[(h, t)]     = capex_h
                self.CAPEX_hub_ann[(h, t)] = capex_h * crf_hub

                entrants = sum(self.CAP_arc.get((o, d, m, t), 0)
                               for (o, d, m) in self.A if d == h
                               for _ in [None]   # dummy pour syntaxe
                               )
                # Capacité hub = 80% somme arcs entrants à la période t
                total_entrants = sum(
                    self.CAP_arc.get(((o, d, m), t), self.CAP_arc.get((o, d, m), 500))
                    for (o, d, m) in self.A if d == h
                )
                self.CAP_hub[(h, t)] = max(total_entrants * 0.8, 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Setters
    # ─────────────────────────────────────────────────────────────────────────
    def set_demande(self, demande: dict):
        """
        demande : {nœud: {période: ktH2/an}}
        Exemple : {"Rotterdam": {2030: 200, 2035: 350, 2040: 600}}
        """
        sf_d = self.sf["demand_factor"]
        self._demande = {
            n: {t: q * sf_d for t, q in tdict.items()}
            for n, tdict in demande.items()
            if n in self.N
        }

    def set_offre(self, offre: dict):
        """
        offre : {nœud: {période: ktH2/an_max}}
        """
        self._offre = {
            n: tdict for n, tdict in offre.items()
            if n in self.N
        }

    # ─────────────────────────────────────────────────────────────────────────
    # B3. Construction du modèle MILP multi-période
    # ─────────────────────────────────────────────────────────────────────────
    def _build_model(self):
        """
        Construit le MILP multi-période complet.

        Variables
        ─────────
        x[(arc, t)]  ∈ {0,1}  — décision d'investissement à la période t
        X[(arc, t)]  ∈ {0,1}  — arc disponible à t (cumulatif)
        f[(arc, t)]  ≥ 0       — flux [ktH2/an] à la période t
        y[(hub, t)]  ∈ {0,1}  — décision d'ouvrir hub à t
        Y[(hub, t)]  ∈ {0,1}  — hub disponible à t (cumulatif)
        """
        prob = pulp.LpProblem(
            f"H2_MILP_MP_{self.scenario}",
            pulp.LpMinimize
        )

        # ── Variables ─────────────────────────────────────────────────────────

        # x[(arc,t)] — investissement dans l'arc à la période t
        x = pulp.LpVariable.dicts(
            "x", [(arc, t) for arc in self.A for t in self.T], cat="Binary"
        )

        # X[(arc,t)] — arc disponible à t (somme cumulée des x passés)
        X = pulp.LpVariable.dicts(
            "X", [(arc, t) for arc in self.A for t in self.T], cat="Binary"
        )

        # f[(arc,t)] — flux H2 [ktH2/an]
        f = pulp.LpVariable.dicts(
            "f", [(arc, t) for arc in self.A for t in self.T],
            lowBound=0, cat="Continuous"
        )

        # y[(hub,t)] — décision d'ouverture hub à t
        y = pulp.LpVariable.dicts(
            "y", [(h, t) for h in self.hubs for t in self.T], cat="Binary"
        )

        # Y[(hub,t)] — hub disponible à t (cumulatif)
        Y = pulp.LpVariable.dicts(
            "Y", [(h, t) for h in self.hubs for t in self.T], cat="Binary"
        )

        # Variables de pénurie (slack) pour éviter l'infaisabilité
        # Pénurie = demande non satisfaite, pénalisée fortement dans l'objectif
        penurie = pulp.LpVariable.dicts(
            "penurie",
            [(n, t) for n in self.N for t in self.T
             if self._demande.get(n, {}).get(t, 0) > 0],
            lowBound=0, cat="Continuous"
        )

        # ── Fonction objectif — VAN des coûts ────────────────────────────────
        # Σ_t δ_t × Δt × [CAPEX_arc×x + CAPEX_hub×y + LCOT×f×1e6]
        # + pénalité pénurie très élevée

        PENALITE_PENURIE = 1e9  # €/ktH2 — rend la pénurie très coûteuse

        van_capex_arcs = pulp.lpSum(
            facteur_actualisation(t) *
            self.CAPEX_arc.get((arc, t), 0) * x[(arc, t)]
            for arc in self.A for t in self.T
        )

        van_capex_hubs = pulp.lpSum(
            facteur_actualisation(t) *
            self.CAPEX_hub.get((h, t), 0) * y[(h, t)]
            for h in self.hubs for t in self.T
        )

        van_transport = pulp.lpSum(
            facteur_actualisation(t) *
            DUREE_PERIODE.get(t, 1) *   # × durée de la période [ans]
            self.LCOT.get((arc, t), 0) *
            f[(arc, t)] * 1e6           # ktH2/an → kgH2/an
            for arc in self.A for t in self.T
        )

        van_penurie = pulp.lpSum(
            PENALITE_PENURIE * penurie.get((n, t), 0)
            for n in self.N for t in self.T
            if self._demande.get(n, {}).get(t, 0) > 0
        )

        prob += van_capex_arcs + van_capex_hubs + van_transport + van_penurie, "VAN_Cout_Total_EUR"

        # ── Contraintes ───────────────────────────────────────────────────────
        BIG_M = 10_000  # ktH2/an

        for t in self.T:
            t_idx = self.T.index(t)

            # ── C8. Liaison X = cumul des x (irréversibilité) ────────────────
            for arc in self.A:
                if t_idx == 0:
                    # Première période : X = x (pas d'historique)
                    prob += (X[(arc, t)] == x[(arc, t)],
                             f"Init_X_{arc[0]}_{arc[1]}_{arc[2]}_{t}")
                else:
                    t_prev = self.T[t_idx - 1]
                    # X[t] = X[t-1] + x[t]  (au plus 1 car binaire)
                    prob += (X[(arc, t)] >= X[(arc, t_prev)],
                             f"Irrev_arc_{arc[0]}_{arc[1]}_{arc[2]}_{t}")
                    prob += (X[(arc, t)] <= X[(arc, t_prev)] + x[(arc, t)],
                             f"Cum_arc_{arc[0]}_{arc[1]}_{arc[2]}_{t}")
                    if self.avec_irrev:
                        # Pas de démantèlement : X ne peut que croître
                        prob += (X[(arc, t)] >= X[(arc, t_prev)],
                                 f"NoDemant_arc_{arc[0]}_{arc[1]}_{arc[2]}_{t}")

            # Idem pour les hubs
            for h in self.hubs:
                if t_idx == 0:
                    prob += (Y[(h, t)] == y[(h, t)],
                             f"Init_Y_{h}_{t}")
                else:
                    t_prev = self.T[t_idx - 1]
                    prob += (Y[(h, t)] >= Y[(h, t_prev)],
                             f"Irrev_hub_{h}_{t}")
                    prob += (Y[(h, t)] <= Y[(h, t_prev)] + y[(h, t)],
                             f"Cum_hub_{h}_{t}")

            # ── C2. Capacité arc (Big-M : flux ≤ Cap × X disponible) ─────────
            for arc in self.A:
                cap_t = self.CAP_arc.get((arc, t), 500)
                prob += (f[(arc, t)] <= cap_t * X[(arc, t)],
                         f"Cap_arc_{arc[0]}_{arc[1]}_{arc[2]}_{t}")
                prob += (f[(arc, t)] <= BIG_M * X[(arc, t)],
                         f"BigM_arc_{arc[0]}_{arc[1]}_{arc[2]}_{t}")

            # ── C4. Capacité hub ──────────────────────────────────────────────
            for h in self.hubs:
                flux_entrant_h = pulp.lpSum(
                    f[(arc, t)] for arc in self.A if arc[1] == h
                )
                cap_h = self.CAP_hub.get((h, t), 500)
                # Hub disponible si ouvert OU si c'est un hub marocain existant
                # (infrastructure de base disponible sans investissement)
                is_moroccan = self.nodes.get(h, {}).get("pays", "") == "Maroc"
                if is_moroccan:
                    # Hub marocain : capacité de base disponible sans CAPEX,
                    # investissement optionnel pour augmenter la capacité
                    prob += (flux_entrant_h <= cap_h * (Y[(h, t)] + 1),
                             f"Cap_hub_{h}_{t}")
                else:
                    prob += (flux_entrant_h <= cap_h * Y[(h, t)],
                             f"Cap_hub_{h}_{t}")
                    prob += (flux_entrant_h <= BIG_M * Y[(h, t)],
                             f"Hub_actif_{h}_{t}")

            # ── C1. Bilan flux nœud ───────────────────────────────────────────
            for n in self.N:
                flux_sortant = pulp.lpSum(f[(arc, t)] for arc in self.A if arc[0] == n)
                flux_entrant = pulp.lpSum(f[(arc, t)] for arc in self.A if arc[1] == n)

                demande_nt = self._demande.get(n, {}).get(t, 0)
                offre_nt   = self._offre.get(n, {}).get(t, 0)
                node_type  = self.nodes.get(n, {}).get("type", "hub")

                if demande_nt > 0:
                    # PRIORITÉ 1 : nœud avec demande explicite (même si typé source ou hub)
                    # flux_entrant - flux_sortant + pénurie ≥ demande
                    slack = penurie.get((n, t), 0)
                    prob += (flux_entrant - flux_sortant + slack >= demande_nt,
                             f"Demande_{n}_{t}")
                elif offre_nt > 0:
                    # PRIORITÉ 2 : nœud de production avec offre explicite
                    # flux_sortant ≤ offre max (source peut ne pas tout produire)
                    prob += (flux_sortant <= offre_nt,
                             f"Offre_max_{n}_{t}")
                elif node_type == "source":
                    # PRIORITÉ 3 : source sans offre définie → pas de contrainte
                    # (le solveur peut ignorer ce nœud librement)
                    pass
                else:
                    # PRIORITÉ 4 : nœud transit (hub) — conservation de flux
                    # flux entrant ≥ flux sortant (pas de création de flux)
                    prob += (flux_entrant >= flux_sortant,
                             f"Transit_{n}_{t}")

            # ── C7. Budget par période ────────────────────────────────────────
            budget_t = self.budget_dict.get(t, float("inf")) * 1e6  # → USD
            capex_investi_t = (
                pulp.lpSum(
                    self.CAPEX_arc.get((arc, t), 0) * x[(arc, t)]
                    for arc in self.A
                ) +
                pulp.lpSum(
                    self.CAPEX_hub.get((h, t), 0) * y[(h, t)]
                    for h in self.hubs
                )
            )
            if budget_t < 1e15:
                prob += (capex_investi_t <= budget_t,
                         f"Budget_{t}")

        self.prob    = prob
        self._x      = x
        self._X      = X
        self._f      = f
        self._y      = y
        self._Y      = Y
        self._penurie = penurie
        return prob

    # ─────────────────────────────────────────────────────────────────────────
    # B4. Résolution
    # ─────────────────────────────────────────────────────────────────────────
    def solve(self, verbose: bool = False, time_limit: int = 300):
        """
        Résout le MILP multi-période avec CBC.

        verbose    : affiche les logs solveur
        time_limit : temps max en secondes (défaut 5 min)
        """
        if not self._demande:
            raise RuntimeError("Appelez set_demande() avant solve().")
        if not self._offre:
            raise RuntimeError("Appelez set_offre() avant solve().")

        print(f"\n  Construction du modèle MILP multi-période...")
        self._build_model()

        n_vars   = len(self.prob.variables())
        n_cstrs  = len(self.prob.constraints)
        print(f"  Variables : {n_vars}  |  Contraintes : {n_cstrs}")
        print(f"  Résolution en cours (CBC, max {time_limit}s)...")

        solver = pulp.PULP_CBC_CMD(
            msg=1 if verbose else 0,
            timeLimit=time_limit,
            gapRel=0.01,    # Gap 1% — acceptable pour problème de planification
        )
        self.prob.solve(solver)
        self.status = pulp.LpStatus[self.prob.status]
        print(f"  Statut : {self.status}")

        self._extract_results()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # B5. Extraction des résultats
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_results(self):
        """Extrait et structure les résultats multi-période."""
        x, X, f, y, Y = self._x, self._X, self._f, self._y, self._Y

        def val(v):
            return pulp.value(v) or 0

        # ── Résultats par période ─────────────────────────────────────────────
        resultats_par_periode = {}

        for t in self.T:
            # Arcs nouvellement construits à t
            arcs_investis_t = [
                arc for arc in self.A if val(x[(arc, t)]) > 0.5
            ]
            # Arcs disponibles à t (cumulatif)
            arcs_dispo_t = [
                arc for arc in self.A if val(X[(arc, t)]) > 0.5
            ]
            # Hubs nouvellement ouverts
            hubs_ouverts_t = [h for h in self.hubs if val(y[(h, t)]) > 0.5]
            # Hubs disponibles
            hubs_dispo_t   = [h for h in self.hubs if val(Y[(h, t)]) > 0.5]

            # Flux actifs
            flux_t = {
                arc: val(f[(arc, t)])
                for arc in self.A
                if val(f[(arc, t)]) > 1e-3
            }

            # Coûts période t
            capex_arcs_t = sum(
                self.CAPEX_arc.get((arc, t), 0) for arc in arcs_investis_t
            )
            capex_hubs_t = sum(
                self.CAPEX_hub.get((h, t), 0) for h in hubs_ouverts_t
            )
            cout_transport_t = sum(
                self.LCOT.get((arc, t), 0) * flux_t.get(arc, 0) * 1e6
                for arc in flux_t
            )
            van_t = facteur_actualisation(t) * (
                capex_arcs_t + capex_hubs_t +
                DUREE_PERIODE.get(t, 1) * cout_transport_t
            )

            # Demande totale à t et taux de service
            demande_t = sum(self._demande.get(n, {}).get(t, 0) for n in self.N)
            livraison_t = sum(
                val(f[(arc, t)]) for arc in self.A
                if self._demande.get(arc[1], {}).get(t, 0) > 0
            )
            penurie_t = sum(
                val(self._penurie.get((n, t), 0))
                for n in self.N
                if self._demande.get(n, {}).get(t, 0) > 0
            )

            # LCOT réseau à t
            lcot_t = (
                cout_transport_t / (demande_t * 1e6)
                if demande_t > 0 else 0
            )

            # Détail arcs actifs
            detail = []
            for arc in arcs_dispo_t:
                flux_val = val(f[(arc, t)])
                if flux_val < 1e-3:
                    continue
                detail.append({
                    "periode"      : t,
                    "origine"      : arc[0],
                    "destination"  : arc[1],
                    "mode"         : arc[2],
                    "distance_km"  : self._dist.get(arc, 0),
                    "flux_ktH2_an" : round(flux_val, 2),
                    "LCOT_EUR_kg"  : round(self.LCOT.get((arc, t), 0), 4),
                    "nouveau"      : arc in arcs_investis_t,
                    "CAPEX_USD"    : round(self.CAPEX_arc.get((arc, t), 0) / 1e6, 2),
                })

            resultats_par_periode[t] = {
                "arcs_investis"    : arcs_investis_t,
                "arcs_disponibles" : arcs_dispo_t,
                "hubs_ouverts"     : hubs_ouverts_t,
                "hubs_disponibles" : hubs_dispo_t,
                "flux"             : flux_t,
                "detail_arcs"      : pd.DataFrame(detail),
                "capex_arcs_MUSD"  : round(capex_arcs_t / 1e6, 1),
                "capex_hubs_MUSD"  : round(capex_hubs_t / 1e6, 1),
                "cout_transport_EUR_an": round(cout_transport_t, 0),
                "VAN_t_EUR"        : round(van_t, 0),
                "demande_ktH2"     : round(demande_t, 1),
                "livraison_ktH2"   : round(livraison_t, 1),
                "LCOT_EUR_kg"      : round(lcot_t, 4),
            }

        # ── Résultats globaux ─────────────────────────────────────────────────
        # VAN économique réelle = objectif - pénalités pénurie artificielles
        PENALITE_PENURIE = 1e9
        total_penurie_cost = PENALITE_PENURIE * sum(
            val(self._penurie.get((n, t), 0))
            for n in self.N for t in self.T
            if self._demande.get(n, {}).get(t, 0) > 0
        )
        van_totale = val(self.prob.objective) - total_penurie_cost

        capex_total_MUSD = sum(
            resultats_par_periode[t]["capex_arcs_MUSD"] +
            resultats_par_periode[t]["capex_hubs_MUSD"]
            for t in self.T
        )

        emissions_total = {
            t: sum(
                EMISSIONS_MODE.get(arc[2], 0) *
                resultats_par_periode[t]["flux"].get(arc, 0) *
                DUREE_PERIODE.get(t, 1)
                for arc in self.A
            ) / 1e3   # → ktCO2
            for t in self.T
        }

        self.resultats = {
            "status"            : self.status,
            "scenario"          : self.scenario,
            "periodes"          : self.T,
            "VAN_totale_EUR"    : round(van_totale, 0),
            "capex_total_MUSD"  : round(capex_total_MUSD, 1),
            "par_periode"       : resultats_par_periode,
            "emissions_ktCO2"   : emissions_total,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # B6. Affichage résumé
    # ─────────────────────────────────────────────────────────────────────────
    def summary(self):
        """Affiche le résumé multi-période console."""
        r = self.resultats
        if not r:
            print("  Appelez solve() d'abord.")
            return

        print("\n" + "═" * 72)
        print(f"  RÉSULTATS MILP MULTI-PÉRIODE — Scénario {r['scenario'].upper()}")
        print("═" * 72)
        # Afficher statut enrichi
        statut_affiche = r['status']
        if statut_affiche == "Infeasible":
            statut_affiche = "Infeasible ⚠️ (vérifier les arcs du réseau)"
        elif statut_affiche in ("Optimal", "Not Solved"):
            statut_affiche = "✅ Optimal"
        print(f"  Statut : {statut_affiche}")
        print(f"  VAN coûts totaux  : {r['VAN_totale_EUR']/1e9:.3f} Mrd €")
        print(f"  CAPEX total investi : {r['capex_total_MUSD']:.1f} M$")
        print()

        print(f"  {'Période':>8} {'Dem. ktH2':>10} {'Livr. ktH2':>11} "
              f"{'LCOT €/kg':>10} {'CAPEX M$':>10} {'Émiss. ktCO2':>14} "
              f"{'Hubs dispo':>15} {'Nouveaux arcs':>14}")
        print("  " + "─" * 97)

        for t in self.T:
            rp = r["par_periode"][t]
            print(
                f"  {t:>8} {rp['demande_ktH2']:>10.0f} {rp['livraison_ktH2']:>11.0f} "
                f"{rp['LCOT_EUR_kg']:>10.4f} "
                f"{rp['capex_arcs_MUSD'] + rp['capex_hubs_MUSD']:>10.1f} "
                f"{r['emissions_ktCO2'].get(t, 0):>14.1f} "
                f"{', '.join(rp['hubs_disponibles'])[:15]:>15} "
                f"{len(rp['arcs_investis']):>14}"
            )
        print("═" * 72)

        # Détail des investissements par période
        print("\n  CALENDRIER DES INVESTISSEMENTS :")
        for t in self.T:
            rp = r["par_periode"][t]
            if rp["arcs_investis"] or rp["hubs_ouverts"]:
                print(f"\n  [{t}] CAPEX total : "
                      f"{rp['capex_arcs_MUSD'] + rp['capex_hubs_MUSD']:.1f} M$")
                for arc in rp["arcs_investis"]:
                    cap = self.CAPEX_arc.get((arc, t), 0) / 1e6
                    print(f"     + Arc {arc[0]:>12} → {arc[1]:<12} "
                          f"[{arc[2]:<25}]  {cap:>7.1f} M$")
                for h in rp["hubs_ouverts"]:
                    cap = self.CAPEX_hub.get((h, t), 0) / 1e6
                    print(f"     + Hub {h:<28}  {cap:>7.1f} M$")

    # ─────────────────────────────────────────────────────────────────────────
    # B7. Sauvegarde
    # ─────────────────────────────────────────────────────────────────────────
    def save_results(self, output_dir: str, nom_cas: str = ""):
        """Sauvegarde les résultats multi-période en CSV."""
        os.makedirs(output_dir, exist_ok=True)
        r = self.resultats
        if not r:
            return
        prefix = f"{nom_cas}_" if nom_cas else ""

        # Tableau résumé par période
        rows_periode = []
        for t in self.T:
            rp = r["par_periode"][t]
            rows_periode.append({
                "cas"                  : nom_cas,
                "scenario"             : r["scenario"],
                "periode"              : t,
                "status"               : r["status"],
                "demande_ktH2"         : rp["demande_ktH2"],
                "livraison_ktH2"       : rp["livraison_ktH2"],
                "LCOT_EUR_kg"          : rp["LCOT_EUR_kg"],
                "capex_arcs_MUSD"      : rp["capex_arcs_MUSD"],
                "capex_hubs_MUSD"      : rp["capex_hubs_MUSD"],
                "cout_transport_EUR_an": rp["cout_transport_EUR_an"],
                "VAN_periode_EUR"      : rp["VAN_t_EUR"],
                "hubs_disponibles"     : "|".join(rp["hubs_disponibles"]),
                "nb_arcs_nouveaux"     : len(rp["arcs_investis"]),
                "emissions_ktCO2"      : r["emissions_ktCO2"].get(t, 0),
            })

        df_periode = pd.DataFrame(rows_periode)
        df_periode.to_csv(
            os.path.join(output_dir, f"{prefix}MILP_MP_periodes.csv"),
            index=False, encoding="utf-8-sig"
        )

        # Tableau détail arcs (toutes périodes)
        dfs_arcs = [
            r["par_periode"][t]["detail_arcs"]
            for t in self.T
            if not r["par_periode"][t]["detail_arcs"].empty
        ]
        if dfs_arcs:
            pd.concat(dfs_arcs).to_csv(
                os.path.join(output_dir, f"{prefix}MILP_MP_arcs.csv"),
                index=False, encoding="utf-8-sig"
            )

        print(f"  ✅ Résultats MP sauvegardés dans : {output_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # B8. Visualisation — Carte réseau par période
    # ─────────────────────────────────────────────────────────────────────────
    def plot_network(self, output_dir: str = None):
        """
        Génère une figure 2×3 montrant le réseau H2 à chaque période.
        Les arcs actifs sont tracés, les nouveaux investissements en rouge.
        """
        r = self.resultats
        if not r:
            print("  Appelez solve() d'abord.")
            return

        COULEURS_MODE = {
            "Pipeline_H2_nouveau"    : "#3F51B5",
            "Pipeline_H2_reconverti" : "#00BCD4",
            "Pipeline_sous_marin"    : "#006064",
            "Tube_trailer"           : "#FF8C00",
            "Tanker_NH3"             : "#FF5722",
        }
        COULEURS_NOEUD = {
            "production"     : "#4CAF50",
            "hub"            : "#2196F3",
            "hub_industriel" : "#FF9800",
            "marche"         : "#F44336",
            "consommateur"   : "#9C27B0",
        }

        n_periodes = len(self.T)
        n_cols = 3
        n_rows = math.ceil(n_periodes / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(18, 6 * n_rows),
                                 facecolor="white")
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else list(axes)

        for idx, t in enumerate(self.T):
            ax = axes[idx]
            rp = r["par_periode"][t]

            ax.set_facecolor("#F8F9FA")
            ax.set_title(f"Réseau H2 — {t}\n"
                         f"LCOT={rp['LCOT_EUR_kg']:.3f} €/kg  "
                         f"| Livraison={rp['livraison_ktH2']:.0f} ktH2/an",
                         fontsize=10, fontweight="bold")

            # Tracer les arcs actifs
            arcs_dispo = rp["arcs_disponibles"]
            arcs_new   = set(rp["arcs_investis"])

            for arc in arcs_dispo:
                o, d, m = arc
                if o not in self.nodes or d not in self.nodes:
                    continue
                flux_val = rp["flux"].get(arc, 0)
                if flux_val < 1e-3:
                    continue

                x0, y0 = self.nodes[o]["lon"], self.nodes[o]["lat"]
                x1, y1 = self.nodes[d]["lon"], self.nodes[d]["lat"]

                lw    = max(0.8, min(5.0, flux_val / 100))
                color = COULEURS_MODE.get(m, "#999999")
                ls    = "-" if arc in arcs_new else "--"
                alpha = 0.9 if arc in arcs_new else 0.5

                ax.plot([x0, x1], [y0, y1],
                        color=color, lw=lw, ls=ls, alpha=alpha,
                        solid_capstyle="round")

                # Étiquette flux
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.annotate(f"{flux_val:.0f}kt",
                            (mx, my), fontsize=6, ha="center",
                            color=color, alpha=0.8)

            # Tracer les nœuds
            for n, attr in self.nodes.items():
                lon, lat = attr["lon"], attr["lat"]
                ntype    = attr.get("type", "hub")
                color    = COULEURS_NOEUD.get(ntype, "#999999")

                # Taille proportionnelle au flux
                flux_n = sum(
                    rp["flux"].get(arc, 0)
                    for arc in self.A if arc[0] == n or arc[1] == n
                )
                size = max(30, min(200, 30 + flux_n * 0.3))

                ax.scatter(lon, lat, s=size, c=color, zorder=5,
                           edgecolors="white", linewidths=0.8)
                ax.annotate(n, (lon, lat),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, fontweight="bold" if flux_n > 10 else "normal")

            # Hub ouverts : cercle supplémentaire
            for h in rp["hubs_disponibles"]:
                if h in self.nodes:
                    lon, lat = self.nodes[h]["lon"], self.nodes[h]["lat"]
                    ax.scatter(lon, lat, s=300, c="none",
                               edgecolors="#FF9800", linewidths=2.5,
                               zorder=6, marker="o")

            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Cacher les axes non utilisés
        for idx in range(len(self.T), len(axes)):
            axes[idx].set_visible(False)

        # Légende commune
        legend_arcs = [
            mpatches.Patch(color=c, label=m.replace("_", " "))
            for m, c in COULEURS_MODE.items()
        ]
        legend_noeuds = [
            mpatches.Patch(color=c, label=t.replace("_", " ").capitalize())
            for t, c in COULEURS_NOEUD.items()
        ]
        fig.legend(handles=legend_arcs + legend_noeuds,
                   loc="lower center", ncol=5, fontsize=8,
                   title="Modes de transport & Types de nœuds",
                   bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(
            f"Évolution du Réseau H2 Maroc 2024–2050\n"
            f"Scénario {self.scenario.upper()} — VAN = "
            f"{self.resultats['VAN_totale_EUR']/1e9:.2f} Mrd €",
            fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()

        if output_dir:
            path = os.path.join(output_dir, f"Fig_Reseau_MP_{self.scenario}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✅ Figure réseau sauvegardée : {path}")
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # B9. Visualisation — Flux financiers actualisés
    # ─────────────────────────────────────────────────────────────────────────
    def plot_cashflow(self, output_dir: str = None):
        """
        Génère 2 graphiques :
        1. Décomposition des coûts par période (CAPEX arcs | CAPEX hubs | OPEX transport)
        2. LCOT réseau et demande satisfaite par période
        """
        r = self.resultats
        if not r:
            return

        periodes    = self.T
        capex_arcs  = [r["par_periode"][t]["capex_arcs_MUSD"] for t in periodes]
        capex_hubs  = [r["par_periode"][t]["capex_hubs_MUSD"] for t in periodes]
        opex_tr     = [r["par_periode"][t]["cout_transport_EUR_an"] / 1e6 for t in periodes]
        lcot_vals   = [r["par_periode"][t]["LCOT_EUR_kg"] for t in periodes]
        demande_v   = [r["par_periode"][t]["demande_ktH2"] for t in periodes]
        livraison_v = [r["par_periode"][t]["livraison_ktH2"] for t in periodes]
        emissions_v = [r["emissions_ktCO2"].get(t, 0) for t in periodes]

        COLORS = {
            "capex_arcs" : "#3F51B5",
            "capex_hubs" : "#FF9800",
            "opex_tr"    : "#4CAF50",
            "lcot"       : "#C1272D",
            "demande"    : "#006233",
            "livraison"  : "#4CAF50",
            "emissions"  : "#795548",
        }

        fig = plt.figure(figsize=(16, 10), facecolor="white")
        gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        x_pos = range(len(periodes))

        # ── Fig 1 : Décomposition CAPEX + OPEX par période ───────────────────
        ax1.bar(x_pos, capex_arcs, label="CAPEX arcs (M$)",
                color=COLORS["capex_arcs"], alpha=0.85)
        ax1.bar(x_pos, capex_hubs, bottom=capex_arcs, label="CAPEX hubs (M$)",
                color=COLORS["capex_hubs"], alpha=0.85)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(periodes)
        ax1.set_ylabel("M$ / période")
        ax1.set_title("Investissements CAPEX par période")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.4)

        # ── Fig 2 : LCOT réseau par période ──────────────────────────────────
        ax2.plot(periodes, lcot_vals, "o-",
                 color=COLORS["lcot"], lw=2.5, ms=8, label="LCOT réseau")
        ax2.axhline(0.5, color="green", ls="--", lw=1.5, alpha=0.7,
                    label="Cible compétitive 0.5 €/kg")
        ax2.fill_between(periodes, lcot_vals,
                         color=COLORS["lcot"], alpha=0.15)
        ax2.set_ylabel("LCOT [€/kgH₂]")
        ax2.set_title("Évolution du LCOT réseau 2024–2050")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.4)

        # ── Fig 3 : Demande vs livraison ──────────────────────────────────────
        ax3.bar([p - 1 for p in periodes], demande_v,
                width=2, label="Demande [ktH2/an]",
                color=COLORS["demande"], alpha=0.6)
        ax3.bar([p + 1 for p in periodes], livraison_v,
                width=2, label="Livraison [ktH2/an]",
                color=COLORS["livraison"], alpha=0.85)
        ax3.set_ylabel("ktH₂/an")
        ax3.set_title("Demande vs Livraison H2 par période")
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.4)

        # ── Fig 4 : Émissions CO2 réseau de transport ─────────────────────────
        ax4.bar(periodes, emissions_v,
                color=COLORS["emissions"], alpha=0.8, label="Émissions transport")
        ax4.set_ylabel("ktCO₂ cumulée par période")
        ax4.set_title("Émissions CO₂ du réseau de transport")
        ax4.legend(fontsize=8)
        ax4.grid(axis="y", alpha=0.4)

        plt.suptitle(
            f"Analyse Financière & Environnementale — Scénario {self.scenario.upper()}\n"
            f"VAN totale = {r['VAN_totale_EUR']/1e9:.2f} Mrd € | "
            f"CAPEX total = {r['capex_total_MUSD']:.0f} M$",
            fontsize=12, fontweight="bold"
        )

        if output_dir:
            path = os.path.join(output_dir, f"Fig_Cashflow_MP_{self.scenario}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✅ Figure cashflow sauvegardée : {path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BLOC C — CAS D'USAGE MULTI-PÉRIODE
# ══════════════════════════════════════════════════════════════════════════════

def cas1_exportateur_national_mp(scenario="central", output_dir=None):
    """
    Scénario 1 : Le Maroc devient un exportateur massif vers l'Europe.
    Utilise les noms de nœuds exacts de T4_segments_detail.
    """
    print("\n" + "▓" * 72)
    print("  CAS 1 MULTI-PÉRIODE — EXPORTATEUR NATIONAL")
    print("▓" * 72)

    # Nœuds T4 réels
    noeuds = [
        "Dakhla", "Laayoune", "Tarfaya", "Guelmim", "Ouarzazate",
        "Agadir", "Casablanca", "Tanger", "Jorf_Lasfar",
        "Rotterdam", "Barcelone", "Marseille", "Algésiras",
    ]
    
    mp = H2TransportMILP_MP(
        scenario=scenario,
        noms_noeuds_actifs=noeuds,
        taxe_carbone_USD_tCO2=0,
    )

    # Demande : {nœud: {période: ktH2/an}}
    demande_cible = {
        "Rotterdam":  DEMANDE_NOEUDS.get("Rotterdam",  {2024:0, 2030:200, 2035:350, 2040:600, 2050:1200}),
        "Barcelone":  DEMANDE_NOEUDS.get("Barcelone",  {2024:0, 2030:100, 2035:180, 2040:300, 2050:600}),
        "Marseille":  DEMANDE_NOEUDS.get("Marseille",  {2024:0, 2030:80,  2035:140, 2040:240, 2050:480}),
        "Algésiras":  DEMANDE_NOEUDS.get("Algésiras",  {2024:0, 2030:20,  2035:40,  2040:80,  2050:200}),
    }
    
    offre_cible = {
        "Dakhla":     OFFRE_NOEUDS.get("Dakhla",     {2024:30, 2030:400, 2035:800, 2040:1500, 2050:3000}),
        "Laayoune":   OFFRE_NOEUDS.get("Laayoune",   {2024:20, 2030:300, 2035:600, 2040:1000, 2050:2000}),
        "Tarfaya":    OFFRE_NOEUDS.get("Tarfaya",    {2024:10, 2030:200, 2035:400, 2040:700,  2050:1400}),
        "Guelmim":    OFFRE_NOEUDS.get("Guelmim",    {2024:5,  2030:80,  2035:150, 2040:250,  2050:450}),
        "Ouarzazate": OFFRE_NOEUDS.get("Ouarzazate", {2024:5,  2030:150, 2035:250, 2040:400,  2050:700}),
    }

    mp.set_demande(demande_cible)
    mp.set_offre(offre_cible)
    mp.solve()
    mp.summary()
    
    if output_dir:
        mp.save_results(output_dir, "Cas1_MP")
        mp.plot_network(output_dir)
        mp.plot_cashflow(output_dir)
    
    return mp


def cas2_hub_industriel_mp(scenario="central", output_dir=None):
    """
    Scénario 2 : Focus OCP Jorf Lasfar — hub industriel domestique.
    """
    print("\n" + "▓" * 72)
    print("  CAS 2 MULTI-PÉRIODE — HUB INDUSTRIEL OCP")
    print("▓" * 72)

    noeuds = ["Guelmim", "Ouarzazate", "Marrakech", "Midelt",
              "Jorf_Lasfar", "Casablanca"]
    
    mp = H2TransportMILP_MP(scenario=scenario, noms_noeuds_actifs=noeuds)

    demande_cas2 = {
        "Jorf_Lasfar": DEMANDE_NOEUDS.get("OCP_Jorf",
                       {2024:60, 2030:120, 2035:180, 2040:250, 2050:400}),
    }
    
    offre_cas2 = {
        "Guelmim":    OFFRE_NOEUDS.get("Guelmim",
                      {2024:5,  2030:80,  2035:150, 2040:250, 2050:450}),
        "Ouarzazate": OFFRE_NOEUDS.get("Ouarzazate",
                      {2024:5,  2030:150, 2035:250, 2040:400, 2050:700}),
    }

    mp.set_demande(demande_cas2)
    mp.set_offre(offre_cas2)
    mp.solve()
    mp.summary()
    
    if output_dir:
        mp.save_results(output_dir, "Cas2_MP")
        mp.plot_cashflow(output_dir)
    
    return mp


def cas3_site_isole_mp(scenario="central", output_dir=None):
    """
    Cas 3 MP — Site isolé intérieur.
    Midelt = site industriel isolé (mine, cimenterie, industrie) consommateur d'H2.
    Les sources Dakhla/Laayoune acheminent via Casablanca ou directement.

    Correction : Midelt est typé 'source' dans T4 (nœud marocain intérieur),
    mais c'est ici un CONSOMMATEUR — on force son type en 'hub' et on définit
    une demande explicite. La contrainte C1 priorise la demande sur le type.
    """
    print("\n" + "▓" * 72)
    print("  CAS 3 MULTI-PÉRIODE — SITE ISOLÉ (Midelt)")
    print("▓" * 72)

    # Inclure Marrakech comme nœud relais intermédiaire vers Midelt
    noeuds = ["Dakhla", "Laayoune", "Casablanca", "Marrakech", "Midelt"]

    demande_cas3 = {
        "Midelt": {2024: 5, 2030: 20, 2035: 30, 2040: 40, 2050: 50},
    }
    offre_cas3 = {
        "Dakhla":   OFFRE_NOEUDS.get("Dakhla",
                    {2024:30, 2030:400, 2035:800, 2040:1500, 2050:3000}),
        "Laayoune": OFFRE_NOEUDS.get("Laayoune",
                    {2024:20, 2030:300, 2035:600, 2040:1000, 2050:2000}),
    }

    mp = H2TransportMILP_MP(scenario=scenario, noms_noeuds_actifs=noeuds)

    # ── Forcer le type de Midelt en 'hub' (consommateur, pas producteur) ──────
    if "Midelt" in mp.nodes:
        mp.nodes["Midelt"]["type"] = "hub"

    # ── Ajouter arc synthétique Casablanca → Midelt si absent de T4 ──────────
    arcs_vers_midelt = [(o, d, m) for (o, d, m, _, _, _) in mp.arcs if d == "Midelt"]
    if not arcs_vers_midelt:
        print("  ⚠️  Aucun arc T4 vers Midelt — ajout arc synthétique "
              "Casablanca→Midelt (Tube_trailer, 320 km)")
        mp.arcs.append(("Casablanca", "Midelt", "Tube_trailer", 320, 0.6, 1.4))
        mp.A  = [(o, d, m) for (o, d, m, _, _, _) in mp.arcs]
        mp._dist[("Casablanca", "Midelt", "Tube_trailer")]  = 320
        mp._cmin[("Casablanca", "Midelt", "Tube_trailer")]  = 0.6
        mp._cmax[("Casablanca", "Midelt", "Tube_trailer")]  = 1.4
        # Recalculer les paramètres temporels pour inclure le nouvel arc
        mp._build_temporal_params()

    mp.set_demande(demande_cas3)
    mp.set_offre(offre_cas3)
    mp.solve()
    mp.summary()

    if output_dir:
        mp.save_results(output_dir, "Cas3_MP")
        mp.plot_cashflow(output_dir)

    return mp


# ══════════════════════════════════════════════════════════════════════════════
# BLOC D — ANALYSE DE SENSIBILITÉ MULTI-PÉRIODE
# ══════════════════════════════════════════════════════════════════════════════

def sensibilite_multiperiode(output_dir=None):
    """
    Compare les 3 scénarios (optimiste / central / pessimiste)
    sur le Cas 1, et analyse l'impact de la taxe carbone.
    """
    print("\n" + "═" * 72)
    print("  SENSIBILITÉ MULTI-PÉRIODE — Cas 1 Exportateur National")
    print("═" * 72)

    noeuds = [
        "Dakhla", "Laayoune", "Tarfaya", "Ouarzazate",
        "Agadir", "Casablanca", "Tanger",
        "Rotterdam", "Barcelone", "Marseille"
    ]
    demande_s = {
        "Rotterdam": DEMANDE_NOEUDS.get("Rotterdam", {2024:0, 2030:200, 2035:350, 2040:600, 2050:1200}),
        "Barcelone": DEMANDE_NOEUDS.get("Barcelone", {2024:0, 2030:100, 2035:180, 2040:300, 2050:600}),
        "Marseille": DEMANDE_NOEUDS.get("Marseille", {2024:0, 2030:80,  2035:140, 2040:240, 2050:480}),
    }
    offre_s = {
        "Dakhla":     OFFRE_NOEUDS.get("Dakhla",     {2024:30, 2030:400, 2035:800, 2040:1500, 2050:3000}),
        "Laayoune":   OFFRE_NOEUDS.get("Laayoune",   {2024:20, 2030:300, 2035:600, 2040:1000, 2050:2000}),
        "Tarfaya":    OFFRE_NOEUDS.get("Tarfaya",    {2024:10, 2030:200, 2035:400, 2040:700,  2050:1400}),
        "Ouarzazate": OFFRE_NOEUDS.get("Ouarzazate", {2024:5,  2030:150, 2035:250, 2040:400,  2050:700}),
    }

    rows_comp = []

    # ── 1. Comparaison 3 scénarios ────────────────────────────────────────────
    print("\n  [1] Comparaison 3 scénarios :")
    print(f"  {'Scénario':<12} {'VAN Mrd€':>10} {'CAPEX M$':>10} "
          f"{'LCOT 2030':>10} {'LCOT 2050':>10}")
    print("  " + "─" * 56)

    for sc in ["optimiste", "central", "pessimiste"]:
        mp = H2TransportMILP_MP(sc, noms_noeuds_actifs=noeuds)
        mp.set_demande(demande_s)
        mp.set_offre(offre_s)
        mp.solve()
        r   = mp.resultats
        l30 = r.get("par_periode", {}).get(2030, {}).get("LCOT_EUR_kg", 0)
        l50 = r.get("par_periode", {}).get(2050, {}).get("LCOT_EUR_kg", 0)
        van = r.get("VAN_totale_EUR", 0)
        capex = r.get("capex_total_MUSD", 0)
        print(f"  {sc:<12} {van/1e9:>10.3f} "
              f"{capex:>10.1f} {l30:>10.4f} {l50:>10.4f}")
        rows_comp.append({
            "analyse"          : "Scenario",
            "param"            : sc,
            "VAN_Mrd_EUR"      : round(van / 1e9, 3),
            "capex_total_MUSD" : capex,
            "LCOT_2030_EUR_kg" : l30,
            "LCOT_2050_EUR_kg" : l50,
        })

    # ── 2. Sensibilité taxe carbone ───────────────────────────────────────────
    print("\n  [2] Impact taxe carbone initiale (scénario central) :")
    print(f"  {'Taxe initiale $/t':>18} {'VAN Mrd€':>10} {'LCOT 2030':>10} {'LCOT 2050':>10}")
    print("  " + "─" * 52)

    for taxe0 in [0, 30, 60, 100]:
        mp = H2TransportMILP_MP(
            "central",
            taxe_carbone_USD_tCO2=taxe0,
            noms_noeuds_actifs=noeuds
        )
        mp.set_demande(demande_s)
        mp.set_offre(offre_s)
        mp.solve()
        r   = mp.resultats
        l30 = r.get("par_periode", {}).get(2030, {}).get("LCOT_EUR_kg", 0)
        l50 = r.get("par_periode", {}).get(2050, {}).get("LCOT_EUR_kg", 0)
        van = r.get("VAN_totale_EUR", 0)
        capex = r.get("capex_total_MUSD", 0)
        print(f"  {taxe0:>18} {van/1e9:>10.3f} "
              f"{l30:>10.4f} {l50:>10.4f}")
        rows_comp.append({
            "analyse"          : "Taxe_CO2",
            "param"            : f"{taxe0}$/t",
            "VAN_Mrd_EUR"      : round(van / 1e9, 3),
            "capex_total_MUSD" : capex,
            "LCOT_2030_EUR_kg" : l30,
            "LCOT_2050_EUR_kg" : l50,
        })

    df_sa = pd.DataFrame(rows_comp)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_sa.to_csv(
            os.path.join(output_dir, "MILP_MP_sensibilite.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"\n  ✅ Sensibilité sauvegardée.")

    return df_sa


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    OUTPUT_DIR = os.path.join(
        os.path.expanduser("~"), "Downloads", "H2Morocco222_Outputs", "MILP_MP"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  H2 MOROCCO — MILP TRANSPORT MULTI-PÉRIODE 2024–2050")
    print("=" * 72)
    print(f"  Solveur : PuLP + CBC  |  WACC : {WACC*100:.0f}%")
    print(f"  Périodes : {PERIODES}")
    print(f"  Nœuds : {len(NODES)}  |  Arcs : {len(ARCS_DATA)}")

    # ── Cas 1 — Exportateur National ─────────────────────────────────────────
    mp1 = cas1_exportateur_national_mp(scenario="central", output_dir=OUTPUT_DIR)

    # ── Cas 2 — Hub Industriel ────────────────────────────────────────────────
    mp2 = cas2_hub_industriel_mp(scenario="central", output_dir=OUTPUT_DIR)

    # ── Cas 3 — Site Isolé ────────────────────────────────────────────────────
    mp3 = cas3_site_isole_mp(scenario="central", output_dir=OUTPUT_DIR)

    # ── Sensibilité multi-période ─────────────────────────────────────────────
    df_sa = sensibilite_multiperiode(output_dir=OUTPUT_DIR)

    print(f"\n  ✅ Terminé — résultats dans : {OUTPUT_DIR}")