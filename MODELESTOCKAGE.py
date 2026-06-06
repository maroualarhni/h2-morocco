

# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODÈLE DE STOCKAGE H₂ — LCOS (Version Jours Dynamiques v2.2)              ║
║                                                                            ║
║                                                                              ║
║  NOUVEAUTÉS v2.2 :                                                           ║
║  [DYN-1] Jours de stockage calculés depuis le profil 8760h par défaut       ║
║  [DYN-2] Argument CLI --jours pour forcer une valeur manuellement           ║
║  [DYN-3] Origine des jours affichée clairement dans chaque résultat         ║
║  [DYN-4] Sensibilité LCOS aux jours de stockage affichée en tableau         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import logging
import os
from typing import Dict, Optional
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_an
import numpy as np
import pandas as pd
from scipy.stats import triang

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONSTANTES PHYSIQUES
# ─────────────────────────────────────────────────────────────────────────────
LHV_H2   = 33.33
HHV_H2   = 39.41
R_IDEAL  = 8.314
M_H2     = 2.016e-3
T_STD    = 293.15
GAMMA_H2 = 1.41
ANNEES   = [2024, 2030, 2035, 2040, 2050]

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "H2Storage_Model_Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# [DYN-1] JOURS DE STOCKAGE — VALEURS PAR DÉFAUT (utilisées UNIQUEMENT si
#         aucun profil 8760h n'est fourni et --jours n'est pas spécifié)
# Ces valeurs sont des estimations régionales de la littérature.
# Priorité : profil 8760h > --jours CLI > valeur par défaut ci-dessous
# ─────────────────────────────────────────────────────────────────────────────
JOURS_STOCKAGE_DEFAULT = {
    "Ouarzazate":  20,   # Fort ensoleillement, faible vent → stockage long (cycles saisonniers)
    "Laayoune":    10,
    "Dakhla":       7,   # Vent quasi-constant → stockage court
    "Tanger":      15,
    "Jorf_Lasfar": 12,
    "Guelmim":     18,
    "_default":    14,
}

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKS IEA/IRENA 2024
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARKS_LCOS = {
    "GH2_350bar": {"min": 0.30, "max": 1.50, "source": "IEA 2024"},
    "GH2_700bar": {"min": 0.50, "max": 2.00, "source": "IEA 2024"},
    "LH2":        {"min": 1.50, "max": 4.00, "source": "IRENA 2023"},
    "NH3":        {"min": 1.20, "max": 3.50, "source": "IEA 2024"},
    "LOHC":       {"min": 1.50, "max": 4.50, "source": "IRENA 2023"},
    "Caverne":    {"min": 0.10, "max": 0.50, "source": "IEA 2024"},
    "eMethanol":  {"min": 1.00, "max": 3.50, "source": "IRENA 2023"},
}

NOTES_HORS_PLAGE = {
    "NH3": (
        "NH3 dépasse le plafond IEA en 2024 : coût du craqueur élevé + "
        "rendement aller-retour ~61%. Compétitif après 2035."
    ),
    "LH2": (
        "LH2 dépasse le plafond IRENA en 2024 : CAPEX liquéfacteur ~200M€ + "
        "9.5 kWh/kg de consommation. Compétitif à grande échelle après 2035."
    ),
    "GH2_700bar": (
        "GH2_700bar légèrement au-dessus du plafond IEA en 2024 : CAPEX tank "
        "élevé + compression 5 étages. Compétitif après 2030."
    ),
}

def validate_vs_benchmark(tech: str, lcos: float) -> str:
    if tech not in BENCHMARKS_LCOS:
        return "N/A"
    b = BENCHMARKS_LCOS[tech]
    if lcos < b["min"]:
        return f"⚠ Sous plancher ({b['min']} €/kg — {b['source']})"
    elif lcos > b["max"]:
        note = NOTES_HORS_PLAGE.get(tech, "")
        return f"⚠ Au-dessus plafond ({b['max']} €/kg — {b['source']}) | {note}"
    else:
        return f"✅ Dans plage [{b['min']}–{b['max']}] ({b['source']})"

# ─────────────────────────────────────────────────────────────────────────────
# FILTRE GÉOLOGIQUE PAR RÉGION
# ─────────────────────────────────────────────────────────────────────────────
TECH_DISPONIBLE_REGION = {
    "Dakhla":      ["GH2_350bar", "GH2_700bar", "LH2", "NH3", "LOHC", "eMethanol"],
    "Ouarzazate":  ["GH2_350bar", "GH2_700bar", "NH3", "eMethanol"],
    "Laayoune":    ["GH2_350bar", "GH2_700bar", "LH2", "NH3", "LOHC", "eMethanol"],
    "Tanger":      ["GH2_350bar", "GH2_700bar", "LH2", "NH3", "LOHC", "Caverne", "eMethanol"],
    "Jorf_Lasfar": ["GH2_350bar", "GH2_700bar", "NH3", "LOHC", "Caverne", "eMethanol"],
    "Guelmim":     ["GH2_350bar", "GH2_700bar", "NH3", "eMethanol"],
    "_default":    ["GH2_350bar", "GH2_700bar", "LH2", "NH3", "LOHC", "Caverne", "eMethanol"],
}

DENSITIES = {
    "GH2_350bar": 23.5, "GH2_700bar": 40.2, "LH2": 70.8,
    "NH3": 682, "LOHC": 64.5, "Caverne": 120, "eMethanol": 792
}


# ─────────────────────────────────────────────────────────────────────────────
# COURBES D'APPRENTISSAGE
# ─────────────────────────────────────────────────────────────────────────────
LEARNING = {
    "GH2_tank":    {"c24": 600,   "r1": 0.030, "r2": 0.020, "floor": 200},
    "GH2_tank700": {"c24": 900,   "r1": 0.028, "r2": 0.018, "floor": 280},
    "LH2_liq":     {"c24": 200e6, "r1": 0.040, "r2": 0.025, "floor": 80e6},
    "NH3_synth":   {"c24": 130e6, "r1": 0.025, "r2": 0.015, "floor": 60e6},
    "NH3_crack":   {"c24": 90e6,  "r1": 0.050, "r2": 0.030, "floor": 20e6},
    "LOHC_sys":    {"c24": 100e6, "r1": 0.035, "r2": 0.025, "floor": 45e6},
    "EMeth_syn":   {"c24": 75e6,  "r1": 0.030, "r2": 0.020, "floor": 35e6},
    "Caverne":     {"c24": 400,   "r1": 0.010, "r2": 0.005, "floor": 300},
}

def capex_learning(key: str, year: int) -> float:
    c = LEARNING[key]
    if year <= 2024:
        return c["c24"]
    elif year <= 2030:
        val = c["c24"] * (1 - c["r1"]) ** (year - 2024)
    else:
        c2030 = c["c24"] * (1 - c["r1"]) ** (2030 - 2024)
        val   = c2030 * (1 - c["r2"]) ** (year - 2030)
    return max(val, c["floor"])


# ─────────────────────────────────────────────────────────────────────────────
# [DYN-1] PROFIL SYNTHÉTIQUE AMÉLIORÉ — SAISONNALITÉ + MIX RÉGIONAL
# ─────────────────────────────────────────────────────────────────────────────
# Source données : résultats GEE v3 (12 régions du Maroc)
# CF solaire : GHI/8760 × PR=0.80 (norme IEC 61724-1 / NREL PVWatts v8)
# CF éolien  : Weibull-Rayleigh k=2, η_terrain=0.85 (IEC 61400-12-1)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import math

# ═════════════════════════════════════════════════════════════════════════════
# A) DONNÉES GEE DES 12 RÉGIONS
# ═════════════════════════════════════════════════════════════════════════════

DONNEES_GEE_REGIONS = {
    "Dakhla":           {"ghi": 2393.1, "vent": 8.71, "s_sol": 3.191, "s_eol": 3.054, "eol_viable": True},
    "Laayoune":         {"ghi": 2282.2, "vent": 7.43, "s_sol": 3.092, "s_eol": 2.733, "eol_viable": True},
    "Guelmim":          {"ghi": 2273.6, "vent": 5.93, "s_sol": 3.092, "s_eol": 2.802, "eol_viable": True},
    "Souss_Massa":      {"ghi": 2241.2, "vent": 5.45, "s_sol": 3.252, "s_eol": 2.256, "eol_viable": True},
    "Marrakech_Safi":   {"ghi": 2152.6, "vent": 5.68, "s_sol": 2.950, "s_eol": 2.816, "eol_viable": True},
    "Casablanca":       {"ghi": 2048.6, "vent": 3.55, "s_sol": 2.596, "s_eol": 2.414, "eol_viable": True},
    "Draa_Tafilalet":   {"ghi": 2252.9, "vent": 1.95, "s_sol": 3.241, "s_eol": 1.939, "eol_viable": False},
    "Beni_Mellal":      {"ghi": 2134.1, "vent": 1.88, "s_sol": 2.745, "s_eol": 1.984, "eol_viable": False},
    "Fes_Meknes":       {"ghi": 2120.1, "vent": 1.79, "s_sol": 2.887, "s_eol": 2.212, "eol_viable": False},
    "Rabat_Sale":       {"ghi": 1989.9, "vent": 3.20, "s_sol": 2.581, "s_eol": 2.299, "eol_viable": True},
    "Oriental":         {"ghi": 2205.6, "vent": 4.50, "s_sol": 3.241, "s_eol": 2.227, "eol_viable": True},
    # ── TANGER CORRIGÉ ──────────────────────────────────────────────────
    # Le pixel GEE original (35.87°N, -3.84°W) tombait dans le Rif montagneux
    # → vent 1.28 m/s = ABSURDE pour la région du détroit de Gibraltar
    # Correction : données NASA POWER / Atlas éolien CDER pour la zone côtière
    # du détroit (Jbel Khalladi 120 MW, Koudia Al Baida 50 MW installés)
    # Vent réel côte détroit : 7.5-9.0 m/s à 100m (mesures mâts CDER)
    "Tanger":           {"ghi": 1971.6, "vent": 7.80, "s_sol": 2.572, "s_eol": 3.100, "eol_viable": True},
}

# ═════════════════════════════════════════════════════════════════════════════
# B) CALCUL DU MIX RÉGIONAL DYNAMIQUE
# ═════════════════════════════════════════════════════════════════════════════

SEUIL_VENT_VIABLE = 3.0   # m/s — seuil physique cut-in turbine
SEUIL_CF_EOL_MIX  = 0.15  # CF minimum pour inclure l'éolien dans le mix
                           # Source : IRENA 2024 "Renewable Power Generation Costs"
                           # Un CF < 15% donne un LCOE éolien > 0.15 €/kWh,
                           # non compétitif vs solaire (~0.03-0.05 €/kWh au Maroc).
                           # Investir dans l'éolien en dessous de ce seuil
                           # augmente le coût sans bénéfice de diversification.

def calculer_mix_regional(region_data: dict) -> dict:
    """
    Calcule le mix solaire/éolien optimal pour chaque région
    basé sur les scores d'aptitude GEE et la viabilité éolienne.

    Règles (2 seuils) :
    1. Seuil physique  : vent < 3 m/s → éolien impossible (turbine ne tourne pas)
    2. Seuil économique : CF_éolien < 15% → éolien non rentable (LCOE trop élevé)
       → Dans les 2 cas : 100% solaire

    Si éolien viable :
    - Pondération proportionnelle aux scores d'aptitude GEE
    - Bonus éolien +15% si vent > 7 m/s (sites exceptionnels type Dakhla)
    - Bonus solaire +10% si GHI > 2300 (sites type Ouarzazate)
    """
    mix = {}
    for region, d in region_data.items():
        # Calculer le CF éolien pour vérifier le seuil économique
        cf_eol = cf_eolien_weibull(d["vent"])

        if not d["eol_viable"] or d["vent"] < SEUIL_VENT_VIABLE or cf_eol < SEUIL_CF_EOL_MIX:
            # Éolien non viable (physiquement ou économiquement) → 100% solaire
            mix[region] = {"solaire": 1.0, "eolien": 0.0}
        else:
            total = d["s_sol"] + d["s_eol"]
            w_sol = d["s_sol"] / total
            w_eol = d["s_eol"] / total

            if d["vent"] > 7.0:
                w_eol = min(w_eol + 0.15, 0.80)
                w_sol = 1.0 - w_eol
            elif d["ghi"] > 2300:
                w_sol = min(w_sol + 0.10, 0.85)
                w_eol = 1.0 - w_sol

            mix[region] = {
                "solaire": round(w_sol, 3),
                "eolien":  round(w_eol, 3),
            }
    return mix

# NOTE : MIX_REGIONAL est calculé APRÈS cf_eolien_weibull (voir section C)

# ═════════════════════════════════════════════════════════════════════════════
# C) CAPACITY FACTORS PAR RÉGION
# ═════════════════════════════════════════════════════════════════════════════

def cf_eolien_weibull(v_mean, k=2, v_ci=3, v_r=12, v_o=25, eta=0.85):
    """
    CF éolien par distribution de Weibull (Rayleigh si k=2).
    Source : IEC 61400-12-1 / Manwell, McGowan, Rogers 2009
    Calibration : Weibull(9.5 m/s) × 0.85 → CF=45.9% ≈ Tarfaya réel 43%

    Args:
        v_mean : vitesse moyenne du vent à hauteur moyeu (m/s)
        k      : paramètre de forme Weibull (2 = Rayleigh)
        v_ci   : vitesse cut-in (m/s)
        v_r    : vitesse nominale (m/s)
        v_o    : vitesse cut-out (m/s)
        eta    : rendement terrain (wake + dispo + pertes élec)
    """
    if v_mean < SEUIL_VENT_VIABLE:
        return 0.0
    c = v_mean / math.gamma(1 + 1/k)
    num   = math.exp(-(v_ci/c)**k) - math.exp(-(v_r/c)**k)
    denom = (v_r/c)**k - (v_ci/c)**k
    cf = (num / denom - math.exp(-(v_o/c)**k)) * eta
    return round(max(cf, 0.0), 4)

# CF solaire : GHI / 8760 × PR (PR=0.80, IEC 61724-1, NREL PVWatts v8)
# Validation : Ouarzazate → CF calc=19.84% ≈ NOOR réel 19.8%
CF_SOLAIRE_REGION = {
    r: round(d["ghi"] / 8760 * 0.80, 4)
    for r, d in DONNEES_GEE_REGIONS.items()
}

CF_EOLIEN_REGION = {
    r: cf_eolien_weibull(d["vent"])
    for r, d in DONNEES_GEE_REGIONS.items()
}

# Calcul du mix APRÈS que cf_eolien_weibull soit défini
MIX_REGIONAL = calculer_mix_regional(DONNEES_GEE_REGIONS)

# ═════════════════════════════════════════════════════════════════════════════
# D) SAISONNALITÉ MENSUELLE
# ═════════════════════════════════════════════════════════════════════════════

# Solaire — facteur multiplicatif par mois (Maroc, latitude ~30°N)
SAISONNALITE_SOLAIRE = [0.70, 0.80, 0.95, 1.05, 1.15, 1.20,
                        1.25, 1.20, 1.10, 0.95, 0.75, 0.65]

# Éolien — profils saisonniers par zone climatique
SAISONNALITE_EOLIEN = {
    # Côte atlantique : alizés forts oct-fév, calme juin-août
    "atlantique": [1.15, 1.10, 1.05, 0.95, 0.85, 0.75,
                   0.70, 0.72, 0.85, 1.00, 1.15, 1.20],
    # Détroit de Gibraltar : régime hivernal dominant
    "nord":       [1.25, 1.15, 1.05, 0.90, 0.80, 0.70,
                   0.65, 0.68, 0.80, 1.00, 1.20, 1.30],
    # Intérieur continental : thermique estival modéré
    "interieur":  [0.90, 0.85, 0.95, 1.05, 1.10, 1.15,
                   1.10, 1.05, 1.00, 0.95, 0.90, 0.85],
}

# Mapping région → zone climatique éolienne
ZONE_EOLIEN = {
    "Dakhla": "atlantique", "Laayoune": "atlantique",
    "Guelmim": "atlantique", "Souss_Massa": "atlantique",
    "Marrakech_Safi": "atlantique", "Casablanca": "atlantique",
    "Tanger": "nord",
    "Draa_Tafilalet": "interieur", "Beni_Mellal": "interieur",
    "Fes_Meknes": "interieur", "Rabat_Sale": "interieur",
    "Oriental": "interieur",
}

# ═════════════════════════════════════════════════════════════════════════════
# E) CYCLE SOLAIRE JOURNALIER (variable par mois, latitude Maroc)
# ═════════════════════════════════════════════════════════════════════════════

# Heure de lever du soleil approx par mois (GMT, latitude ~30°N)
LEVER_SOLEIL = [7.5, 7.0, 6.5, 6.0, 5.5, 5.5, 6.0, 6.0, 6.5, 7.0, 7.0, 7.5]
# Durée du jour (heures) par mois
DUREE_JOUR   = [10.5, 11.0, 12.0, 13.0, 14.0, 14.5, 14.5, 14.0, 13.0, 12.0, 11.0, 10.5]

# Mapping heure → mois (0-11) pour 8760h (année non bissextile)
JOURS_PAR_MOIS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MOIS_PAR_HEURE = np.repeat(np.arange(12), [j * 24 for j in JOURS_PAR_MOIS])

# ═════════════════════════════════════════════════════════════════════════════
# F) GÉNÉRATION DU PROFIL 8760h
# ═════════════════════════════════════════════════════════════════════════════

def _generer_composante_solaire(rng: np.random.Generator) -> np.ndarray:
    """
    Composante solaire 8760h avec :
    - Cycle journalier sinusoïdal (lever/coucher variable par mois)
    - Saisonnalité mensuelle
    - Nébulosité aléatoire (variabilité inter-horaire)
    """
    solar = np.zeros(8760)
    for h in range(8760):
        m      = MOIS_PAR_HEURE[h]
        h_jour = h % 24
        lever  = LEVER_SOLEIL[m]
        duree  = DUREE_JOUR[m]
        if lever <= h_jour <= lever + duree:
            phase = (h_jour - lever) / duree * np.pi
            solar[h] = np.sin(phase) * SAISONNALITE_SOLAIRE[m]

    # Nébulosité : facteur 0.3–1.0 (jamais zéro total si le soleil est levé)
    nebulosite = np.clip(1.0 - 0.2 * rng.standard_normal(8760), 0.3, 1.0)
    solar *= nebulosite
    return solar


def _generer_composante_eolienne(
    region: str,
    cf_eol: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Composante éolienne 8760h avec :
    - CF Weibull régional
    - Saisonnalité mensuelle par zone climatique
    - Bruit autocorrélé (α=0.85) simulant la persistance météo
    """
    if cf_eol <= 0:
        return np.zeros(8760)

    zone     = ZONE_EOLIEN.get(region, "atlantique")
    sais_eol = SAISONNALITE_EOLIEN[zone]

    # Bruit autocorrélé — AR(1), coefficient α=0.85
    # Simule que le vent persiste d'une heure à l'autre
    alpha     = 0.85
    bruit_brut = rng.standard_normal(8760)
    bruit_c    = np.zeros(8760)
    bruit_c[0] = bruit_brut[0]
    for i in range(1, 8760):
        bruit_c[i] = alpha * bruit_c[i-1] + math.sqrt(1 - alpha**2) * bruit_brut[i]

    wind = np.zeros(8760)
    for h in range(8760):
        m = MOIS_PAR_HEURE[h]
        wind[h] = cf_eol * sais_eol[m] + 0.15 * bruit_c[h]

    return np.clip(wind, 0, 1)


def _ecretage_et_redistribution(
    profil: np.ndarray,
    qty_an: float,
    debit_max_ratio: float
) -> np.ndarray:
    """
    Écrête la production au débit max et redistribue l'excès
    sur les heures creuses (proxy contrainte de stockage).

    Args:
        profil          : profil brut normalisé (kg/h)
        qty_an          : production annuelle cible (kg)
        debit_max_ratio : ratio max prod instantanée / prod moyenne
    """
    debit_moyen = qty_an / 8760
    debit_max   = debit_moyen * debit_max_ratio

    exces  = np.maximum(profil - debit_max, 0)
    profil = np.minimum(profil, debit_max)

    if exces.sum() > 0:
        # Identifier les heures où on peut redistribuer
        heures_creuses = profil < debit_moyen * 0.5
        if heures_creuses.any():
            capacite_dispo = debit_max - profil[heures_creuses]
            redistrib = min(exces.sum(), capacite_dispo.sum())
            if capacite_dispo.sum() > 0:
                profil[heures_creuses] += redistrib * (
                    capacite_dispo / capacite_dispo.sum()
                )

    return profil


def generer_profil_synthetique(
    region: str,
    qty_an: float,
    debit_max_ratio: float = 2.5,
    seed: int = 42
) -> np.ndarray:
    """
    Génère un profil de production H₂ horaire (8760h) synthétique.

    Intègre :
    1. Mix solaire/éolien optimal par région (calculé depuis scores GEE)
    2. Saisonnalité mensuelle solaire + éolienne (par zone climatique)
    3. Cycle journalier solaire réaliste (lever/coucher par mois)
    4. Nébulosité aléatoire sur le solaire
    5. Vent autocorrélé (AR1, α=0.85) — persistance météorologique
    6. CF éolien Weibull-Rayleigh calibré sur données terrain
    7. Écrêtage au débit max + redistribution heures creuses

    Args:
        region          : nom de la région (clé de DONNEES_GEE_REGIONS)
        qty_an          : production annuelle cible (kg H₂)
        debit_max_ratio : ratio max production instantanée / moyenne
                          (2.5 = on ne stocke pas plus de 2.5× la moyenne)
        seed            : graine aléatoire pour reproductibilité

    Returns:
        np.ndarray : 8760 valeurs (kg H₂/h)
    """
    rng = np.random.default_rng(seed)

    # Récupération du mix et des CF régionaux
    mix    = MIX_REGIONAL.get(region, {"solaire": 1.0, "eolien": 0.0})
    cf_eol = CF_EOLIEN_REGION.get(region, 0.0)

    # Génération des composantes
    solar = _generer_composante_solaire(rng)
    wind  = _generer_composante_eolienne(region, cf_eol, rng)

    # Combinaison selon le mix GEE régional
    profil = mix["solaire"] * solar + mix["eolien"] * wind
    profil = np.maximum(profil, 0)

    # Normalisation à la production annuelle cible
    if profil.sum() > 0:
        profil = profil / profil.sum() * qty_an
    else:
        profil = np.full(8760, qty_an / 8760)

    # Écrêtage et redistribution
    profil = _ecretage_et_redistribution(profil, qty_an, debit_max_ratio)

    return profil


# ═════════════════════════════════════════════════════════════════════════════
# G) CALCUL DES JOURS DE STOCKAGE DEPUIS UN PROFIL 8760h
# ═════════════════════════════════════════════════════════════════════════════

def jours_stockage_from_profil(
    profil_8760h: np.ndarray,
    debit_charge_max: float = None,
    debit_decharge_max: float = None
) -> float:
    """
    Calcule les jours de stockage nécessaires à partir d'un profil horaire.

    Méthode : analyse du déficit cumulé maximum entre production et demande
    moyenne constante (méthode du réservoir équivalent).

    Améliorations vs version simple :
    - Prise en compte des contraintes de débit charge/décharge
    - Minimum garanti de 1 jour

    Args:
        profil_8760h        : array 8760 valeurs de production H₂ (kg/h)
        debit_charge_max    : kg/h max en charge (None = illimité)
        debit_decharge_max  : kg/h max en décharge (None = illimité)

    Returns:
        jours_stockage : durée de stockage dimensionnante (jours), min 1 jour
    """
    demande_moyenne = np.mean(profil_8760h)
    ecart = profil_8760h - demande_moyenne

    # Appliquer les contraintes de débit si spécifiées
    if debit_charge_max is not None:
        ecart = np.where(ecart > 0, np.minimum(ecart, debit_charge_max), ecart)
    if debit_decharge_max is not None:
        ecart = np.where(ecart < 0, np.maximum(ecart, -debit_decharge_max), ecart)

    deficit_cumule   = np.cumsum(ecart)
    amplitude        = deficit_cumule.max() - deficit_cumule.min()
    prod_journaliere = demande_moyenne * 24

    jours = amplitude / prod_journaliere if prod_journaliere > 0 else 14
    return round(max(jours, 1.0), 1)


# ═════════════════════════════════════════════════════════════════════════════
# H) ANALYSE DE SENSIBILITÉ LCOS AUX JOURS DE STOCKAGE
# ═════════════════════════════════════════════════════════════════════════════

def analyse_sensibilite_jours(
    optimizer_class,
    region: str,
    annee: int,
    scenario: str,
    qty: float,
    lcoh_prod: float,
    plage_jours: list = None
) -> pd.DataFrame:
    """
    Calcule le LCOS de chaque technologie pour différentes durées de stockage.

    Args:
        optimizer_class : classe de l'optimiseur de stockage
        region          : nom de la région
        annee           : année de projection
        scenario        : 'optimiste', 'central', 'pessimiste'
        qty             : quantité annuelle H₂ (kg)
        lcoh_prod       : LCOH de production (€/kg)
        plage_jours     : liste de durées à tester

    Returns:
        DataFrame avec colonnes : jours_test, tech, LCOS, LCOH_total
    """
    if plage_jours is None:
        plage_jours = [3, 5, 7, 10, 14, 20, 30]

    rows = []
    for j in plage_jours:
        model = optimizer_class(region, annee, scenario)
        model.jours        = j
        model.source_jours = f"sensibilité ({j}j)"
        df_j = model.run_all(qty, lcoh_prod)
        df_j["jours_test"] = j
        rows.append(df_j[["tech", "LCOS", "LCOH_total", "jours_test"]])

    df_sens = pd.concat(rows, ignore_index=True)

    print("\n Sensibilité du LCOS aux jours de stockage")
    print("─" * 80)
    pivot = df_sens.pivot(index="jours_test", columns="tech", values="LCOS").round(3)
    print(pivot.to_string())
    print("\n  → Lecture : chaque cellule = LCOS (€/kg) pour N jours de stockage")
    print("  → Un LCOS qui augmente fortement avec les jours = technologie")
    print("    dont le CAPEX de réservoir est le facteur dominant.\n")

    return df_sens


# ═════════════════════════════════════════════════════════════════════════════
# I) SCÉNARIOS + BREAK-EVEN (inchangés)
# ═════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "optimiste":  {"lt_adj": 0.9,  "wacc": 0.06, "px_adj": 0.85},
    "central":    {"lt_adj": 1.0,  "wacc": 0.08, "px_adj": 1.0},
    "pessimiste": {"lt_adj": 1.15, "wacc": 0.11, "px_adj": 1.25},
}

PRIX_MARCHE_H2 = {
    "europe_2024": 6.0,
    "europe_2030": 4.5,
    "europe_2035": 3.5,
    "europe_2040": 3.0,
    "local_industrie": 4.0,
}

def analyse_breakeven(df: pd.DataFrame, annee: int) -> pd.DataFrame:
    """Ajoute les colonnes de break-even par rapport au prix marché."""
    cle     = f"europe_{annee}" if annee in [2024, 2030, 2035, 2040] else "europe_2030"
    prix    = PRIX_MARCHE_H2.get(cle, 4.5)
    df      = df.copy()
    df["prix_marche_EUR_kg"] = prix
    df["marge_EUR_kg"]       = (prix - df["LCOH_total"]).round(3)
    df["rentable"]           = df["LCOH_total"] < prix
    df["breakeven_vente"]    = df["LCOH_total"].round(2)
    df["surplus_pct"]        = ((prix - df["LCOH_total"]) / prix * 100).round(1)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# J) UTILITAIRE — AFFICHAGE RÉCAPITULATIF
# ═════════════════════════════════════════════════════════════════════════════

def afficher_mix_regions():
    """Affiche le mix solaire/éolien et les CF de toutes les régions."""
    print(f"\n{'Région':<20} {'Sol%':>5} {'Eol%':>5} {'CF_sol':>7} {'CF_eol':>7} "
          f"{'Vent':>6} {'GHI':>7} {'Jours_stk':>10}")
    print("─" * 85)

    for r, m in MIX_REGIONAL.items():
        d = DONNEES_GEE_REGIONS[r]
        # Calcul jours de stockage pour 10 kt/an comme référence
        profil = generer_profil_synthetique(r, 10_000_000)
        jours  = jours_stockage_from_profil(profil)

        print(f"{r:<20} {m['solaire']*100:>5.1f} {m['eolien']*100:>5.1f} "
              f"{CF_SOLAIRE_REGION[r]*100:>6.1f}% {CF_EOLIEN_REGION[r]*100:>6.1f}% "
              f"{d['vent']:>5.1f} {d['ghi']:>7.1f} {jours:>8.1f} j")


def resume_profil(region: str, qty_an: float = 10_000_000):
    """Affiche un résumé détaillé du profil généré pour une région."""
    profil = generer_profil_synthetique(region, qty_an)
    mix    = MIX_REGIONAL.get(region, {})
    jours  = jours_stockage_from_profil(profil)

    print(f"\n{'═'*60}")
    print(f"  PROFIL 8760h — {region}")
    print(f"{'═'*60}")
    print(f"  Mix       : {mix.get('solaire',0)*100:.1f}% solaire / "
          f"{mix.get('eolien',0)*100:.1f}% éolien")
    print(f"  CF sol    : {CF_SOLAIRE_REGION.get(region,0)*100:.1f}%")
    print(f"  CF eol    : {CF_EOLIEN_REGION.get(region,0)*100:.1f}%")
    print(f"  Prod tot  : {profil.sum()/1e6:.2f} kt/an")
    print(f"  Moy       : {profil.mean():.1f} kg/h")
    print(f"  Max       : {profil.max():.1f} kg/h")
    print(f"  Min       : {profil.min():.1f} kg/h")
    print(f"  Heures=0  : {(profil == 0).sum()} h ({(profil == 0).sum()/87.6:.1f}%)")
    print(f"  Jours stk : {jours} jours")
    print(f"{'═'*60}\n")

    return profil, jours


# ═════════════════════════════════════════════════════════════════════════════
# EXÉCUTION DIRECTE (test)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    afficher_mix_regions()
    print("\n")
    for r in ["Dakhla", "Ouarzazate", "Casablanca", "Tanger"]:
        if r == "Ouarzazate":
            r = "Draa_Tafilalet"
        resume_profil(r)
# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def capital_recovery_factor(wacc: float, n: int) -> float:
    if wacc < 1e-9:
        return 1.0 / n
    return (wacc * (1 + wacc) ** n) / ((1 + wacc) ** n - 1)

def compression_work_isentropic(
    p_in: float, p_out: float, stages: int = 2,
    eta_is: float = 0.75, eta_mec: float = 0.95
) -> float:
    if p_out <= p_in:
        return 0.0
    ratio = p_out / p_in
    exp   = (GAMMA_H2 - 1) / (stages * GAMMA_H2)
    w_j   = (
        stages * (GAMMA_H2 / (GAMMA_H2 - 1))
        * (R_IDEAL / M_H2) * T_STD * (ratio ** exp - 1)
    )
    return w_j / 3.6e6 / (eta_is * eta_mec)

# ─────────────────────────────────────────────────────────────────────────────
#
# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS LCOS PAR TECHNOLOGIE
# ─────────────────────────────────────────────────────────────────────────────
def lcos_gh2(qty, cap_kg, p_tank_eurkg, p_comp_eurkw, e_comp, px,
             wacc=0.08, lt=20, lcoh_prod=0.0):
    capex = p_tank_eurkg * cap_kg + p_comp_eurkw
    crf   = capital_recovery_factor(wacc, lt)
    lcos  = (capex * crf + 0.02 * capex + e_comp * qty * px) / qty
    return {"LCOS": round(lcos, 4), "LCOH_total": round(lcos + lcoh_prod, 4),
            "eff": 0.97, "co2": 0.8}

def lcos_gh2_700(qty, cap_kg, p_tank_eurkg, p_comp_eurkw, e_comp, px,
                 wacc=0.08, lt=20, lcoh_prod=0.0):
    capex = p_tank_eurkg * cap_kg + p_comp_eurkw
    crf   = capital_recovery_factor(wacc, lt)
    lcos  = (capex * crf + 0.022 * capex + e_comp * qty * px) / qty
    return {"LCOS": round(lcos, 4), "LCOH_total": round(lcos + lcoh_prod, 4),
            "eff": 0.96, "co2": 1.1}

def lcos_lh2(qty, cap_kg, p_res, p_liq, e_liq, px, boil, j,
             wacc=0.08, lt=20, lcoh_prod=0.0):
    perte = 1 - (1 - boil / 100) ** j
    out   = max(qty * (1 - perte), qty * 0.01)
    capex = p_res * cap_kg + p_liq
    crf   = capital_recovery_factor(wacc, lt)
    lcos  = (capex * crf + 0.025 * capex + e_liq * qty * px) / out
    return {"LCOS": round(lcos, 4), "LCOH_total": round(lcos + lcoh_prod, 4),
            "eff": round(1 - perte, 4), "co2": 4.5}

def lcos_nh3(qty, cs, cstock, ccrack, es, ec, px, eta_s, eta_c,
             wacc=0.08, lt=25, lcoh_prod=0.0):
    eta_rt = (eta_s / 100) * 0.99 * (eta_c / 100)
    capex  = cs + cstock + ccrack
    crf    = capital_recovery_factor(wacc, lt)
    lcos   = (capex * crf + 0.03 * capex + (es + ec + 3.6) * qty * px) / (qty * eta_rt)
    return {"LCOS": round(lcos, 4), "LCOH_total": round(lcos + lcoh_prod, 4),
            "eff": round(eta_rt, 4), "co2": 2.2}

def lcos_lohc(qty, ch, cs, cd, eh, ed, ch_th, px, perte=0.08,
              wacc=0.08, lt=15, lcoh_prod=0.0):
    eta       = 1 - perte
    capex     = ch + cs + cd
    crf       = capital_recovery_factor(wacc, lt)
    heat_cost = ch_th * qty * 0.03
    lcos      = (capex * crf + 0.03 * capex + (eh + ed) * qty * px + heat_cost) / (qty * eta)
    return {"LCOS": round(lcos, 4), "LCOH_total": round(lcos + lcoh_prod, 4),
            "eff": round(eta, 4), "co2": 1.8}

def lcos_cavern(qty, cap_kwh_perkg, e_comp, px, eff=0.98,
                wacc=0.08, lt=50, lcoh_prod=0.0):
    capex = cap_kwh_perkg * qty * LHV_H2
    crf   = capital_recovery_factor(wacc, lt)
    lcos  = (capex * crf + 0.01 * capex + e_comp * qty * px) / (qty * eff)
    return {"LCOS": round(lcos, 4), "LCOH_total": round(lcos + lcoh_prod, 4),
            "eff": eff, "co2": 0.3}

def lcos_emethanol(qty, cs, csto, cr, es, er, px, eta_s, eta_r,
                   wacc=0.08, lt=20, lcoh_prod=0.0, 
                   p_co2=100.0, ratio_co2_h2=7.5):
    """
    Args:
        p_co2 : Prix de la tonne de CO2 capté (€/t_co2)
        ratio_co2_h2 : kg de CO2 nécessaires par kg d'H2 final (~7.5)
    """
    # Rendement Round-Trip
    eta_rt = (eta_s / 100) * 0.998 * (eta_r / 100)
    
    # Capex et Annuité
    capex  = cs + csto + (cr or 0)
    crf    = capital_recovery_factor(wacc, lt)
    
    # Coût du CO2 (p_co2 en €/t, donc on divise par 1000 pour avoir €/kg)
    co2_cost_per_kg_h2 = (p_co2 / 1000) * ratio_co2_h2 * qty
    
    # Calcul LCOS avec ajout du coût CO2 et de l'énergie (es + er + catalyseurs/utilités)
    lcos = (capex * crf + 0.03 * capex + (es + er + 1.5) * qty * px + co2_cost_per_kg_h2) / (qty * eta_rt)
    
    return {
        "LCOS": round(lcos, 4), 
        "LCOH_total": round(lcos + lcoh_prod, 4),
        "eff": round(eta_rt, 4), 
        "co2_impact": round(co2_cost_per_kg_h2 / (qty * eta_rt), 4)
    }

LCOS_FUNCS = {
    "GH2_350bar": lcos_gh2,
    "GH2_700bar": lcos_gh2_700,
    "LH2":        lcos_lh2,
    "NH3":        lcos_nh3,
    "LOHC":       lcos_lohc,
    "Caverne":    lcos_cavern,
    "eMethanol":  lcos_emethanol,
}

# ─────────────────────────────────────────────────────────────────────────────
# MOTEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PRIX ÉLECTRICITÉ PAR RÉGION (EUR/kWh)
# ─────────────────────────────────────────────────────────────────────────────
# Les projets H₂ vert produisent leur PROPRE électricité ENR.
# Le prix = LCOE hybride du site, calculé par BASEDEDONNEES.py :
#   LCOE = (CAPEX × CRF + OPEX) / (CF_hybride × 8760)
#   avec CAPEX PV=550$/kW, CAPEX éolien=1100$/kW, DR=8%
#   Conversion USD→EUR : × 0.9217 (BCE Jan 2024)
#
# Les valeurs sont lues depuis le CSV T1 généré par BASEDEDONNEES.py.
# Si le CSV n'est pas trouvé, des valeurs par défaut sont utilisées.
# ─────────────────────────────────────────────────────────────────────────────

def _charger_prix_elec_depuis_t1() -> dict:
    """
    Lit le LCOE hybride EUR/kWh depuis le CSV T1 (BASEDEDONNEES.py).
    Cherche dans plusieurs emplacements possibles.
    Retourne un dict {region: prix_eur_kwh}.
    """
    import os
    from pathlib import Path

    # Emplacements possibles du CSV T1
    chemins = [
        Path.home() / "Downloads" / "h2pipeline" / "outputs" / "clean_csv" / "t1_ressources_energetiques_clean.csv",
        Path.home() / "Downloads" / "h2pipeline" / "outputs" / "clean_csv" / "t1_ressources_clean.csv",
        Path.home() / "Downloads" / "H2Morocco222_Outputs" / "csv" / "T1_ressources_energetiques.csv",
    ]

    for chemin in chemins:
        if chemin.exists():
            try:
                df = pd.read_csv(chemin, encoding="utf-8-sig")

                # Trouver la colonne LCOE hybride EUR
                col_lcoe_eur = None
                col_lcoe_usd = None
                col_region = None

                for c in df.columns:
                    cl = c.lower()
                    if "lcoe" in cl and "hybride" in cl and "eur" in cl:
                        col_lcoe_eur = c
                    elif "lcoe" in cl and "hybride" in cl and "usd" in cl:
                        col_lcoe_usd = c
                    elif cl in ["region", "région"]:
                        col_region = c

                if col_region is None:
                    continue

                prix = {}
                taux_usd_eur = 0.9217
                # Mapping noms T1 (villes) → noms MODELESTOCKAGE (régions)
                MAPPING_T1_VERS_REGION = {
                    "Agadir":      "Souss_Massa",
                    "Ouarzazate":  "Draa_Tafilalet",
                    "Marrakech":   "Marrakech_Safi",
                    "Midelt":      "Beni_Mellal",
                    "Nador":       "Oriental",
                    "Jorf_Lasfar": "Casablanca",   # même région Casablanca-Settat
                    "Boujdour":    "Laayoune",     # même région Laâyoune-Sakia El Hamra
                    "Guelmim":     "Guelmim",        # même nom
                    "Casablanca":  "Casablanca",     # même nom  
                }

                for _, row in df.iterrows():
                  nom_t1 = str(row[col_region]).strip()
                  # Garder le nom original + ajouter le nom région
                  noms = [nom_t1]
                  if nom_t1 in MAPPING_T1_VERS_REGION:
                     noms.append(MAPPING_T1_VERS_REGION[nom_t1])

                  # Lire le LCOE
                  valeur = None
                  if col_lcoe_eur and pd.notna(row.get(col_lcoe_eur)):
                    valeur = float(row[col_lcoe_eur])
                  elif col_lcoe_usd and pd.notna(row.get(col_lcoe_usd)):
                     valeur = float(row[col_lcoe_usd]) * taux_usd_eur

                  if valeur is not None:
                    for nom in noms:
                     prix[nom] = valeur

                if prix:
                    log.info(f"   ✅ PRIX_ELEC chargé depuis T1 : {chemin.name} ({len(prix)} régions)")
                    for r, p in sorted(prix.items(), key=lambda x: x[1]):
                        log.info(f"      {r:<20} {p:.4f} EUR/kWh (LCOE hybride T1)")
                    return prix

            except Exception as e:
                log.warning(f"   ⚠️ Erreur lecture {chemin}: {e}")

    return {}


# Charger depuis T1
_prix_t1 = _charger_prix_elec_depuis_t1()

if _prix_t1:
    # Utiliser les vraies valeurs T1
    PRIX_ELEC_REGION = _prix_t1
    PRIX_ELEC_REGION["_default"] = sum(_prix_t1.values()) / len(_prix_t1)  # moyenne
    log.info(f"   → Prix par défaut (moyenne) : {PRIX_ELEC_REGION['_default']:.4f} EUR/kWh")
else:
    # Fallback : valeurs calculées manuellement si T1 introuvable
    log.warning("   ⚠️ CSV T1 non trouvé — utilisation des valeurs par défaut")
    log.warning("      Exécutez BASEDEDONNEES.py d'abord pour des valeurs exactes")

    # Formule : LCOE = (CAPEX×CRF + OPEX) / (CF×8760) en USD, × 0.9217 → EUR
    # CAPEX_sol=550, OPEX_sol=12, LT=25, DR=8% → CRF=0.0937
    # CAPEX_eol=1100, OPEX_eol=35, LT=20, DR=8% → CRF=0.1019
    _crf_sol = 0.0937
    _crf_eol = 0.1019
    _taux = 0.9217

    def _lcoe_fallback(cf_sol, cf_eol, w_eol):
        """Calcule le LCOE hybride en EUR/kWh à partir des CF."""
        lcoe_s = (550 * _crf_sol + 12) / (cf_sol * 8760) if cf_sol > 0.01 else 999
        lcoe_e = (1100 * _crf_eol + 35) / (cf_eol * 8760) if cf_eol >= 0.20 else None
        if lcoe_e is not None:
            lcoe_h = w_eol * lcoe_e + (1 - w_eol) * lcoe_s
        else:
            lcoe_h = lcoe_s
        return round(lcoe_h * _taux, 4)

    PRIX_ELEC_REGION = {}
    for r, d in DONNEES_GEE_REGIONS.items():
        cf_s = CF_SOLAIRE_REGION[r]
        cf_e = CF_EOLIEN_REGION[r]
        m = MIX_REGIONAL[r]
        PRIX_ELEC_REGION[r] = _lcoe_fallback(cf_s, cf_e, m["eolien"])
    PRIX_ELEC_REGION["Ouarzazate"] = PRIX_ELEC_REGION.get("Draa_Tafilalet", 0.040)
    PRIX_ELEC_REGION["Jorf_Lasfar"] = PRIX_ELEC_REGION.get("Casablanca", 0.050)
    PRIX_ELEC_REGION["_default"] = 0.045

class StorageOptimizer:
    def __init__(self, region: str = "Dakhla", annee: int = 2024, scenario: str = "central"):
        self.region   = region
        self.annee    = annee
        self.scenario = scenario
        self.sc       = SCENARIOS[scenario]
        self.px       = (
            PRIX_ELEC_REGION.get(region, PRIX_ELEC_REGION["_default"])
            * self.sc["px_adj"]
        )
        self.wacc  = self.sc["wacc"]
        self.techs_disponibles = TECH_DISPONIBLE_REGION.get(
            region, TECH_DISPONIBLE_REGION["_default"]
        )

        # ── [DYN-1] Jours de stockage : valeur par défaut région ─────────
        self.jours        = JOURS_STOCKAGE_DEFAULT.get(
            region, JOURS_STOCKAGE_DEFAULT["_default"]
        )
        self.source_jours = f"défaut région ({self.jours}j)"

    # ── [DYN-2] Setter depuis profil 8760h ───────────────────────────────
    def set_profil_8760h(self, profil: np.ndarray):
        """
        Injecte un profil horaire réel ou synthétique et recalcule
        les jours de stockage dynamiquement.
        La valeur calculée remplace le défaut région.
        """
        jours_calcules    = jours_stockage_from_profil(profil)
        self.jours        = jours_calcules
        self.source_jours = f"profil 8760h calculé ({jours_calcules}j)"
        log.info(
            f"   → [DYN] Jours de stockage recalculés depuis profil : "
            f"{jours_calcules} j  (remplace défaut région)"
        )

    # ── [DYN-2] Setter depuis CLI ─────────────────────────────────────────
    def set_jours_manuel(self, jours: float):
        """
        Force les jours de stockage depuis la CLI (--jours).
        Priorité maximale — écrase le défaut et le profil calculé.
        """
        self.jours        = round(jours, 1)
        self.source_jours = f"CLI --jours ({self.jours}j)"
        log.info(
            f"   → [DYN] Jours de stockage forcés manuellement : "
            f"{self.jours} j  (--jours CLI)"
        )

    def run_all(self, qty: float, lcoh_prod: float = 0.0) -> pd.DataFrame:
        rows = []
        j      = self.jours          # ← valeur dynamique effective
        cap_kg = qty * j / 365.0

        # [DYN-3] Affichage visible de la valeur utilisée
        log.info(
            f"   → [DYN] Jours de stockage utilisés : {j} j  "
            f"[source : {self.source_jours}]"
        )
        log.info(
            f"   → Capacité de stockage dimensionnée : "
            f"{cap_kg:,.0f} kg  ({qty:,.0f} kg/an × {j}j / 365)"
        )

        for tech, func in LCOS_FUNCS.items():

            if tech not in self.techs_disponibles:
                log.info(f"   ⊘ {tech} — non disponible à {self.region}")
                continue

            if tech == "GH2_350bar":
                p = {
                    "qty": qty, "cap_kg": cap_kg,
                    "p_tank_eurkg": capex_learning("GH2_tank", self.annee),
                    "p_comp_eurkw": 1200 * compression_work_isentropic(1, 350, 3) * 0.95,
                    "e_comp":       compression_work_isentropic(1, 350, 3),
                    "px": self.px, "wacc": self.wacc,
                    "lt": int(20 * self.sc["lt_adj"]), "lcoh_prod": lcoh_prod
                }
            elif tech == "GH2_700bar":
                p = {
                    "qty": qty, "cap_kg": cap_kg,
                    "p_tank_eurkg": capex_learning("GH2_tank700", self.annee),
                    "p_comp_eurkw": 1500 * compression_work_isentropic(1, 700, 5) * 0.95,
                    "e_comp":       compression_work_isentropic(1, 700, 5),
                    "px": self.px, "wacc": self.wacc,
                    "lt": int(20 * self.sc["lt_adj"]), "lcoh_prod": lcoh_prod
                }
            elif tech == "LH2":
                p = {
                    "qty": qty, "cap_kg": cap_kg,
                    "p_res": 1200,
                    "p_liq": capex_learning("LH2_liq", self.annee) * 1.1,
                    "e_liq": 9.5, "px": self.px, "boil": 0.2, "j": j,
                    "wacc": self.wacc,
                    "lt": int(20 * self.sc["lt_adj"]), "lcoh_prod": lcoh_prod
                }
            elif tech == "NH3":
                p = {
                    "qty": qty,
                    "cs":     capex_learning("NH3_synth", self.annee),
                    "cstock": 320 * qty / 1000,
                    "ccrack": capex_learning("NH3_crack", self.annee),
                    "es": 10.5, "ec": 15.0, "px": self.px,
                    "eta_s": 72, "eta_c": 85, "wacc": self.wacc,
                    "lt": int(25 * self.sc["lt_adj"]), "lcoh_prod": lcoh_prod
                }
            elif tech == "LOHC":
                p = {
                    "qty": qty,
                    "ch": capex_learning("LOHC_sys", self.annee) * 0.8,
                    "cs": 18e6,
                    "cd": capex_learning("LOHC_sys", self.annee) * 1.1,
                    "eh": 4.5, "ed": 3.0, "ch_th": 8.0, "px": self.px,
                    "perte": 0.08, "wacc": self.wacc,
                    "lt": int(15 * self.sc["lt_adj"]), "lcoh_prod": lcoh_prod
                }
            elif tech == "Caverne":
                p = {
                    "qty": qty, "cap_kwh_perkg": 3.0, "e_comp": 1.5,
                    "px": self.px, "eff": 0.98, "wacc": self.wacc,
                    "lt": int(50 * self.sc["lt_adj"]), "lcoh_prod": lcoh_prod
                }
            elif tech == "eMethanol":
                p = {
                    "qty": qty,
                    "cs":   capex_learning("EMeth_syn", self.annee) * 0.9,
                    "csto": 12e6, "cr": 0,
                    "es": 7.5, "er": 11.0, "px": self.px,
                    "eta_s": 74, "eta_r": 68, "wacc": self.wacc,
                    "lt": int(20 * self.sc["lt_adj"]), "lcoh_prod": lcoh_prod
                }

            res = func(**p)
            res["validation_IEA"] = validate_vs_benchmark(tech, res["LCOS"])
            res.update({
                "tech":           tech,
                "region":         self.region,
                "annee":          self.annee,
                "scenario":       self.scenario,
                "lcoh_prod":      lcoh_prod,
                "jours_stockage": j,                     # ← valeur dynamique
                "source_jours":   self.source_jours,     # ← [DYN-3] origine visible
                "cap_stockage_kg": round(cap_kg, 0),     # ← capacité effective
                "density_kg_m3":  DENSITIES.get(tech, 0),
                "qty_kg_an":      qty,
            })
            rows.append(res)

        df = pd.DataFrame(rows).sort_values("LCOS").reset_index(drop=True)
        df = analyse_breakeven(df, self.annee)
        return df

# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
MC_DIST = {
    "GH2_350bar": {
        "capex_res": (450, 600, 850),
        "px_elec":   (0.015, 0.025, 0.040),
        "wacc":      (0.05, 0.08, 0.12)
    },
    "GH2_700bar": {
        "capex_res": (700, 900, 1200),
        "px_elec":   (0.015, 0.025, 0.040),
        "wacc":      (0.05, 0.08, 0.12)
    },
    "LH2": {
        "p_liq":   (150e6, 200e6, 260e6),
        "boiloff": (0.12, 0.20, 0.32),
        "px_elec": (0.015, 0.025, 0.040)
    },
    "NH3": {
        "cs":      (100e6, 130e6, 170e6),
        "eta_c":   (80, 85, 90),
        "px_elec": (0.015, 0.025, 0.040)
    },
}

def run_mc(tech: str, qty: float, n: int = 5000, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    if tech not in MC_DIST:
        return {"mean": float("nan"), "P10": float("nan"), "P90": float("nan"), "CV_pct": 0}
    dist    = MC_DIST[tech]
    samples = {
        k: triang.rvs((c - a) / (b - a), loc=a, scale=b - a, size=n, random_state=rng)
        for k, (a, c, b) in dist.items()
    }
    if tech in ("GH2_350bar", "GH2_700bar"):
        w, px  = samples["wacc"], samples["px_elec"]
        p_bar  = 350 if tech == "GH2_350bar" else 700
        e_c    = compression_work_isentropic(1, p_bar, 3 if p_bar == 350 else 5)
        cap    = samples["capex_res"] * qty * 15 / 365 + 6e6
        crf    = (w * (1 + w) ** 20) / ((1 + w) ** 20 - 1)
        lcos_v = (cap * crf + 0.02 * cap + e_c * qty * px) / qty
    elif tech == "LH2":
        px, boil = samples["px_elec"], samples["boiloff"]
        perte    = 1 - (1 - boil / 100) ** 10
        out      = np.maximum(qty * (1 - perte), qty * 0.01)
        cap      = 1200 * qty * 10 / 365 + 200e6
        lcos_v   = (cap * 0.10185 + 0.025 * cap + 9.5 * qty * px) / out
    else:
        px, eta_c = samples["px_elec"], samples["eta_c"]
        cap    = 130e6 + 300 * qty / 1000 + 90e6
        crf    = (0.08 * 1.08 ** 25) / (1.08 ** 25 - 1)
        eta    = 0.72 * 0.99 * (eta_c / 100)
        lcos_v = (cap * crf + 0.03 * cap + 29.1 * qty * px) / (qty * eta)
    lcos_v = np.clip(lcos_v, 0.1, 25)
    return {
        "mean":   round(float(np.mean(lcos_v)), 3),
        "P10":    round(float(np.percentile(lcos_v, 10)), 3),
        "P90":    round(float(np.percentile(lcos_v, 90)), 3),
        "CV_pct": round(float(np.std(lcos_v) / np.mean(lcos_v) * 100), 1),
    }

def run_sobol(tech: str, qty: float, n_base: int = 1024) -> Optional[Dict]:
    if not HAS_SALIB or tech not in MC_DIST:
        return None
    dist   = MC_DIST[tech]
    names  = list(dist.keys())
    bounds = [[v[0], v[2]] for v in dist.values()]
    prob   = {"num_vars": len(names), "names": names, "bounds": bounds}
    X      = saltelli.sample(prob, n_base, calc_second_order=False)
    Y      = np.zeros(X.shape[0])
    for i, row in enumerate(X):
        p = {names[j]: row[j] for j in range(len(names))}
        if tech in ("GH2_350bar", "GH2_700bar"):
            p_bar = 350 if tech == "GH2_350bar" else 700
            e_c   = compression_work_isentropic(1, p_bar, 3 if p_bar == 350 else 5)
            cap   = p.get("capex_res", 600) * qty * 15 / 365 + 6e6
            crf   = (p["wacc"] * (1 + p["wacc"]) ** 20) / ((1 + p["wacc"]) ** 20 - 1)
            Y[i]  = (cap * crf + 0.02 * cap + e_c * qty * p["px_elec"]) / qty
        elif tech == "LH2":
            perte = 1 - (1 - p["boiloff"] / 100) ** 10
            out   = max(qty * (1 - perte), qty * 0.01)
            cap   = 1200 * qty * 10 / 365 + 200e6
            Y[i]  = (cap * 0.10185 + 0.025 * cap + 9.5 * qty * p["px_elec"]) / out
        else:
            Y[i] = 2.0
    Si = sobol_an.analyze(prob, Y, calc_second_order=False, print_to_console=False)
    return {
        "tech": tech,
        "S1":   {n: round(float(Si["S1"][j]), 3) for j, n in enumerate(names)},
        "ST":   {n: round(float(Si["ST"][j]), 3) for j, n in enumerate(names)},
    }

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-RÉGIONS
# ─────────────────────────────────────────────────────────────────────────────
def run_multi_region(qty: float, lcoh_prod: float, scenario: str = "central") -> pd.DataFrame:
    regions  = [r for r in JOURS_STOCKAGE_DEFAULT.keys() if r != "_default"]
    all_rows = []
    for region in regions:
        for annee in ANNEES:
            model = StorageOptimizer(region, annee, scenario)
            df    = model.run_all(qty, lcoh_prod)
            all_rows.append(df)
            log.info(
                f"   ✓ {region} | {annee} — optimal: "
                f"{df.iloc[0]['tech']} ({df.iloc[0]['LCOS']:.2f} €/kg) "
                f"| {df.iloc[0]['source_jours']}"
            )
    return pd.concat(all_rows, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT MILP
# ─────────────────────────────────────────────────────────────────────────────
def export_for_milp(df: pd.DataFrame, path: str):
    cols = [
        "tech", "LCOS", "LCOH_total", "lcoh_prod", "eff", "co2",
        "density_kg_m3", "qty_kg_an", "jours_stockage", "source_jours",
        "cap_stockage_kg", "rentable", "marge_EUR_kg"
    ]
    milp = df[[c for c in cols if c in df.columns]].to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(milp, f, indent=2, ensure_ascii=False)
    log.info(f"✅ Export MILP : {path}")

# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Modèle Stockage H₂ — LCOS multi-technologie (v2.2 jours dynamiques)"
    )
    parser.add_argument("--tech",      default="all",
                        choices=["all"] + list(LCOS_FUNCS.keys()))
    parser.add_argument("--qty",       type=float, default=1e7)
    parser.add_argument("--region",    default="Dakhla",
                        choices=[r for r in JOURS_STOCKAGE_DEFAULT if r != "_default"])
    parser.add_argument("--annee",     type=int, default=2024, choices=ANNEES)
    parser.add_argument("--scenario",  default="central",
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--lcoh_prod", type=float, default=3.76,
                        help="LCOH production (€/kg) reçu de l'étape Production")

    # ── [DYN-2] Nouveaux arguments jours ────────────────────────────────
    parser.add_argument("--jours",      type=float, default=None,
                        help="[DYN] Forcer les jours de stockage (ex: --jours 10). "
                             "Priorité sur --profil_8760 et sur la valeur défaut région.")
    parser.add_argument("--profil_8760", action="store_true",
                        help="[DYN] Calculer les jours depuis un profil synthétique 8760h. "
                             "Écrasé par --jours si les deux sont spécifiés.")
    parser.add_argument("--sensibilite_jours", action="store_true",
                        help="[DYN-4] Afficher la sensibilité du LCOS aux jours de stockage "
                             "(teste 3, 5, 7, 10, 14, 20, 30 jours)")

    parser.add_argument("--mc",           action="store_true")
    parser.add_argument("--sobol",        action="store_true")
    parser.add_argument("--multi_region", action="store_true")
    args = parser.parse_args()

    log.info(f"🔹 Stockage H₂ | Région={args.region} | Année={args.annee} | Scénario={args.scenario}")
    log.info(f"   → LCOH_prod reçu de l'étape Production : {args.lcoh_prod} €/kg")

    model = StorageOptimizer(args.region, args.annee, args.scenario)

    # ── Résolution de la source des jours (priorité : --jours > profil > défaut)
    if args.jours is not None:
        model.set_jours_manuel(args.jours)
    elif args.profil_8760:
        log.info("   → Génération du profil synthétique 8760h...")
        profil = generer_profil_synthetique(args.region, args.qty)
        model.set_profil_8760h(profil)
        print(
            f"\n⚠  Profil synthétique utilisé — remplacez par un vrai profil PyPSA "
            f"pour des résultats représentatifs."
        )

    # ── [DYN-3] Résumé clair avant les calculs ───────────────────────────
    print(f"\n⏱  Jours de stockage : {model.jours} j  [{model.source_jours}]")
    print(f"   Capacité stockage : {args.qty * model.jours / 365:,.0f} kg")
    print("─" * 60)

    df = model.run_all(args.qty, args.lcoh_prod)
    if args.tech != "all":
        df = df[df["tech"] == args.tech]

    # ── Affichage principal ─────────────────────────────────────────────
    print("\n📊 Résultats LCOS par technologie de stockage")
    print("─" * 95)
    print(df[[
        "tech", "LCOS", "LCOH_total", "eff", "co2",
        "jours_stockage", "source_jours", "rentable", "marge_EUR_kg"
    ]].to_string(index=False))

    # ── Validation benchmarks ───────────────────────────────────────────
    print("\n📋 Validation vs benchmarks IEA/IRENA 2024")
    print("─" * 95)
    for _, row in df.iterrows():
        print(f"  {row['tech']:<12} LCOS={row['LCOS']:.3f} €/kg")
        print(f"               → {row['validation_IEA']}")

    # ── Break-even ──────────────────────────────────────────────────────
    prix_ref = df.iloc[0]["prix_marche_EUR_kg"]
    print(f"\n💰 Analyse rentabilité (prix marché européen = {prix_ref} €/kg)")
    print("─" * 95)
    for _, row in df.iterrows():
        statut = "✅ RENTABLE" if row["rentable"] else "❌ Non rentable"
        print(
            f"  {row['tech']:<12} LCOH_total={row['LCOH_total']:.2f} "
            f"| marge={row['marge_EUR_kg']:+.2f} €/kg | {statut}"
        )

    # ── [DYN-4] Sensibilité aux jours ───────────────────────────────────
    if args.sensibilite_jours:
        analyse_sensibilite_jours(
            StorageOptimizer, args.region, args.annee, args.scenario,
            args.qty, args.lcoh_prod
        )

    # ── Monte Carlo ─────────────────────────────────────────────────────
    if args.mc:
        print("\n Analyse Monte Carlo (incertitude LCOS)")
        print("─" * 70)
        for tech in df["tech"]:
            r = run_mc(tech, args.qty)
            print(
                f"  {tech:<12} | μ={r['mean']:.2f} | "
                f"P10={r['P10']:.2f} | P90={r['P90']:.2f} | CV={r['CV_pct']:.1f}%"
            )

    # ── Sobol ───────────────────────────────────────────────────────────
    if args.sobol:
        print("\n Indices de Sobol (sensibilité LCOS)")
        print("─" * 70)
        for tech in df["tech"]:
            s = run_sobol(tech, args.qty)
            if s:
                print(f"  {tech} — S1={s['S1']} | ST={s['ST']}")

    # ── Multi-régions ───────────────────────────────────────────────────
    if args.multi_region:
        print("\n🗺  Comparaison multi-régions × multi-années")
        print("─" * 70)
        df_multi = run_multi_region(args.qty, args.lcoh_prod, args.scenario)
        pivot = df_multi.groupby(["region", "annee"]).apply(
            lambda x: x.sort_values("LCOS").iloc[0][["tech", "LCOS", "source_jours"]]
        ).reset_index()
        print(pivot.to_string(index=False))
        out_multi = os.path.join(OUTPUT_DIR, f"storage_multi_region_{args.scenario}.csv")
        df_multi.to_csv(out_multi, index=False)

    # ── Exports ─────────────────────────────────────────────────────────
    out_csv  = os.path.join(OUTPUT_DIR, f"storage_{args.region}_{args.annee}.csv")
    out_json = out_csv.replace(".csv", "_milp_input.json")
    df.to_csv(out_csv, index=False)
    export_for_milp(df, out_json)
    log.info(f"💾 Résultats sauvegardés : {OUTPUT_DIR}")

    print(
        f"\n✅ Technologie optimale : {df.iloc[0]['tech']} "
        f"— LCOS = {df.iloc[0]['LCOS']} €/kg "
        f"| {model.jours} jours [{model.source_jours}]"
    )

if __name__ == "__main__":
    main()