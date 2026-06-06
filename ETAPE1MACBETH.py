# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ÉTAPE 1 — MODÈLE RÉEL V4                                                   ║
║                                                                              ║
║  CORRECTIONS V4 :                                                            ║
║  [PHYSIQUE - SOURCES D'ÉNERGIE]                                              ║
║    1. Électrolyse      → EnR pure  (CI_enr  = 20 gCO2/kWh)                 ║
║    2. Compression/Liq  → réseau hybride (CI_grid par région via CF)         ║
║    3. NH3 / e-méthanol → mix industriel (CI_ind = CI_grid × 1.15)           ║
║                                                                              ║
║  [OPEX TECH-SPÉCIFIQUE]                                                      ║
║    4. NH3   → OPEX_base + coût catalyseur Haber-Bosch (3 ans)               ║
║    5. LH2   → OPEX_base + pertes boil-off (0.2%/j × 14j × prix H2)         ║
║    6. LOHC  → OPEX_base + dégradation huile DBT (2%/an)                    ║
║                                                                              ║
║  [MACBETH]                                                                   ║
║    7. Vérification + correction transitivité avant LP                        ║
║    8. Ancrage explicite s_best=1 / s_worst=0 (LP faisabilité pure)          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/h2morocco_db")
SCHEMA = "h2morocco"

LHV_H2_kWh   = 33.33 #PCI inférieur de l'hydrogène
ETA_ELEC      = 0.70
ELEC_CONS     = LHV_H2_kWh / ETA_ELEC          # ~47.6 kWh/kgH2

# Intensités carbone des 3 sources (gCO2/kWh)
CI_ENR_MIN    = 20.0    # EnR pure PV+éolien
CI_GRID_MAX   = 500.0   # Réseau fossile pur
CI_IND_OVERH  = 1.15    # Overhead procédé industriel (+15%)

# ── Seuils réglementaires CO2 (kgCO2/kgH2) ─────────────────────────────────
# Source : Règlement délégué UE 2023/1184 (RFNBO) + IEA Hydrogen 2023
CO2_SEUIL_UE        = 3.38   # Seuil H2 vert UE (< 3.38 = certifiable RFNBO)
CO2_SEUIL_BEST      = 1.0    # Seuil "H2 vert premium" (< 1 kg = meilleure classe)
CO2_REF_GRIS        = 10.0   # H2 gris SMR sans CCS (référence haute)

# ── CO2 upstream par technologie (kgCO2/kgH2) ───────────────────────────────
# Fabrication équipements + transport + fin de vie (ACV amont)
# Sources : IRENA 2022, IEA 2023, NREL LCA database
CO2_UPSTREAM = {
    'GH2_350bar':     1.2,   # Électrolyseur PEM + réservoir acier
    'GH2_700bar':     1.4,   # Idem + compression haute pression
    'LH2':            1.8,   # Électrolyseur + liquéfacteur (acier, isolation)
    'NH3':            2.1,   # Électrolyseur + réacteur Haber-Bosch + stockage
    'LOHC':           1.6,   # Électrolyseur + colonne hydrogénation/déshydrogénation
    'Caverne_saline': 1.0,   # Électrolyseur + infrastructure souterraine minimale
    'e_methanol':     2.3,   # Électrolyseur + réacteur + capture CO2 source
}

def calculer_co2_upstream_regional(tech, info_region):
    """CO2 upstream modulé par 3 facteurs régionaux physiquement justifiés."""
    BASE = {
        'GH2_350bar': 1.2, 'GH2_700bar': 1.4, 'LH2': 1.8,
        'NH3': 2.1, 'LOHC': 1.6, 'Caverne_saline': 1.0, 'e_methanol': 2.3,
    }
    base = BASE.get(tech, 1.5)
    dist_port = float(info_region.get('distance_port_km', 200))
    f_dist = 1.15 if dist_port > 500 else (1.08 if dist_port > 200 else 1.00)
    cf = float(info_region.get('cf_hybride_pct', 20)) / 100.0
    f_cf = 0.90 if cf > 0.30 else (0.95 if cf > 0.20 else 1.00)
    cout_eau = float(info_region.get('cout_eau_usd_m3', 0.5))
    f_eau = 1.12 if cout_eau > 1.5 else (1.06 if cout_eau > 0.8 else 1.00)
    return base * f_dist * f_cf * f_eau

# OPEX spécifiques NH3
NH3_CATAL_COST_USD_KG = 0.012   # $/kgNH3
NH3_CATAL_LIFE_YRS    = 3.0
NH3_YIELD_KG_PER_KGH2 = 5.6

# OPEX spécifiques LH2
LH2_BOILOFF_PCT_DAY   = 0.20    # %/jour
LH2_STORAGE_DAYS      = 14.0
LH2_PRICE_USD_KGH2    = 6.0

# OPEX spécifiques LOHC
LOHC_OIL_DEGRAD_PCT_YR = 2.0   # %/an
LOHC_OIL_COST_USD_KG   = 3.5
LOHC_OIL_KG_PER_KGH2   = 16.0

N_PERTURBATIONS  = 200
PERTURBATION_PCT = 0.20

DELTA_CAT = {0: 0.0, 1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25}
CRITERIA  = ['CAPEX', 'OPEX', 'Densite', 'LCOS', 'TRL', 'CO2_Real', 'Eau_Real']


def get_engine():
    return create_engine(DB_URL)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 : CHARGEMENT DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

# Valeurs par défaut pour les données manquantes en base
DEFAULTS_T3 = {
    'GH2_350bar':     {'CAPEX_reservoir': 600,  'energie_compression': 2.0,  'OPEX_pct_CAPEX': 2.0,  'energie_synthese': 0.0,  'densite_vol': 23,   'LCOS': 0.4,  'TRL': 9, 'eau_proc_L_kgH2': 9.0,  'CO2_proc_gCO2_kgH2': 0.0},
    'GH2_700bar':     {'CAPEX_reservoir': 900,  'energie_compression': 3.5,  'OPEX_pct_CAPEX': 2.5,  'energie_synthese': 0.0,  'densite_vol': 40,   'LCOS': 0.5,  'TRL': 9, 'eau_proc_L_kgH2': 9.0,  'CO2_proc_gCO2_kgH2': 0.0},
    'LH2':            {'CAPEX_reservoir': 1200, 'energie_compression': 12.0, 'OPEX_pct_CAPEX': 3.5,  'energie_synthese': 0.0,  'densite_vol': 70,   'LCOS': 0.8,  'TRL': 7, 'eau_proc_L_kgH2': 9.0,  'CO2_proc_gCO2_kgH2': 0.0},
    'NH3':            {'CAPEX_stockage':  350,  'energie_compression': 8.0,  'OPEX_pct_CAPEX': 3.0,  'energie_synthese': 10.0, 'densite_vol': 120,  'LCOS': 0.3,  'TRL': 8, 'eau_proc_L_kgH2': 18.0, 'CO2_proc_gCO2_kgH2': 0.0},
    'LOHC':           {'CAPEX_stockage':  400,  'energie_compression': 5.0,  'OPEX_pct_CAPEX': 3.5,  'energie_synthese': 4.0,  'densite_vol': 57,   'LCOS': 0.6,  'TRL': 6, 'eau_proc_L_kgH2': 12.0, 'CO2_proc_gCO2_kgH2': 0.0},
    'Caverne_saline': {'CAPEX_reservoir': 200,  'energie_compression': 1.5,  'OPEX_pct_CAPEX': 1.5,  'energie_synthese': 0.0,  'densite_vol': 200,  'LCOS': 0.2,  'TRL': 8, 'eau_proc_L_kgH2': 9.0,  'CO2_proc_gCO2_kgH2': 0.0},
    'e_methanol':     {'CAPEX_stockage':  80,   'energie_compression': 9.0,  'OPEX_pct_CAPEX': 4.0,  'energie_synthese': 6.0,  'densite_vol': 100,  'LCOS': 0.7,  'TRL': 6, 'eau_proc_L_kgH2': 15.0, 'CO2_proc_gCO2_kgH2': 0.0},
}

def charger_donnees_physiques(engine):
    df_t1 = pd.read_sql(
        f"""SELECT region, ghi_kwh_m2_an, cf_hybride_pct, distance_port_km,
                   cout_eau_usd_m3, connexion_reseau_elec, latitude_n, longitude_w
            FROM {SCHEMA}.t1_ressources""", engine
    ).set_index('region')

    df_t3_raw = pd.read_sql(
        f"SELECT technologie, parametre, valeur_mode FROM {SCHEMA}.t3_technologies_stockage",
        engine
    )
    df_t3 = df_t3_raw.pivot(index='technologie', columns='parametre', values='valeur_mode')
    df_t3.columns = [str(c).strip() for c in df_t3.columns]

    # Compléter les valeurs manquantes avec les defaults
    for tech, defaults in DEFAULTS_T3.items():
        if tech not in df_t3.index:
            df_t3.loc[tech] = np.nan
        for col, val in defaults.items():
            if col not in df_t3.columns:
                df_t3[col] = np.nan
            if pd.isna(df_t3.loc[tech, col]):
                df_t3.loc[tech, col] = val

    # Colonnes unifiées
    if 'CAPEX_systeme_complet' not in df_t3.columns:
        df_t3['CAPEX_systeme_complet'] = np.nan
    for tech in df_t3.index:
        if pd.isna(df_t3.loc[tech, 'CAPEX_systeme_complet']):
            for c in ('CAPEX_stockage', 'CAPEX_reservoir'):
                if c in df_t3.columns and not pd.isna(df_t3.loc[tech, c]):
                    df_t3.loc[tech, 'CAPEX_systeme_complet'] = df_t3.loc[tech, c]
                    break

    if 'OPEX_pct_CAPEX' not in df_t3.columns:
        df_t3['OPEX_pct_CAPEX'] = np.nan

    print(f"✅ Données chargées : {len(df_t1)} régions, {len(df_t3)} technologies")
    print(f"   Colonnes T3 : {list(df_t3.columns)}")
    return df_t1, df_t3

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2A : INTENSITÉS CARBONE PAR SOURCE
# ─────────────────────────────────────────────────────────────────────────────

def get_ci_sources(info_region):
    """
    3 sources d'énergie aux intensités carbone distinctes :

    EnR   (électrolyse PV+éolien)    → CI fixe = 20 gCO2/kWh
          L'électrolyse est toujours sur EnR dédiée dans un projet H2 vert.

    Grid  (compression / liquéfaction) → CI = f(CF_hybride régional)
          Ces procédés peuvent tirer du réseau la nuit ou par absence de vent.
          CI_grid = 500*(1-CF) + 20*CF

    Ind   (synthèse NH3 / méthanol)  → CI = CI_grid * 1.15
          Procédés haute température (Haber-Bosch 400°C, MeOH 250°C) :
          chaleur industrielle non entièrement renouvelable → overhead +15%.
    """
    cf      = float(info_region['cf_hybride_pct']) / 100.0
    ci_enr  = CI_ENR_MIN
    ci_grid = CI_GRID_MAX * (1.0 - cf) + CI_ENR_MIN * cf
    ci_ind  = ci_grid * CI_IND_OVERH
    return ci_enr, ci_grid, ci_ind

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2B : CO2 PAR TECHNOLOGIE
# ─────────────────────────────────────────────────────────────────────────────

# Pour chaque techno : quelle source alimente chaque flux énergétique
# 'enr' | 'grid' | 'ind'
ENERGY_SOURCE_MAP = {
    'GH2_350bar':    {'elec': 'enr', 'compr': 'grid', 'synth': None  },
    'GH2_700bar':    {'elec': 'enr', 'compr': 'grid', 'synth': None  },
    'LH2':           {'elec': 'enr', 'compr': 'grid', 'synth': None  },
    'NH3':           {'elec': 'enr', 'compr': 'grid', 'synth': 'ind' },
    'LOHC':          {'elec': 'enr', 'compr': 'grid', 'synth': 'grid'},
    'Caverne_saline':{'elec': 'enr', 'compr': 'grid', 'synth': None  },
    'e_methanol':    {'elec': 'enr', 'compr': 'grid', 'synth': 'ind' },
}


def calculer_co2_tech(tech, row, ci_enr, ci_grid, ci_ind):
    ci_map  = {'enr': ci_enr, 'grid': ci_grid, 'ind': ci_ind}
    src     = ENERGY_SOURCE_MAP.get(tech, {'elec':'enr','compr':'grid','synth':None})

    e_compr   = float(row.get('energie_compression', 0) or 0)
    e_synth   = float(row.get('energie_synthese',    0) or 0)
    co2_dir   = float(row.get('CO2_proc_gCO2_kgH2', 0) or 0) / 1000.0

    co2_elec  = (ELEC_CONS * ci_map[src['elec']])  / 1000.0
    co2_compr = (e_compr   * ci_map[src['compr']]) / 1000.0
    co2_synth = (e_synth   * ci_map[src['synth'] or src['compr']]) / 1000.0

    return co2_elec + co2_compr + co2_synth + co2_dir


def score_co2_zone(co2_total_kg):
    """
    Score CO2 en 3 zones réglementaires — échelle non-linéaire.

    CORRECTION vs version précédente :
      Ancienne formule : score = (10 - CO2) / 9 × 100
        → Référence arbitraire (10 / 1 kg)
        → Échelle linéaire : réduire de 9→8 rapporte autant que 2→1
        → Aucune discontinuité au seuil réglementaire UE

    Nouvelle formule : 3 zones avec pentes différentes
        Zone A — CO2 > CO2_REF_GRIS         → score = 0   (hors référence)
        Zone B — CO2_SEUIL_UE < CO2 ≤ CO2_REF_GRIS  → score  0–40
                 (H2 non certifiable UE : progrès faiblement récompensé)
        Zone C — CO2_SEUIL_BEST < CO2 ≤ CO2_SEUIL_UE → score 40–85
                 (H2 vert certifiable : zone de performance normale)
        Zone D — CO2 ≤ CO2_SEUIL_BEST              → score 85–100
                 (H2 vert premium : forte prime pour excellence)

    Justification des paliers :
      - 40 pts au seuil UE (3.38 kg) : passe le minimum réglementaire
      - 85 pts au seuil premium (1.0 kg) : excellence reconnue
      - 100 pts impossible à atteindre en pratique (CO2 = 0 théorique)
      → Encourage fortement à passer sous 3.38, puis sous 1.0
    """
    if co2_total_kg >= CO2_REF_GRIS:
        return 0.0

    if co2_total_kg > CO2_SEUIL_UE:
        # Zone B : 0 → 40 pts entre CO2_REF_GRIS et CO2_SEUIL_UE
        frac = (CO2_REF_GRIS - co2_total_kg) / (CO2_REF_GRIS - CO2_SEUIL_UE)
        return round(frac * 40.0, 2)

    if co2_total_kg > CO2_SEUIL_BEST:
        # Zone C : 40 → 85 pts entre CO2_SEUIL_UE et CO2_SEUIL_BEST
        frac = (CO2_SEUIL_UE - co2_total_kg) / (CO2_SEUIL_UE - CO2_SEUIL_BEST)
        return round(40.0 + frac * 45.0, 2)

    # Zone D : 85 → 100 pts pour CO2 ≤ CO2_SEUIL_BEST
    # Asymptote douce : bonus décroissant → jamais 100 en pratique
    frac = max(0.0, (CO2_SEUIL_BEST - co2_total_kg) / CO2_SEUIL_BEST)
    return round(85.0 + min(frac, 1.0) * 15.0, 2)


def score_eau_zone(total_water_L):
    """
    Score Eau en 2 zones avec seuil de stress hydrique réglementaire.

    Seuil FAO/OCDE : > 25 L/kgH2 dans une zone de stress = pénalité forte.
    Seuil d'excellence : < 12 L/kgH2 (électrolyse PEM optimisée, peu de procédé).

    Zone A : total_water > 50 L → score = 0   (inacceptable)
    Zone B : 25 < total_water ≤ 50 → score  0–50  (stress hydrique)
    Zone C : 12 < total_water ≤ 25 → score 50–85  (acceptable)
    Zone D : total_water ≤ 12      → score 85–100  (excellence)
    """
    EAU_MAX   = 50.0   # L/kgH2 : seuil inacceptable
    EAU_STRESS= 25.0   # L/kgH2 : seuil stress hydrique FAO
    EAU_BEST  = 12.0   # L/kgH2 : seuil excellence

    if total_water_L >= EAU_MAX:
        return 0.0

    if total_water_L > EAU_STRESS:
        frac = (EAU_MAX - total_water_L) / (EAU_MAX - EAU_STRESS)
        return round(frac * 50.0, 2)

    if total_water_L > EAU_BEST:
        frac = (EAU_STRESS - total_water_L) / (EAU_STRESS - EAU_BEST)
        return round(50.0 + frac * 35.0, 2)

    frac = max(0.0, (EAU_BEST - total_water_L) / EAU_BEST)
    return round(85.0 + min(frac, 1.0) * 15.0, 2)


def calculer_scores_physiques(df_t1, df_t3, region_cible):
    """
    Calcule les scores CO2 et Eau pour chaque technologie.

    CO2 total = CO2_operationnel (3 sources) + CO2_upstream (ACV amont)
      CO2_op   : calculé dynamiquement via get_ci_sources + ENERGY_SOURCE_MAP
      CO2_up   : valeur fixe par techno (CO2_UPSTREAM dict) — fabrication + transport

    Score CO2 : fonction non-linéaire à 3 zones réglementaires (score_co2_zone)
    Score Eau : fonction non-linéaire à 2 zones avec seuil stress hydrique (score_eau_zone)
    """
    if region_cible not in df_t1.index:
        raise ValueError(f"Région '{region_cible}' introuvable dans T1")

    info                    = df_t1.loc[region_cible]
    ci_enr, ci_grid, ci_ind = get_ci_sources(info)
    coeff_stress            = 2.0 if float(info['cout_eau_usd_m3']) > 0.8 else 1.0

    print(f"\n  Intensités carbone — {region_cible} :")
    print(f"    CI_enr  (ElectrolyseEnR) = {ci_enr:.0f} gCO2/kWh")
    print(f"    CI_grid (Réseau régional)= {ci_grid:.0f} gCO2/kWh")
    print(f"    CI_ind  (Mix industriel) = {ci_ind:.0f} gCO2/kWh")
    print(f"    Seuil H2 vert UE         = {CO2_SEUIL_UE} kgCO2/kgH2")
    print(f"    Seuil H2 vert premium    = {CO2_SEUIL_BEST} kgCO2/kgH2")

    dist_port = float(info.get('distance_port_km', 200))
    cf        = float(info.get('cf_hybride_pct', 20)) / 100.0
    cf_reduction = max(0.80, 1.0 - (cf * 0.5))

    results = {}
    for tech in df_t3.index:
        row = df_t3.loc[tech]

        # CO2 opérationnel (3 sources différenciées)
        co2_op = calculer_co2_tech(tech, row, ci_enr, ci_grid, ci_ind)

        # CO2 upstream ACV amont modulé par facteurs régionaux
        co2_up = calculer_co2_upstream_regional(tech, info)

        # CO2 total cycle de vie complet
        co2_total = co2_op + co2_up

        # Zone réglementaire
        if co2_total <= CO2_SEUIL_BEST:
            zone = 'premium'
        elif co2_total <= CO2_SEUIL_UE:
            zone = 'vert_UE'
        elif co2_total < CO2_REF_GRIS:
            zone = 'hors_certification'
        else:
            zone = 'gris'

        score_co2 = score_co2_zone(co2_total)

        # Eau avec facteur CF
        w_proc      = float(row.get('eau_proc_L_kgH2', 0) or 0)
        total_water = (9.0 + w_proc) * coeff_stress * cf_reduction
        score_eau   = score_eau_zone(total_water)

        # Facteurs site-spécifiques
        port_factor    = max(0, 100 - dist_port * 0.15)
        surface_factor = min(100, info.get('surface_km2', 1000) * 0.05)

        results[tech] = {
            'CO2_op_kg':    round(co2_op,    3),
            'CO2_up_kg':    round(co2_up,    3),
            'CO2_total_kg': round(co2_total, 3),
            'Zone_CO2':     zone,
            'Score_CO2':    score_co2,
            'Water_L':      round(total_water, 2),
            'Score_Eau':    score_eau,
            'Port_factor':  port_factor,
            'Surface_factor': surface_factor,
        }
    return results

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2C : OPEX TECH-SPÉCIFIQUE
# ─────────────────────────────────────────────────────────────────────────────

def calculer_opex_tech(tech, row, capex, prod_tH2_an=1000.0):
    """
    OPEX_total = OPEX_base + OPEX_specifique

    NH3  → surcoût catalyseur Haber-Bosch
           prod × ratio_NH3/H2 × coût_catal / durée_vie
    LH2  → pertes boil-off valorisées
           prod × (%/j / 100) × jours_stock × prix_H2
    LOHC → remplacement huile DBT dégradée
           prod × kg_DBT/kgH2 × coût_DBT × taux_dégradation
    """
    OPEX_BASE_PCT = {
        'GH2_350bar':2.0,'GH2_700bar':2.5,'LH2':3.5,
        'NH3':4.0,'LOHC':3.5,'Caverne_saline':1.5,'e_methanol':4.0,
    }
    pct_t3 = row.get('OPEX_pct_CAPEX', None)
    try:
        pct_val = float(pct_t3)
        if not np.isnan(pct_val):
            opex_base = capex * pct_val / 100.0
        else:
            opex_base = capex * OPEX_BASE_PCT.get(tech, 3.0) / 100.0
    except (TypeError, ValueError):
        opex_base = capex * OPEX_BASE_PCT.get(tech, 3.0) / 100.0

    prod_kgH2 = prod_tH2_an * 1000.0
    opex_spec = 0.0
    source_label = '—'

    if tech == 'NH3':
        opex_spec    = (prod_kgH2 * NH3_YIELD_KG_PER_KGH2
                        * NH3_CATAL_COST_USD_KG / NH3_CATAL_LIFE_YRS)
        source_label = 'Catalyseur Haber-Bosch'

    elif tech == 'LH2':
        opex_spec    = (prod_kgH2 * (LH2_BOILOFF_PCT_DAY / 100.0)
                        * LH2_STORAGE_DAYS * LH2_PRICE_USD_KGH2)
        source_label = f'Boil-off {LH2_BOILOFF_PCT_DAY}%/j × {LH2_STORAGE_DAYS}j'

    elif tech == 'LOHC':
        opex_spec    = (prod_kgH2 * LOHC_OIL_KG_PER_KGH2
                        * LOHC_OIL_COST_USD_KG * LOHC_OIL_DEGRAD_PCT_YR / 100.0)
        source_label = 'Dégradation huile DBT'

    return opex_base + opex_spec, opex_base, opex_spec, source_label


def calculer_opex_reel(df_t3_f, techs, prod_tH2_an=1000.0):
    capex_col = next(
        (c for c in ('CAPEX_systeme_complet','CAPEX_reservoir') if c in df_t3_f.columns),
        None
    )
    opex_total  = pd.Series(index=techs, dtype=float)
    detail_rows = []

    for t in techs:
        row   = df_t3_f.loc[t]
        capex = float(row.get(capex_col, 0) or 0) if capex_col else 0.0
        tot, base, spec, lbl = calculer_opex_tech(t, row, capex, prod_tH2_an)
        opex_total[t] = tot
        detail_rows.append({
            'technologie':     t,
            'CAPEX':           round(capex, 0),
            'OPEX_base':       round(base, 0),
            'OPEX_specifique': round(spec, 0),
            'OPEX_total':      round(tot,  0),
            'Source_surcoût':  lbl,
        })

    return opex_total, pd.DataFrame(detail_rows).set_index('technologie')

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 : AHP
# ─────────────────────────────────────────────────────────────────────────────

class AHPEngine:
    RI_TABLE = {3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}

    def __init__(self, criteria_names, pairwise_matrix, scenario_name):
        self.criteria = criteria_names
        self.matrix   = np.array(pairwise_matrix, dtype=float)
        self.n        = len(criteria_names)
        self.scenario = scenario_name
        self.weights  = None
        self.CR       = None

    def calculate(self):
        col_sums     = self.matrix.sum(axis=0)
        norm         = self.matrix / col_sums
        self.weights = norm.mean(axis=1)
        self.weights /= self.weights.sum()
        Aw           = self.matrix @ self.weights
        lambda_max   = np.mean(Aw / self.weights)
        CI           = (lambda_max - self.n) / (self.n - 1)
        RI           = self.RI_TABLE.get(self.n, 1.45)
        self.CR      = CI / RI if RI else 0.0
        return self.weights, self.CR

    def is_valid(self):
        return self.CR < 0.10


def get_ahp_scenarios():
    mat_export = np.array([
        [1,   1,   3,   1/2, 2,   1/5, 1/4],
        [1,   1,   3,   1/2, 2,   1/5, 1/4],
        [1/3, 1/3, 1,   1/3, 1,   1/7, 1/6],
        [2,   2,   3,   1,   3,   1/3, 1/2],
        [1/2, 1/2, 1,   1/3, 1,   1/5, 1/4],
        [5,   5,   7,   3,   5,   1,   2  ],
        [4,   4,   6,   2,   4,   1/2, 1  ],
    ], dtype=float)

    mat_hub = np.array([
        [1,   2,   4,   1/2, 3,   3,   3  ],
        [1/2, 1,   3,   1/3, 2,   2,   2  ],
        [1/4, 1/3, 1,   1/5, 1,   1,   1  ],
        [2,   3,   5,   1,   4,   4,   4  ],
        [1/3, 1/2, 1,   1/4, 1,   1,   1  ],
        [1/3, 1/2, 1,   1/4, 1,   1,   1  ],
        [1/3, 1/2, 1,   1/4, 1,   1,   1  ],
    ], dtype=float)

    mat_iso = np.array([
        [1,   1,   1/2, 1,   1/4, 1/2, 1/5],
        [1,   1,   1/2, 1,   1/4, 1/2, 1/5],
        [2,   2,   1,   2,   1/2, 1,   1/3],
        [1,   1,   1/2, 1,   1/4, 1/2, 1/5],
        [4,   4,   2,   4,   1,   3,   1/2],
        [2,   2,   1,   2,   1/3, 1,   1/4],
        [5,   5,   3,   5,   2,   4,   1  ],
    ], dtype=float)

    return {
        'EXPORT':         {'matrix': mat_export, 'criteria': CRITERIA},
        'HUB_INDUSTRIEL': {'matrix': mat_hub,    'criteria': CRITERIA},
        'SITE_ISOLE':     {'matrix': mat_iso,    'criteria': CRITERIA},
    }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 : MACBETH CORRIGÉ
# ─────────────────────────────────────────────────────────────────────────────

def build_macbeth_matrix(techs, data_values, sense='min'):
    n   = len(techs)
    mat = np.zeros((n, n), dtype=int)
    v   = np.array(data_values, dtype=float)
    rng = np.ptp(v)
    if rng < 1e-12:
        return mat
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = (v[j]-v[i])/rng if sense == 'min' else (v[i]-v[j])/rng
            a    = abs(diff)
            cat  = (5 if a > 0.75 else 4 if a > 0.50 else 3 if a > 0.30
                    else 2 if a > 0.15 else 1 if a > 0.05 else 0)
            mat[i, j] = cat if diff > 0 else -cat
    return mat


def verifier_transitivite(cat_matrix):
    """
    Corrige les violations de transitivité dans la matrice MACBETH.

    Règle : si i > j (cat>0) et j > k (cat>0) alors i doit être > k.
    Correction : cat(i,k) = max(1, min(cat_ij, cat_jk) - 1)
    Antisymétrie maintenue : cat(k,i) = -cat(i,k)
    """
    n   = cat_matrix.shape[0]
    mat = cat_matrix.copy()
    viol = 0
    for i in range(n):
        for j in range(n):
            if mat[i, j] <= 0:
                continue
            for k in range(n):
                if k == i or mat[j, k] <= 0:
                    continue
                if mat[i, k] <= 0:
                    corr       = max(1, min(int(mat[i,j]), int(mat[j,k])) - 1)
                    mat[i, k]  =  corr
                    mat[k, i]  = -corr
                    viol      += 1
    return mat, viol


def solve_macbeth_lp(techs, cat_matrix):
    """
    LP MACBETH avec ancrage explicite s_best=1, s_worst=0.

    Formulation (faisabilité pure — plus stable que maximiser l'étendue) :
      Variables  : s_i ∈ [0,1]
      Inégalités : s_i - s_j ≥ delta_cat  pour chaque cat(i,j) > 0
      Égalités   : s_best=1, s_worst=0   (ancrage sur sommes de lignes)
      Objectif   : min 0   (pas de biais vers une solution particulière)

    Avantage : toujours borné [0,1], solution unique si transitivité OK.
    """
    n = len(techs)
    if n < 2:
        return {t: 0.5 for t in techs}

    sums      = cat_matrix.sum(axis=1)
    idx_best  = int(np.argmax(sums))
    idx_worst = int(np.argmin(sums))
    if idx_best == idx_worst:
        return {t: 0.5 for t in techs}

    A_ub, b_ub = [], []
    for i in range(n):
        for j in range(n):
            k = int(cat_matrix[i, j])
            if k <= 0:
                continue
            row    = [0.0] * n
            row[i] = -1.0; row[j] = 1.0
            A_ub.append(row)
            b_ub.append(-DELTA_CAT[k])

    row_best         = [0.0] * n; row_best[idx_best]   = 1.0
    row_worst        = [0.0] * n; row_worst[idx_worst]  = 1.0
    A_eq = [row_best, row_worst]
    b_eq = [1.0, 0.0]

    try:
        res = linprog(
            [0.0]*n,
            A_ub=A_ub or None, b_ub=b_ub or None,
            A_eq=A_eq, b_eq=b_eq,
            bounds=[(0.0, 1.0)]*n, method='highs'
        )
    except Exception:
        return None

    if res.status != 0:
        return None

    return {techs[i]: float(res.x[i]) for i in range(n)}


def fallback_rank_scores(techs, cat_matrix):
    sums = cat_matrix.sum(axis=1).astype(float)
    lo, hi = sums.min(), sums.max()
    if hi == lo:
        return {t: 0.5 for t in techs}
    return {techs[i]: float((sums[i]-lo)/(hi-lo)) for i in range(len(techs))}


def solve_macbeth_scores(techs, matrices_dict, criteria_weights):
    scores = {t: 0.0 for t in techs}
    for crit, mat in matrices_dict.items():
        w = criteria_weights.get(crit, 0.0)
        if w < 1e-9:
            continue
        mat_ok, n_viol = verifier_transitivite(mat)
        if n_viol > 0:
            print(f"    ⚠️  {crit} : {n_viol} violation(s) transitivité corrigée(s)")
        s = solve_macbeth_lp(techs, mat_ok)
        if s is None:
            print(f"    ⚠️  LP infaisable '{crit}' → fallback rang")
            s = fallback_rank_scores(techs, mat_ok)
        for t, v in s.items():
            scores[t] += w * v * 100.0
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 : SENSIBILITÉ & ROBUSTESSE
# ─────────────────────────────────────────────────────────────────────────────

def analyser_sensibilite(techs, matrices, weights_base, scen_name,
                          n_pert=N_PERTURBATIONS, pct=PERTURBATION_PCT):
    np.random.seed(42)
    criteria = list(weights_base.keys())
    w_arr    = np.array([weights_base[c] for c in criteria])

    all_ranks  = {t: [] for t in techs}
    all_scores = {t: [] for t in techs}

    for _ in range(n_pert):
        noise  = np.random.uniform(1 - pct, 1 + pct, size=len(w_arr))
        w_pert = w_arr * noise
        w_pert /= w_pert.sum()
        sc     = solve_macbeth_scores(techs, matrices, dict(zip(criteria, w_pert)))
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        for rang, (tech, score) in enumerate(ranked, 1):
            all_ranks[tech].append(rang)
            all_scores[tech].append(score)

    rows = []
    for t in techs:
        rngs = all_ranks[t]
        rows.append({
            'technologie': t,
            'rang_median':  int(np.median(rngs)),
            'rang_std':     round(float(np.std(rngs)), 2),
            'score_moyen':  round(float(np.mean(all_scores[t])), 2),
            'score_std':    round(float(np.std(all_scores[t])), 2),
            'pct_top1':     round(100.0 * rngs.count(1) / n_pert, 1),
        })

    df_rob = pd.DataFrame(rows).sort_values('rang_median')

    print(f"\n  {'─'*62}")
    print(f"  ROBUSTESSE — {scen_name}  (N={n_pert}, ±{int(pct*100)}%)")
    print(f"  {'─'*62}")
    print(f"  {'Tech':<18} {'RgMed':>6} {'±Rg':>5} {'ScMoy':>7} {'±Sc':>5} {'%Top1':>6}")
    for _, r in df_rob.iterrows():
        flag = "✅" if r['rang_std'] < 1.5 else "⚠️ "
        print(f"  {flag} {r['technologie']:<16} {r['rang_median']:>6} "
              f"{r['rang_std']:>5.2f} {r['score_moyen']:>7.1f} "
              f"{r['score_std']:>5.1f} {r['pct_top1']:>5.1f}%")
    return df_rob

# ─────────────────────────────────────────────────────────────────────────────
# EXÉCUTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def run_model_intelligent(region_cible='Dakhla'):
    print("\n" + "="*70)
    print(f"  MODÈLE RÉEL V4 — ANALYSE {region_cible.upper()}")
    print("="*70)

    engine       = get_engine()
    df_t1, df_t3 = charger_donnees_physiques(engine)

    TECH_LIST = ['GH2_350bar','GH2_700bar','LH2','NH3',
                 'LOHC','Caverne_saline','e_methanol']
    techs = [t for t in df_t3.index if t in TECH_LIST]
    if not techs:
        print("❌ Aucune technologie trouvée."); return

    df_t3_f = df_t3.loc[techs]

    # Scores physiques CO2 / Eau
    phys = calculer_scores_physiques(df_t1, df_t3_f, region_cible)

    # OPEX tech-spécifique
    opex_total, opex_detail = calculer_opex_reel(df_t3_f, techs)
    print(f"\n  OPEX décomposé (base + surcoût physique) :")
    print(f"  {'Tech':<18} {'Base $':>10} {'Spéc. $':>10} {'Total $':>10}  Source")
    for t in techs:
        r = opex_detail.loc[t]
        print(f"  {t:<18} {r['OPEX_base']:>10,.0f} {r['OPEX_specifique']:>10,.0f} "
              f"{r['OPEX_total']:>10,.0f}  {r['Source_surcoût']}")

    # data_df
    data_df   = pd.DataFrame(index=techs)
    capex_col = next(
        (c for c in ('CAPEX_systeme_complet','CAPEX_reservoir') if c in df_t3_f.columns),
        None
    )
    if capex_col:
        data_df['CAPEX'] = df_t3_f.loc[techs, capex_col]
    data_df['OPEX'] = opex_total
    for col, crit in [('densite_vol','Densite'),('LCOS','LCOS'),('TRL','TRL')]:
        if col in df_t3_f.columns:
            data_df[crit] = df_t3_f.loc[techs, col]

    for t in techs:
        if t not in phys:
            raise KeyError(f"Tech '{t}' absente de phys")
        data_df.loc[t, 'CO2_Real'] = phys[t]['Score_CO2']
        data_df.loc[t, 'Eau_Real'] = phys[t]['Score_Eau']

    cols_t3 = [c for c in data_df.columns if c not in ('CO2_Real','Eau_Real')]
    data_df[cols_t3] = data_df[cols_t3].fillna(0)

    if data_df['CO2_Real'].sum() == 0 or data_df['Eau_Real'].sum() == 0:
        raise ValueError("CO2_Real ou Eau_Real tous nuls")

    print(f"\n  Scores environnement (CO2 opérationnel + upstream ACV) :")
    print(f"  {'Tech':<18} {'CO2_op':>7} {'CO2_up':>7} {'CO2_tot':>8} {'Zone':<20} {'Sc_CO2':>7} {'Eau_L':>6} {'Sc_Eau':>7}")
    for t in techs:
        p = phys[t]
        print(f"  {t:<18} {p['CO2_op_kg']:>7.2f} {p['CO2_up_kg']:>7.2f} "
              f"{p['CO2_total_kg']:>8.2f} {p['Zone_CO2']:<20} "
              f"{p['Score_CO2']:>7.1f} {p['Water_L']:>6.1f} {p['Score_Eau']:>7.1f}")

    # AHP + MACBETH + Sensibilité
    scenarios      = get_ahp_scenarios()
    sens_crit      = {'CAPEX':'min','OPEX':'min','LCOS':'min',
                      'Densite':'max','TRL':'max','CO2_Real':'max','Eau_Real':'max'}
    final_results  = []
    robustesse_all = []

    for scen_name, scen_data in scenarios.items():
        print(f"\n{'='*70}\n  SCÉNARIO : {scen_name}\n{'='*70}")

        ahp = AHPEngine(scen_data['criteria'], scen_data['matrix'], scen_name)
        weights, cr = ahp.calculate()
        if not ahp.is_valid():
            print(f"  ⚠️  CR={cr:.3f} → poids égaux")
            weights = np.ones(len(CRITERIA)) / len(CRITERIA)
        else:
            print(f"  ✅ AHP valide (CR={cr:.4f})")

        dict_weights = dict(zip(scen_data['criteria'], weights))
        print(f"\n  Poids AHP :")
        for c, w in sorted(dict_weights.items(), key=lambda x: -x[1]):
            print(f"    {c:<12} {w:.3f}  {'█'*int(w*40)}")

        matrices = {}
        for crit in scen_data['criteria']:
            if crit in data_df.columns:
                matrices[crit] = build_macbeth_matrix(
                    techs, data_df[crit].values, sens_crit.get(crit,'min')
                )

        scores  = solve_macbeth_scores(techs, matrices, dict_weights)
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        final_results.append({'Scenario':scen_name,'Ranking':ranking,'Weights':dict_weights})

        print(f"\n  Classement nominal :")
        for rang, (t, sc) in enumerate(ranking, 1):
            print(f"    {rang}. {t:<18} {sc:.1f}/100")

        df_rob = analyser_sensibilite(techs, matrices, dict_weights, scen_name)
        df_rob['scenario'] = scen_name
        robustesse_all.append(df_rob)

    # Sauvegarde
    rows = []
    for res in final_results:
        for rang, (tech, score) in enumerate(res['Ranking'], 1):
            rows.append({'region':region_cible,'scenario':res['Scenario'],
                         'rang':rang,'technologie':tech,'score_global':round(score,2)})
    df_res     = pd.DataFrame(rows)
    df_rob_all = pd.concat(robustesse_all, ignore_index=True)
    df_rob_all['region'] = region_cible

    try:
        df_res.to_sql('results_etape1_v4', engine, schema=SCHEMA,
                      if_exists='replace', index=False)
        df_rob_all.to_sql('results_robustesse_v4', engine, schema=SCHEMA,
                          if_exists='replace', index=False)
        opex_detail.reset_index().assign(region=region_cible).to_sql(
            'opex_detail_v4', engine, schema=SCHEMA, if_exists='replace', index=False)
        print(f"\n💾 results_etape1_v4 + results_robustesse_v4 + opex_detail_v4 → {SCHEMA}")
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde : {e}")

    print("\n" + "="*70 + "\n  TERMINÉ V4\n" + "="*70)
    return df_res, df_rob_all, opex_detail


if __name__ == "__main__":
    import os
    regions = [
        'Dakhla', 'Laayoune', 'Ouarzazate', 'Tanger',
        'Agadir', 'Casablanca', 'Guelmim', 'Boujdour',
        'Nador', 'Marrakech', 'Midelt', 'Jorf_Lasfar'
    ]
    all_results    = []
    all_robustesse = []
    engine = get_engine()

    for region in regions:
        print(f"\n{'='*70}")
        print(f"  LANCEMENT MACBETH — {region}")
        print(f"{'='*70}")
        try:
            df_res, df_rob_all, opex_detail = run_model_intelligent(region)
            if df_res is not None:
                all_results.append(df_res)
            if df_rob_all is not None:
                all_robustesse.append(df_rob_all)
        except Exception as e:
            print(f"Erreur {region}: {e}")
            continue

    if not all_results:
        print("\nAucun résultat genere. Verifiez les erreurs ci-dessus.")
    else:
        df_final     = pd.concat(all_results,    ignore_index=True)
        df_rob_final = pd.concat(all_robustesse, ignore_index=True)

        print(f"\n{'='*70}")
        print("  RESUME FINAL")
        print(f"{'='*70}")
        print(f"  Regions traitees : {df_final['region'].nunique()}")
        print(f"  Scenarios        : {df_final['scenario'].unique().tolist()}")
        print(f"  Total lignes     : {len(df_final)}")

        try:
            df_final.to_sql('results_etape1_v4', engine, schema=SCHEMA,
                            if_exists='replace', index=False)
            df_rob_final.to_sql('results_robustesse_v4', engine, schema=SCHEMA,
                                if_exists='replace', index=False)
            print(f"\nPostgreSQL mis a jour — {len(df_final)} lignes sauvegardees")
        except Exception as e:
            print(f"Erreur sauvegarde DB : {e}")

        save_dir = os.path.dirname(os.path.abspath(__file__))
        df_final.to_csv(os.path.join(save_dir, "results_etape1_v4.csv"),
                        index=False, encoding='utf-8-sig')
        df_rob_final.to_csv(os.path.join(save_dir, "results_robustesse_v4.csv"),
                            index=False, encoding='utf-8-sig')
        print(f"CSV sauvegardes dans {save_dir}")