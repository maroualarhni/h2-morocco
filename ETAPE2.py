# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ÉTAPE 2 — OPTIMISATION PRODUCTION H2 VERT          ║
║   PyPSA (8760h) + NSGA-II (multi-objectifs) + Intermittence + Monte Carlo  ║
║                                                                              ║
║   Objectifs :                                                                ║
║     f1 → Minimiser le LCOH ($/kg H2)                                        ║
║     f2 → Maximiser la fiabilité (taux de couverture demande)                ║
║                                                                              ║
║   Variables décisionnelles :                                                 ║
║     x1 → Capacité PV (MW)                                                   ║
║     x2 → Capacité éolienne (MW)                                             ║
║     x3 → Capacité électrolyseur (MW)                                        ║
║     x4 → Capacité batterie (MWh) — 0 si scénario SANS batterie             ║
║                                                                              ║
║   v2 — Scénarios batterie :                                                ║
║     - Scénarios AVEC et SANS batterie (comparaison systématique)             ║
║     - PyPSA : Slack generator, min_load, bilan corrigé                      ║
║     - Électrolyseur : ramping + min-load + dégradation                      ║
║     - LCOH : décomposé par composante (waterfall)                            ║
║     - Fig13/14 : comparaison batterie + décomposition LCOH                  ║
║     - Cache simulation + Export Excel multi-feuilles                         ║
║                                                                              ║
║    v3 — Intermittence des sources           :                               ║
║     MODE 1 : Optimisation standard (profils déterministes)                  ║
║     MODE 2 : AR(1) — corrélation temporelle + rafales + calmes              ║
║     MODE 3 : HMM — 3 régimes météo (Ensoleillé/Nuageux/Couvert)            ║
║     MODE 4 : AR1 + HMM combinés (intermittence complète)                    ║
║     MODE 5 : Monte Carlo N tirages → IC 90% LCOH et fiabilité               ║
║              Fig15 : profils originaux vs perturbés                          ║
║              Fig16 : distribution LCOH + convergence IC                     ║
║                                                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
import os
import json
import time
import hashlib
from copy import deepcopy
from functools import lru_cache
import logging
import importlib.util, sys

logging.getLogger('pypsa').setLevel(logging.ERROR)
logging.getLogger('linopy').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
np.random.seed(42)

# PyPSA — modélisation réseau électrique
try:
    import pypsa
    PYPSA_OK = True
    print("✓ PyPSA disponible :", pypsa.__version__)
except ImportError:
    PYPSA_OK = False
    print("  PyPSA non installé — pip install pypsa")
    print("   Mode dégradé : simulation simplifiée activée")

# openpyxl pour export Excel multi-feuilles
try:
    import openpyxl
    EXCEL_OK = True
except ImportError:
    EXCEL_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.path.expanduser("~"), "Downloads", "H2Morocco222_Outputs")
OUTPUT_DIR2 = os.path.join(OUTPUT_DIR, "etape2")
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/h2morocco_db")
SCHEMA = "h2morocco"
for sub in ["csv", "figures", "reports"]:
    os.makedirs(f"{OUTPUT_DIR}/{sub}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR2}/{sub}", exist_ok=True)

# Paramètres financiers
PARAMS_FIN = {
    'DR'        : 0.08,   # Taux actualisation 8% (WACC MENA — IEA 2024)
    'LT_PV'     : 25,     # Durée vie PV (ans)
    'LT_EOL'    : 20,     # Durée vie éolien (ans)
    'LT_ELEC'   : 20,     # Durée vie électrolyseur (ans)
    'LT_BAT'    : 15,     # Durée vie batterie (ans)
    'ANNEE_REF' : 2024,
}

# Paramètres techno-économiques
PARAMS_TECHNO = {
    # PV — IRENA 2024
    'CAPEX_PV'       : 550,    # $/kW
    'OPEX_PV'        : 12,     # $/kW/an
    'DEGRAD_PV'      : 0.005,  #  dégradation annuelle 0.5%/an (NREL 2023)
    # Éolien — IRENA 2024
    'CAPEX_EOL'      : 1100,   # $/kW
    'OPEX_EOL'       : 35,     # $/kW/an
    # Électrolyseur PEM — NREL H2A 2024
    'CAPEX_PEM'      : 900,    # $/kW
    'OPEX_PEM'       : 0.03,   # fraction CAPEX/an
    'EFF_PEM'        : 55,     # kWh/kgH2
    'RAMP_PEM'       : 0.10,   # ramping max 10%/min (PEM rapide)
    'MINLOAD_PEM'    : 0.10,   # charge minimale 10% Pnom
    'DEGRAD_PEM'     : 0.015,  # dégradation 1.5%/an efficacité
    # Électrolyseur AEL — IEA 2024
    'CAPEX_AEL'      : 650,    # $/kW
    'OPEX_AEL'       : 0.02,   # fraction CAPEX/an
    'EFF_AEL'        : 52,     # kWh/kgH2
    'RAMP_AEL'       : 0.02,   #  ramping max 2%/min (AEL lent)
    'MINLOAD_AEL'    : 0.20,   # charge minimale 20% Pnom
    'DEGRAD_AEL'     : 0.010,  # dégradation 1.0%/an
    # Batterie Li-ion — IEA 2024
    'CAPEX_BAT'      : 150,    # $/kWh
    'OPEX_BAT'       : 0.01,   # fraction CAPEX/an
    'EFF_BAT_CHG'    : 0.92,   # rendement charge
    'EFF_BAT_DCH'    : 0.92,   # rendement décharge
    'SOC_MIN'        : 0.10,   # état charge minimum 10%
    'SOC_MAX'        : 0.90,   # état charge maximum 90%
    # Eau
    'WATER_CONS'     : 21.1,   # L/kgH2 — IEA 2024
    'WATER_COST'     : 0.72,   # $/m³ — Maroc
    # Durées de vie système (ans) — nécessaires pour _facteur_degradation
    'DUREE_VIE_PEM'  : 20,     # ans — IEA 2024
    'DUREE_VIE_AEL'  : 25,     # ans — IEA 2024
    'DUREE_VIE_SOEC' : 10,     # ans — IEA 2024
}

# ✅ NOUVEAU : Plages par scénario batterie
BOUNDS_AVEC_BAT = {
    'PV_MW'   : (10,  500),
    'EOL_MW'  : (0,   300),
    'ELEC_MW' : (10,  200),
    'BAT_MWH' : (10,  500),   # Batterie obligatoire dans ce scénario
}
BOUNDS_SANS_BAT = {
    'PV_MW'   : (10,  500),
    'EOL_MW'  : (0,   300),
    'ELEC_MW' : (10,  200),
    'BAT_MWH' : (0,   0),     # BAT = 0 forcé dans ce scénario
}

# Demande cible H2
DEMANDE_H2_KGH = 1000  # kg/h (≈ 8760 tH2/an)

# Couleurs
COLORS = {
    'primary'   : '#006233',
    'secondary' : '#C1272D',
    'accent'    : '#FF8C00',
    'PEM'       : '#2196F3',
    'AEL'       : '#4CAF50',
    'PV'        : '#FFC107',
    'EOL'       : '#00BCD4',
    'BAT'       : '#9C27B0',
    'pareto'    : '#E91E63',
    'light_bg'  : '#F8F9FA',
    'grid'      : '#E0E0E0',
    'avec_bat'  : '#6C3483',   # couleur scénario avec batterie
    'sans_bat'  : '#117A65',   #  couleur scénario sans batterie
}

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : COLORS['light_bg'],
    'axes.grid'        : True,
    'grid.color'       : COLORS['grid'],
    'grid.linewidth'   : 0.7,
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
    'axes.titlesize'   : 12,
    'axes.titleweight' : 'bold',
})


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — CHARGEMENT DES DONNÉES T10
# ══════════════════════════════════════════════════════════════════════════════

def charger_profils_T10(region='Dakhla', annee=2024, force_synthetic=False):
    """
    Charge les profils horaires 8760h depuis CSV (base_1.py) ou génère
    un profil synthétique calibré.

    Retourne DataFrame avec :
        - CF_PV_h   : capacity factor PV horaire [0-1]
        - CF_eol_h  : capacity factor éolien horaire [0-1]
    """
    # Essayer plusieurs patterns de noms de fichiers T10
    csv_patterns = [
        os.path.join(OUTPUT_DIR, "csv", f"T10_profils_{region}_{annee}.csv"),
        os.path.join(OUTPUT_DIR, "csv", f"T10_{region}_horaire.csv"),
        os.path.join(OUTPUT_DIR, "csv", f"T10_{region}_{annee}.csv"),
    ]
    csv_path = None
    for p in csv_patterns:
        if os.path.exists(p):
            csv_path = p
            break

    if not force_synthetic and csv_path is not None:
        print(f"  ✓ Chargement T10 depuis CSV : {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if 'CF_PV_h' in df.columns and 'CF_eol_h' in df.columns:
            df = df.iloc[:8760]
            print(f"    → {len(df)} heures | "
                  f"CF_PV={df['CF_PV_h'].mean()*100:.1f}% | "
                  f"CF_eol={df['CF_eol_h'].mean()*100:.1f}%")
            return df
        print("  ⚠️  Colonnes absentes — fallback synthétique")

    print(f"  ℹ️  Génération profil synthétique — {region} {annee}")

    CALIB = {
        'Dakhla'     : {'CF_sol': 0.197, 'CF_eol': 0.415, 'GHI': 2180, 'v_mean': 9.8},
        'Ouarzazate' : {'CF_sol': 0.198, 'CF_eol': 0.225, 'GHI': 2172, 'v_mean': 5.8},
        'Laayoune'   : {'CF_sol': 0.199, 'CF_eol': 0.337, 'GHI': 2175, 'v_mean': 8.2},
        'Tanger'     : {'CF_sol': 0.168, 'CF_eol': 0.156, 'GHI': 1850, 'v_mean': 5.1},
        'Jorf_Lasfar': {'CF_sol': 0.173, 'CF_eol': 0.128, 'GHI': 1900, 'v_mean': 4.8},
        'Agadir'     : {'CF_sol': 0.192, 'CF_eol': 0.164, 'GHI': 2050, 'v_mean': 5.3},
        'Boujdour'   : {'CF_sol': 0.200, 'CF_eol': 0.384, 'GHI': 2160, 'v_mean': 9.1},
        'Casablanca' : {'CF_sol': 0.171, 'CF_eol': 0.105, 'GHI': 1870, 'v_mean': 4.3},
        'Nador'      : {'CF_sol': 0.163, 'CF_eol': 0.180, 'GHI': 1780, 'v_mean': 5.5},
        'Marrakech'  : {'CF_sol': 0.190, 'CF_eol': 0.060, 'GHI': 2080, 'v_mean': 3.8},
        'Midelt'     : {'CF_sol': 0.201, 'CF_eol': 0.150, 'GHI': 2200, 'v_mean': 5.0},
    }
    cal = CALIB.get(region, CALIB['Dakhla'])

    idx        = pd.date_range(f'{annee}-01-01', periods=8760, freq='h')
    heures     = np.arange(8760)
    jour_annee = heures // 24
    heure_jour = heures % 24

    # ── Profil PV synthétique ─────────────────────────────────────────────────
    declinaison   = 23.45 * np.sin(2 * np.pi * (jour_annee - 81) / 365)
    heure_solaire = heure_jour - 12
    angle_zenit   = np.cos(np.radians(declinaison)) * np.cos(np.radians(15 * heure_solaire))
    angle_zenit   = np.clip(angle_zenit, 0, 1)
    saison        = 1.0 + 0.15 * np.sin(2 * np.pi * (jour_annee - 172) / 365)
    CF_PV_raw     = angle_zenit * saison * cal['CF_sol'] * 2.5
    bruit_pv      = np.random.beta(8, 2, 8760) * 0.15
    CF_PV         = np.clip(CF_PV_raw * (1 - bruit_pv), 0, 0.95)
    CF_PV         = CF_PV * (cal['CF_sol'] / (CF_PV.mean() + 1e-9))
    CF_PV         = np.clip(CF_PV, 0, 0.95)

    # ── Profil éolien synthétique (Weibull) ───────────────────────────────────
    k_weib  = 2.0
    c_weib  = cal['v_mean'] / (np.sqrt(np.pi) / 2)
    u_weib  = np.random.uniform(0, 1, 8760)
    v_hour  = c_weib * (-np.log(1 - u_weib + 1e-9)) ** (1 / k_weib)
    saison_eol = 1.0 + 0.20 * np.cos(2 * np.pi * (jour_annee - 30) / 365)
    v_hour  = v_hour * saison_eol

    def power_curve(v):
        V_ci, V_r, V_o = 3.0, 12.0, 25.0
        cf = np.zeros_like(v)
        mask_mid = (v >= V_ci) & (v < V_r)
        cf[mask_mid] = np.clip(
            (-0.6994*v[mask_mid]**3 + 19.481*v[mask_mid]**2
             - 90.983*v[mask_mid] + 121) / 2000, 0, 1)
        cf[(v >= V_r) & (v < V_o)] = 1.0
        return cf

    CF_EOL_raw = power_curve(v_hour) * 0.85
    CF_EOL     = CF_EOL_raw * (cal['CF_eol'] / (CF_EOL_raw.mean() + 1e-9))
    CF_EOL     = np.clip(CF_EOL, 0, 1)

    df = pd.DataFrame({'CF_PV_h': CF_PV, 'CF_eol_h': CF_EOL}, index=idx)
    df.index.name = 'datetime_UTC'

    err_pv  = abs(df['CF_PV_h'].mean()  - cal['CF_sol']) / cal['CF_sol'] * 100
    err_eol = abs(df['CF_eol_h'].mean() - cal['CF_eol']) / cal['CF_eol'] * 100
    print(f"    → Synthétique : CF_PV={df['CF_PV_h'].mean()*100:.1f}% (err={err_pv:.1f}%) | "
          f"CF_eol={df['CF_eol_h'].mean()*100:.1f}% (err={err_eol:.1f}%)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — SIMULATION PYPSA (avec/sans batterie)
# ══════════════════════════════════════════════════════════════════════════════

# ✅ NOUVEAU : Cache simulation (clé = hash des paramètres)
#   Évite de recalculer la même configuration 2 fois pendant NSGA-II
_SIM_CACHE = {}
 
 
def _cache_key(PV_MW, EOL_MW, ELEC_MW, BAT_MWH, technologie, region):
    """Génère une clé de cache unique (hash MD5 tronqué)."""
    s = f"{PV_MW:.1f}_{EOL_MW:.1f}_{ELEC_MW:.1f}_{BAT_MWH:.1f}_{technologie}_{region}"
    return hashlib.md5(s.encode()).hexdigest()[:12]
 
 
def clear_cache():
    """Vide le cache de simulation."""
    global _SIM_CACHE
    _SIM_CACHE = {}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# FACTEUR DE DÉGRADATION — CORRECTION #3
# ══════════════════════════════════════════════════════════════════════════════
 
def _facteur_degradation(technologie):
    """
    Facteur de dégradation moyen sur la durée de vie de l'électrolyseur.

    PARAMS_TECHNO contient DEGRAD_PEM/AEL en fraction/an (ex: 0.015 = 1.5%/an).
    La dégradation réduit l'efficacité au fil du temps :
        eff(t) = eff_initiale × (1 - deg_annuel)^t
    Moyenne sur la durée de vie :
        facteur = LT / Σ(1-d)^t  pour t=0..LT-1

    Exemple PEM : deg=1.5%/an, durée=20 ans → facteur ≈ 1.16 (perte ~14%)

    Source : Hydrogen Europe 2024, NREL H2A 2024
    """
    deg_annuel = PARAMS_TECHNO.get(f'DEGRAD_{technologie}', 0.015)
    duree_vie  = PARAMS_TECHNO.get(f'DUREE_VIE_{technologie}',
                                    PARAMS_FIN.get('LT_ELEC', 20))

    # Σ(1 - d)^t pour t = 0..LT-1
    prod_cumulee = sum((1 - deg_annuel)**t for t in range(duree_vie))
    facteur = duree_vie / max(prod_cumulee, 1e-9)

    return facteur
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION PYPSA — FONCTION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
 
def simuler_pypsa(profils_df, PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                  technologie='PEM', region='Dakhla', verbose=False,
                  use_cache=True):
    """
    Simule le système ENR + Électrolyseur ± Batterie avec PyPSA.
 
    CORRECTIONS v3 :
        1. Électrolyseur flexible : charge variable [minload, 1.0] × Pnom
           Modélisé comme Load + Generator négatif (Slack absorbe le surplus)
        2. Curtailment = capacité ENR disponible - ENR dispatchée
        3. Dégradation appliquée à la production H2
        4. H2 vert = uniquement énergie ENR (hors Slack)
        5. Cache sans objet réseau (sérialisation)
 
    Architecture réseau v3 :
        BUS_AC ─── PV_generator    (production solaire, p_max_pu = CF horaire)
               ─── EOL_generator   (production éolienne, p_max_pu = CF horaire)
               ─── [BAT_store]     (batterie — uniquement si BAT_MWH > 0)
               ─── ELEC_load       (charge électrolyseur, p_set variable)
               ─── Curtail_gen     (générateur fictif coût=0 pour absorber surplus)
               ─── SLACK_gen       (génératrice coûteuse pour déficit)
 
    Paramètres
    ----------
    profils_df : DataFrame avec index DatetimeIndex, colonnes 'CF_PV_h', 'CF_eol_h'
    PV_MW      : float — puissance PV installée (MW)
    EOL_MW     : float — puissance éolienne installée (MW)
    ELEC_MW    : float — puissance électrolyseur (MW)
    BAT_MWH    : float — capacité batterie (MWh), 0 = pas de batterie
    technologie: str — 'PEM', 'AEL', ou 'SOEC'
    region     : str — nom de la région (pour le cache)
    verbose    : bool — afficher les messages de debug
    use_cache  : bool — utiliser le cache de simulation
 
    Retourne
    --------
    dict avec clés : H2_prod_kg_an, H2_vert_kg_an, E_fournie_MWh,
                     E_curtail_MWh, fiabilite, taux_curtailment, etc.
    """
    # ── Vérification cache ────────────────────────────────────────────────────
    if use_cache:
        key = _cache_key(PV_MW, EOL_MW, ELEC_MW, BAT_MWH, technologie, region)
        if key in _SIM_CACHE:
            return _SIM_CACHE[key]
 
    # ── Fallback si PyPSA non installé ────────────────────────────────────────
    if not PYPSA_OK:
        result = _simuler_simplifie(profils_df, PV_MW, EOL_MW, ELEC_MW,
                                    BAT_MWH, technologie, region)
        if use_cache:
            _SIM_CACHE[key] = result
        return result
 
    # ── Paramètres technologiques ─────────────────────────────────────────────
    eff        = PARAMS_TECHNO[f'EFF_{technologie}']
    minload    = PARAMS_TECHNO[f'MINLOAD_{technologie}']
    P_elec_kW  = ELEC_MW * 1e3
    P_elec_min = P_elec_kW * minload
    deg_factor = _facteur_degradation(technologie)
 
    try:
        n = pypsa.Network()
        n.set_snapshots(profils_df.index)
        idx_ = profils_df.index
 
        # ── Bus AC ────────────────────────────────────────────────────────────
        n.add("Bus", "bus_ac", carrier="AC")
 
        # ── Générateurs ENR ───────────────────────────────────────────────────
        if PV_MW > 0:
            n.add("Generator", "PV",
                  bus="bus_ac",
                  p_nom=PV_MW * 1e3,       # kW
                  p_max_pu=profils_df['CF_PV_h'].values,
                  marginal_cost=0.001,      # > 0 pour stabilité LP
                  carrier="solar")
 
        if EOL_MW > 0:
            n.add("Generator", "Eolien",
                  bus="bus_ac",
                  p_nom=EOL_MW * 1e3,
                  p_max_pu=profils_df['CF_eol_h'].values,
                  marginal_cost=0.001,
                  carrier="wind")
 
        # ── Batterie (uniquement si BAT_MWH > 0) ─────────────────────────────
        if BAT_MWH > 0:
            bat_power_kW = BAT_MWH / 4 * 1e3   # C-rate C/4
            n.add("StorageUnit", "Batterie",
                  bus="bus_ac",
                  p_nom=bat_power_kW,
                  max_hours=4,
                  efficiency_store=PARAMS_TECHNO['EFF_BAT_CHG'],
                  efficiency_dispatch=PARAMS_TECHNO['EFF_BAT_DCH'],
                  state_of_charge_initial=0.5,
                  cyclic_state_of_charge=True,
                  marginal_cost=0.002,
                  carrier="battery")
 
        # ──  Électrolyseur flexible ───────────────────────────
        # On modélise l'électrolyseur comme un Load avec p_set variable.
        # Le solveur LP détermine la charge optimale heure par heure.
        # Pour respecter le min_load, on ajoute un Generator négatif
        # ("Curtail") qui absorbe le surplus non consommable.
        #
        # Logique : la demande est fixée à P_elec_kW (max),
        # le Curtail_gen (coût 0) peut "annuler" une partie de la demande
        # quand l'ENR est insuffisante mais > min_load.
        # Le Slack (coût 100) n'intervient que si ENR < min_load.
 
        n.add("Load", "Electrolyseur",
              bus="bus_ac",
              p_set=np.full(len(profils_df), P_elec_kW))
 
        # Générateur d'ajustement (permet de réduire la charge effective)
        # p_nom = P_elec_kW × (1 - minload) = marge de flexibilité
        n.add("Generator", "Curtail_load",
              bus="bus_ac",
              p_nom=P_elec_kW * (1.0 - minload),
              marginal_cost=0.01,     # Faible coût mais > ENR
              carrier="curtail_load")
 
        # ── Slack (déficit résiduel) ──────────────────────────────────────────
        n.add("Generator", "Slack",
              bus="bus_ac",
              p_nom=1e6,
              marginal_cost=100.0,
              carrier="slack")
 
        # ── Résolution LP ─────────────────────────────────────────────────────
        n.optimize(solver_name='highs', solver_options={'output_flag': False})
 
        # ── Extraction résultats ──────────────────────────────────────────────
        gen_pv    = n.generators_t.p.get('PV',           pd.Series(0.0, index=idx_))
        gen_eol   = n.generators_t.p.get('Eolien',       pd.Series(0.0, index=idx_))
        gen_slack = n.generators_t.p.get('Slack',         pd.Series(0.0, index=idx_))
        gen_curt  = n.generators_t.p.get('Curtail_load',  pd.Series(0.0, index=idx_))
        bat_dch   = (n.storage_units_t.p.get('Batterie', pd.Series(0.0, index=idx_))
                     if BAT_MWH > 0 else pd.Series(0.0, index=idx_))
 
        # ── Charge effective de l'électrolyseur ───────────────────────────────
        # Charge = Demande_max - Curtail_load (ce qui est réellement consommé)
        P_elec_effective = P_elec_kW - gen_curt.values
 
        # ──  Séparation H2 vert et H2 total ──────────────────
        # H2 vert = uniquement l'énergie ENR (+ batterie) fournie
        # H2 total = inclut le Slack (mais pénalisé dans le LCOH)
        P_enr = gen_pv + gen_eol + bat_dch
        E_fournie_verte = np.minimum(P_enr.values, P_elec_effective).sum()
        E_fournie_slack = gen_slack.sum()
        E_fournie_totale = E_fournie_verte + E_fournie_slack
 
        # ── Fiabilité = heures où ENR couvre ≥ 95% de la charge effective ────
        charge_eff_arr = np.maximum(P_elec_effective, P_elec_min)
        heures_ok = (P_enr.values >= charge_eff_arr * 0.95).sum()
        fiabilite = heures_ok / 8760
 
        
        # Capacité ENR disponible (ce qui POURRAIT être produit)
        cap_pv  = PV_MW * 1e3 * profils_df['CF_PV_h'].values if PV_MW > 0 else np.zeros(8760)
        cap_eol = EOL_MW * 1e3 * profils_df['CF_eol_h'].values if EOL_MW > 0 else np.zeros(8760)
        E_enr_disponible = (cap_pv + cap_eol).sum()
 
        # ENR réellement dispatchée
        E_enr_dispatchee = (gen_pv + gen_eol).sum()
 
        # Curtailment = disponible - dispatché
        E_curtail = max(0.0, E_enr_disponible - E_enr_dispatchee)
        taux_curtail = E_curtail / max(E_enr_disponible, 1)
 
        # ── Dégradation appliquée ────────────────────────────
        H2_vert_kg  = E_fournie_verte / (eff * deg_factor)
        H2_total_kg = E_fournie_totale / (eff * deg_factor)
 
        # ── Indicateurs supplémentaires ───────────────────────────────────────
        H_full_load = int((P_enr.values >= charge_eff_arr * 0.95).sum())
 
        result = {
            'H2_prod_kg_an'     : float(H2_total_kg),
            'H2_vert_kg_an'     : float(H2_vert_kg),
            'E_fournie_MWh'     : float(E_fournie_totale / 1000),
            'E_fournie_verte_MWh': float(E_fournie_verte / 1000),
            'E_curtail_MWh'     : float(E_curtail / 1000),
            'E_enr_dispo_MWh'   : float(E_enr_disponible / 1000),
            'E_enr_dispatch_MWh': float(E_enr_dispatchee / 1000),
            'fiabilite'         : float(fiabilite),
            'taux_curtailment'  : float(taux_curtail),
            'heures_full_load'  : H_full_load,
            'slack_MWh'         : float(E_fournie_slack / 1000),
            'pct_slack'         : float(E_fournie_slack / max(E_fournie_totale, 1) * 100),
            'facteur_degradation': float(deg_factor),
            'statut'            : 'PyPSA_LP',
            'avec_batterie'     : BAT_MWH > 0,
            'network'           : n,
        }
 
        if use_cache:
            # Ne pas mettre l'objet réseau en cache (trop lourd)
            result_no_net = {k: v for k, v in result.items() if k != 'network'}
            _SIM_CACHE[key] = result_no_net
 
        return result
 
    except Exception as e:
        if verbose:
            print(f"  ⚠️  PyPSA erreur : {e} — fallback simulation simplifiée")
        result = _simuler_simplifie(profils_df, PV_MW, EOL_MW, ELEC_MW,
                                    BAT_MWH, technologie, region)
        if use_cache:
            _SIM_CACHE[key] = result
        return result
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION SIMPLIFIÉE — Fallback sans PyPSA
# ══════════════════════════════════════════════════════════════════════════════
 
def _simuler_simplifie(profils_df, PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                        technologie='PEM', region='Dakhla'):
    """
    Simulation simplifiée (bilan énergétique horaire) — fallback sans PyPSA.
 
    CORRECTIONS v3 :
        1. Électrolyseur flexible : charge réduite si ENR > min_load mais < max
        2. Curtailment : calculé AVANT modification du SOC
        3. Dégradation : facteur moyen appliqué
        4. H2 vert : Slack exclu
        5. min_load : en dessous du seuil, l'électrolyseur s'arrête (P=0)
 
    Algorithme horaire (h = 0..8759) :
    ───────────────────────────────────────────────────────────────────────
    P_disponible = P_enr[h] + décharge_batterie (si disponible)
 
    SI P_disponible >= P_elec_max :
        → électrolyseur à pleine charge
        → surplus = P_disponible - P_elec_max → charge batterie → curtailment
    SINON SI P_disponible >= P_elec_min :
        → électrolyseur à P_disponible (charge partielle)
        → pas de curtailment, pas de slack
    SINON :
        → électrolyseur arrêté (en dessous du min_load)
        → toute l'ENR → charge batterie → curtailment
    ───────────────────────────────────────────────────────────────────────
    """
    eff        = PARAMS_TECHNO[f'EFF_{technologie}']
    minload    = PARAMS_TECHNO[f'MINLOAD_{technologie}']
    deg_factor = _facteur_degradation(technologie)
    P_elec_kW  = ELEC_MW * 1e3
    P_elec_min = P_elec_kW * minload
    BAT_kWh    = BAT_MWH * 1e3
 
    CF_PV  = profils_df['CF_PV_h'].values
    CF_EOL = profils_df['CF_eol_h'].values
    P_PV   = CF_PV  * PV_MW  * 1e3    # kW
    P_EOL  = CF_EOL * EOL_MW * 1e3    # kW
    P_enr  = P_PV + P_EOL             # kW
 
    # ── Paramètres batterie ───────────────────────────────────────────────────
    has_bat = BAT_MWH > 0
    SOC     = BAT_kWh * 0.5 if has_bat else 0.0
    SOC_min = BAT_kWh * PARAMS_TECHNO['SOC_MIN'] if has_bat else 0.0
    SOC_max = BAT_kWh * PARAMS_TECHNO['SOC_MAX'] if has_bat else 0.0
    eta_c   = PARAMS_TECHNO['EFF_BAT_CHG']
    eta_d   = PARAMS_TECHNO['EFF_BAT_DCH']
 
    # ── Accumulateurs ─────────────────────────────────────────────────────────
    E_fournie_verte = 0.0
    E_curtail_total = 0.0
    E_enr_dispo     = float(P_enr.sum())
    heures_ok       = 0
 
    for h in range(8760):
        p_enr_h = P_enr[h]
 
        # Décharge batterie si déficit partiel
        p_bat_dch = 0.0
        if has_bat and p_enr_h < P_elec_kW:
            deficit = P_elec_kW - p_enr_h
            dispo_bat = (SOC - SOC_min) * eta_d
            p_bat_dch = min(deficit, dispo_bat)
 
        P_dispo = p_enr_h + p_bat_dch
 
        # ── CAS 1 : Pleine charge ────────────────────────────────────────────
        if P_dispo >= P_elec_kW:
            E_fournie_verte += P_elec_kW
            heures_ok += 1
 
            # Décharge batterie effectivement utilisée
            if p_bat_dch > 0:
                SOC -= p_bat_dch / eta_d
 
            # Surplus → charge batterie → curtailment
            surplus = P_dispo - P_elec_kW - p_bat_dch  # surplus ENR pur
            surplus = p_enr_h - P_elec_kW  # ENR excédentaire (on n'a pas utilisé batterie)
            if surplus > 0:
                #  curtailment calculé AVANT modification SOC
                if has_bat:
                    charge_possible = min(surplus * eta_c, SOC_max - SOC)
                    curtail_h = surplus - charge_possible / eta_c
                    SOC += charge_possible
                else:
                    curtail_h = surplus
                E_curtail_total += max(0, curtail_h)
 
        # ── CAS 2 : Charge partielle (≥ min_load) ────────────────────────────
        elif P_dispo >= P_elec_min:
            E_fournie_verte += P_dispo
            heures_ok += 1  # heure productive (charge partielle)
 
            # Décharge batterie effectivement utilisée
            if p_bat_dch > 0:
                SOC -= p_bat_dch / eta_d
 
        # ── CAS 3 : Arrêt électrolyseur (< min_load) ─────────────────────────
        else:
            # Toute l'ENR va en batterie ou est curtailée
            if has_bat:
                charge_possible = min(p_enr_h * eta_c, SOC_max - SOC)
                curtail_h = p_enr_h - charge_possible / eta_c
                SOC += charge_possible
            else:
                curtail_h = p_enr_h
            E_curtail_total += max(0, curtail_h)
 
    # ── Dégradation appliquée ────────────────────────────────
    H2_vert_kg = E_fournie_verte / (eff * deg_factor)
    fiabilite  = heures_ok / 8760
 
    taux_curtail = E_curtail_total / max(E_enr_dispo, 1)
 
    return {
        'H2_prod_kg_an'      : float(H2_vert_kg),   # Pas de Slack en mode simplifié
        'H2_vert_kg_an'      : float(H2_vert_kg),
        'E_fournie_MWh'      : float(E_fournie_verte / 1000),
        'E_fournie_verte_MWh': float(E_fournie_verte / 1000),
        'E_curtail_MWh'      : float(E_curtail_total / 1000),
        'E_enr_dispo_MWh'    : float(E_enr_dispo / 1000),
        'E_enr_dispatch_MWh' : float((E_enr_dispo - E_curtail_total) / 1000),
        'fiabilite'          : float(fiabilite),
        'taux_curtailment'   : float(taux_curtail),
        'heures_full_load'   : int(heures_ok),
        'slack_MWh'          : 0.0,
        'pct_slack'          : 0.0,
        'facteur_degradation': float(deg_factor),
        'statut'             : 'Simplifie',
        'avec_batterie'      : has_bat,
        'network'            : None,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TESTS UNITAIRES
# ══════════════════════════════════════════════════════════════════════════════
 
def _test_module2():
    """Tests unitaires du module 2 — appelé depuis le pipeline principal."""
    print("=" * 70)
    print("  TEST MODULE 2 — Simulation PyPSA v3 (corrigé)")
    print("=" * 70)
 
    # Créer des profils synthétiques pour test (1 semaine)
    idx = pd.date_range("2024-01-01", periods=8760, freq="h", tz="UTC")
    h = np.arange(8760)
 
    # PV : profil sinusoïdal jour/nuit
    cf_pv = np.maximum(0, np.sin((h % 24 - 6) / 12 * np.pi) * 0.8)
    cf_pv *= (1 + 0.2 * np.sin(2 * np.pi * h / 8760))  # saisonnalité
 
    # Éolien : processus AR1
    np.random.seed(42)
    cf_eol = np.zeros(8760)
    cf_eol[0] = 0.3
    for i in range(1, 8760):
        cf_eol[i] = 0.7 * cf_eol[i-1] + 0.3 * np.random.uniform(0, 0.8)
    cf_eol = np.clip(cf_eol, 0, 1)
 
    profils = pd.DataFrame({
        'CF_PV_h': cf_pv,
        'CF_eol_h': cf_eol,
    }, index=idx)
 
    # Test 1 : facteur dégradation
    print("\n--- Test dégradation ---")
    for tech in ['PEM', 'AEL', 'SOEC']:
        f = _facteur_degradation(tech)
        perte = (f - 1) * 100
        print(f"  {tech}: facteur={f:.4f} (perte moyenne ≈ {perte:.1f}%)")
 
    # Test 2 : simulation simplifiée sans batterie
    print("\n--- Test simplifié SANS batterie ---")
    r1 = _simuler_simplifie(profils, PV_MW=100, EOL_MW=50, ELEC_MW=80,
                             BAT_MWH=0, technologie='PEM')
    print(f"  H2 vert : {r1['H2_vert_kg_an']:,.0f} kg/an")
    print(f"  Fiabilité : {r1['fiabilite']*100:.1f}%")
    print(f"  Curtailment : {r1['taux_curtailment']*100:.1f}%")
    print(f"  Dégradation : facteur={r1['facteur_degradation']:.4f}")
 
    # Test 3 : simulation simplifiée AVEC batterie
    print("\n--- Test simplifié AVEC batterie (50 MWh) ---")
    r2 = _simuler_simplifie(profils, PV_MW=100, EOL_MW=50, ELEC_MW=80,
                             BAT_MWH=50, technologie='PEM')
    print(f"  H2 vert : {r2['H2_vert_kg_an']:,.0f} kg/an")
    print(f"  Fiabilité : {r2['fiabilite']*100:.1f}%")
    print(f"  Curtailment : {r2['taux_curtailment']*100:.1f}%")
    print(f"  Gain H2 vs sans batterie : {(r2['H2_vert_kg_an']/r1['H2_vert_kg_an']-1)*100:+.1f}%")
 
    # Test 4 : simulation PyPSA (si disponible)
    if PYPSA_OK:
        print("\n--- Test PyPSA LP ---")
        r3 = simuler_pypsa(profils, PV_MW=100, EOL_MW=50, ELEC_MW=80,
                           BAT_MWH=0, technologie='PEM', use_cache=False)
        print(f"  Statut : {r3['statut']}")
        print(f"  H2 total : {r3['H2_prod_kg_an']:,.0f} kg/an")
        print(f"  H2 vert  : {r3['H2_vert_kg_an']:,.0f} kg/an")
        print(f"  Slack    : {r3['slack_MWh']:.1f} MWh ({r3['pct_slack']:.1f}%)")
        print(f"  Curtail  : {r3['taux_curtailment']*100:.1f}%")
        print(f"  Fiabilité: {r3['fiabilite']*100:.1f}%")
    else:
        print("\n  ⚠️  PyPSA non installé — test LP ignoré")
 
    # Test 5 : cache
    print("\n--- Test cache ---")
    clear_cache()
    r_a = simuler_pypsa(profils, 100, 50, 80, 0, 'PEM', 'Dakhla', use_cache=True)
    r_b = simuler_pypsa(profils, 100, 50, 80, 0, 'PEM', 'Dakhla', use_cache=True)
    print(f"  Cache : {len(_SIM_CACHE)} entrées")
    print(f"  Résultats identiques : {r_a['H2_vert_kg_an'] == r_b['H2_vert_kg_an']}")
 
    print("\n✅ Tests terminés")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CALCUL LCOH (décomposé par composante)
# ══════════════════════════════════════════════════════════════════════════════

def calculer_LCOH(PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                  H2_prod_kg_an, technologie='PEM',
                  detail=False):
    """
    Calcule le LCOH ($/kgH2) avec décomposition par composante.

    
        - Retourne le détail par composante si detail=True
        - Intègre la dégradation PV sur la durée de vie (facteur moyen)
        - Formule LCOH conforme IRENA (2020) + IEA (2024)

    LCOH = (CAPEX_ann_tot + OPEX_ann_tot + C_eau) / H2_prod_an_moy
    """
    DR = PARAMS_FIN['DR']

    def CRF(LT):
        return (DR * (1 + DR)**LT) / ((1 + DR)**LT - 1)

    # ── CAPEX total ($) ───────────────────────────────────────────────────────
    CAPEX_PV   = PV_MW   * 1e3 * PARAMS_TECHNO['CAPEX_PV']
    CAPEX_EOL  = EOL_MW  * 1e3 * PARAMS_TECHNO['CAPEX_EOL']
    CAPEX_ELEC = ELEC_MW * 1e3 * PARAMS_TECHNO[f'CAPEX_{technologie}']
    CAPEX_BAT  = BAT_MWH * 1e3 * PARAMS_TECHNO['CAPEX_BAT']

    # facteur dégradation PV moyen sur durée de vie
    # Production moyenne = P0 * (1 - somme(degrad^t)) / LT
    d_pv  = PARAMS_TECHNO['DEGRAD_PV']
    LT_PV = PARAMS_FIN['LT_PV']
    fact_degrad_pv = sum((1 - d_pv)**t for t in range(LT_PV)) / LT_PV  # ~0.889

    # ── Annuités CAPEX ($/an) ─────────────────────────────────────────────────
    ann_PV   = CAPEX_PV   * CRF(PARAMS_FIN['LT_PV'])
    ann_EOL  = CAPEX_EOL  * CRF(PARAMS_FIN['LT_EOL'])
    ann_ELEC = CAPEX_ELEC * CRF(PARAMS_FIN['LT_ELEC'])
    ann_BAT  = CAPEX_BAT  * CRF(PARAMS_FIN['LT_BAT'])
    CAPEX_ann = ann_PV + ann_EOL + ann_ELEC + ann_BAT

    # ── OPEX annuel ($/an) ────────────────────────────────────────────────────
    opex_PV   = PV_MW   * 1e3 * PARAMS_TECHNO['OPEX_PV']
    opex_EOL  = EOL_MW  * 1e3 * PARAMS_TECHNO['OPEX_EOL']
    opex_ELEC = CAPEX_ELEC   * PARAMS_TECHNO[f'OPEX_{technologie}']
    opex_BAT  = CAPEX_BAT    * PARAMS_TECHNO['OPEX_BAT']
    OPEX_ann  = opex_PV + opex_EOL + opex_ELEC + opex_BAT

    # ── Coût eau ($/an) ───────────────────────────────────────────────────────
    C_eau = (PARAMS_TECHNO['WATER_CONS'] / 1000
             * PARAMS_TECHNO['WATER_COST']
             * H2_prod_kg_an)

    # ── WaterModel optionnel (nécessite base PostgreSQL) ─────────────────
    _db_available = False
    try:
        _db = _charger_db()
        _db_available = True
    except Exception:
        pass  # Pas de DB → utilise PARAMS_TECHNO['WATER_COST'] directement

    # Parametres Halloran (Oxford 2023)
    _H = {
        "eau_douce"   : {"C_spec":0.30, "C_transp":0.05, "C_elec":0.50},
        "dessalement" : {"C_spec":0.65, "C_transp":0.05, "C_elec":3.50},
    }
    # Distances eau (km) — completent T1 qui ne contient pas ces colonnes
    _DIST = {
        "ouarzazate" : {"d_douce":15,  "d_ocean":220},
        "laayoune"   : {"d_douce":120, "d_ocean":5},
        "dakhla"     : {"d_douce":180, "d_ocean":2},
        "tanger"     : {"d_douce":10,  "d_ocean":10},
        "jorf lasfar": {"d_douce":20,  "d_ocean":20},
        "guelmim"    : {"d_douce":80,  "d_ocean":30},
    }
    _L_KGH2 = 9.0   # L/kgH2 (IEA 2024)


    class WaterModel:
        """
        Calcule le cout eau optimal en EUR/kgH2 depuis T1.

        Usage
        -----
        wm  = WaterModel()
        res = wm.run("Dakhla")
        df  = wm.tableau_regions()
        """

        def __init__(self):
            self.T1 = _db.build_T1_ressources()

        def _t1(self, region):
            mask = self.T1["region"].str.lower() == region.strip().lower()
            if not mask.any():
                raise KeyError(f"Region '{region}' introuvable.")
            return self.T1.loc[mask].iloc[0]

        def _dist(self, region):
            return _DIST.get(region.strip().lower(), {"d_douce":50, "d_ocean":50})

        def _cout_m3(self, source, dist_km, P_elec):
            p = _H[source]
            return p["C_spec"] + p["C_transp"]/100.0*dist_km + p["C_elec"]*P_elec

        def run(self, region, P_elec_EUR_kWh=None):
            """
            Calcule le cout eau optimal (EUR/kgH2).

            Parametres
            ----------
            region          : nom region T1  ex. "Dakhla"
            P_elec_EUR_kWh  : prix electricite locale. Si None -> PPA hybride T1.
            """
            r1 = self._t1(region)
            if P_elec_EUR_kWh is None:
                P_elec_EUR_kWh = float(r1["PPA_local_USD_kWh"]) * self._USD_EUR

            d = self._dist(region)
            c_douce = self._cout_m3("eau_douce",   d["d_douce"], P_elec_EUR_kWh) * _L_KGH2/1000
            c_desal = self._cout_m3("dessalement", d["d_ocean"], P_elec_EUR_kWh) * _L_KGH2/1000

            # Coherence T1 : dessalement_requis -> forcer dessalement
            desal_requis = bool(r1.get("dessalement_requis", True))
            if desal_requis:
                source, cout_opt = "Dessalement", c_desal
            else:
                source, cout_opt = ("Eau douce", c_douce) if c_douce <= c_desal else ("Dessalement", c_desal)

            return {
                "region"                  : region,
                "cout_eau_EUR_kgH2"       : round(cout_opt, 6),
                "source_optimale"         : source,
                "cout_eau_douce_EUR_kgH2" : round(c_douce,  6),
                "cout_desal_EUR_kgH2"     : round(c_desal,  6),
                "dist_eau_douce_km"       : d["d_douce"],
                "dist_ocean_km"           : d["d_ocean"],
                "demande_L_kgH2"          : _L_KGH2,
                "P_elec_EUR_kWh"          : round(P_elec_EUR_kWh, 5),
            }

        def tableau_regions(self):
            """Tableau comparatif 6 regions, trie par cout croissant."""
            rows = []
            for reg in self.T1["region"]:
                try:
                    r = self.run(reg)
                    rows.append({"Region": reg,
                        "Source optimale": r["source_optimale"],
                        "Cout [EUR/kgH2]": r["cout_eau_EUR_kgH2"],
                        "Eau douce [EUR/kgH2]": r["cout_eau_douce_EUR_kgH2"],
                        "Dessalement [EUR/kgH2]": r["cout_desal_EUR_kgH2"],
                        "Dist. douce [km]": r["dist_eau_douce_km"],
                        "Dist. ocean [km]": r["dist_ocean_km"]})
                except Exception as e:
                    print(f"  Attention {reg} : {e}")
            return pd.DataFrame(rows).sort_values("Cout [EUR/kgH2]").reset_index(drop=True)


    # ── LCOH ─────────────────────────────────────────────────────────────────
    if H2_prod_kg_an < 1:
        if detail:
            return 99.0, {}
        return 99.0

    # Production H2 utilisée pour le LCOH
    # NOTE : la dégradation électrolyseur est déjà appliquée dans simuler_pypsa.
    # La dégradation PV affecte le CAPEX annualisé (via fact_degrad_pv sur la
    # production ENR), pas la production H2 directement.
    # On applique fact_degrad_pv seulement sur les annuités PV pour refléter
    # la baisse de production ENR au fil des ans.
    ann_PV_adj = ann_PV / max(fact_degrad_pv, 0.5)  # Annuité PV corrigée
    CAPEX_ann_adj = ann_PV_adj + ann_EOL + ann_ELEC + ann_BAT

    LCOH = (CAPEX_ann_adj + OPEX_ann + C_eau) / max(H2_prod_kg_an, 1)
    LCOH = float(np.clip(LCOH, 0.5, 50.0))

    if detail:
        # Décomposition LCOH par composante ($/kgH2)
        detail_dict = {
            'LCOH_PV_capex'   : ann_PV_adj / max(H2_prod_kg_an, 1),
            'LCOH_EOL_capex'  : ann_EOL  / max(H2_prod_kg_an, 1),
            'LCOH_ELEC_capex' : ann_ELEC / max(H2_prod_kg_an, 1),
            'LCOH_BAT_capex'  : ann_BAT  / max(H2_prod_kg_an, 1),
            'LCOH_PV_opex'    : opex_PV   / max(H2_prod_kg_an, 1),
            'LCOH_EOL_opex'   : opex_EOL  / max(H2_prod_kg_an, 1),
            'LCOH_ELEC_opex'  : opex_ELEC / max(H2_prod_kg_an, 1),
            'LCOH_BAT_opex'   : opex_BAT  / max(H2_prod_kg_an, 1),
            'LCOH_eau'        : C_eau     / max(H2_prod_kg_an, 1),
            'LCOH_total'      : LCOH,
        }
        return LCOH, detail_dict

    return LCOH


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — ALGORITHME NSGA-II (avec paramètre scenario_batterie)
# ══════════════════════════════════════════════════════════════════════════════

class NSGAII:
    """
    Implémentation NSGA-II pour l'optimisation bi-objectif du système H2.

    ✅ NOUVEAU (v2) :
        scenario_batterie : 'avec' | 'sans'
            - 'avec' : BAT_MWH ∈ [10, 500] MWh — optimisé
            - 'sans' : BAT_MWH = 0 forcé — comparaison sans stockage

    Problème : min f1(x) = LCOH, max f2(x) = Fiabilité
    Variables : x = [PV_MW, EOL_MW, ELEC_MW, BAT_MWH]
    """

    def __init__(self, profils_df, technologie='PEM', region='Dakhla',
                 pop_size=60, n_gen=40, seed=42,
                 scenario_batterie='avec'):     
        self.profils_df         = profils_df
        self.technologie        = technologie
        self.region             = region
        self.pop_size           = pop_size
        self.n_gen              = n_gen
        self.scenario_batterie  = scenario_batterie
        self.n_var              = 4
        np.random.seed(seed)

        # Bornes selon scénario batterie
        bounds = (BOUNDS_AVEC_BAT if scenario_batterie == 'avec'
                  else BOUNDS_SANS_BAT)
        self.lb = np.array([bounds['PV_MW'][0],   bounds['EOL_MW'][0],
                            bounds['ELEC_MW'][0],  bounds['BAT_MWH'][0]])
        self.ub = np.array([bounds['PV_MW'][1],   bounds['EOL_MW'][1],
                            bounds['ELEC_MW'][1],  bounds['BAT_MWH'][1]])

        self.historique          = []
        self.front_pareto_final  = None

    def _decode(self, x):
        """Décode individu normalisé → valeurs physiques."""
        vals = self.lb + x * (self.ub - self.lb)
        # ✅ Forcer BAT=0 explicitement pour scénario sans batterie
        if self.scenario_batterie == 'sans':
            vals[3] = 0.0
        return vals

    def evaluer(self, x):
        """Évalue les deux objectifs. Retourne (f1_LCOH, f2_neg_fiabilite)."""
        PV_MW, EOL_MW, ELEC_MW, BAT_MWH = self._decode(x)

        res = simuler_pypsa(self.profils_df, PV_MW, EOL_MW, ELEC_MW,
                            BAT_MWH, self.technologie, self.region,
                            use_cache=True)

        # ✅ FIX : utiliser H2_vert_kg_an (hors Slack) pour LCOH vert
        H2_pour_lcoh = res.get('H2_vert_kg_an', res['H2_prod_kg_an'])
        lcoh = calculer_LCOH(PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                             H2_pour_lcoh, self.technologie)

        fib = res['fiabilite']
        # Pénalités si production insuffisante
        if res['H2_prod_kg_an'] < DEMANDE_H2_KGH * 100:
            lcoh += 20.0
            fib  -= 0.5

        return float(lcoh), float(-fib)

    def _init_population(self):
        """Population initiale mixte (aléatoire + LHS + points de référence)."""
        pop = []
        n   = self.pop_size

        # Aléatoire uniforme (50%)
        n_rand = int(n * 0.5)
        pop.extend([np.random.uniform(0, 1, self.n_var) for _ in range(n_rand)])

        # Latin Hypercube Sampling (30%)
        n_lhs = int(n * 0.3)
        for i in range(n_lhs):
            x = np.array([(i + np.random.uniform()) / n_lhs
                          for _ in range(self.n_var)])
            np.random.shuffle(x)
            pop.append(x)

        # Points de référence (20%) — adaptés au scénario
        if self.scenario_batterie == 'avec':
            refs = [
                np.array([0.7, 0.2, 0.3, 0.3]),   # PV dominant + batterie
                np.array([0.6, 0.5, 0.6, 0.5]),   # Hybride + batterie forte
                np.array([0.3, 0.8, 0.5, 0.4]),   # Éolien dominant (Dakhla)
                np.array([0.5, 0.4, 0.5, 0.4]),   # Équilibré
            ]
        else:
            refs = [
                np.array([0.9, 0.3, 0.4, 0.0]),   # PV fort, sans batterie
                np.array([0.5, 0.9, 0.7, 0.0]),   # Éolien fort, sans batterie
                np.array([0.8, 0.5, 0.6, 0.0]),   # Hybride, sans batterie
                np.array([0.6, 0.4, 0.5, 0.0]),   # Équilibré, sans batterie
            ]

        while len(pop) < n:
            if refs:
                pop.append(refs.pop(0) + np.random.uniform(-0.05, 0.05, self.n_var))
            else:
                pop.append(np.random.uniform(0, 1, self.n_var))

        pop = [np.clip(x, 0, 1) for x in pop[:n]]
        # ✅ Forcer BAT=0 pour scénario sans batterie
        if self.scenario_batterie == 'sans':
            for x in pop:
                x[3] = 0.0
        return pop

    @staticmethod
    def _domine(f_a, f_b):
        return (all(a <= b for a, b in zip(f_a, f_b)) and
                any(a <  b for a, b in zip(f_a, f_b)))

    def _fast_non_dominated_sort(self, fitnesses):
        n = len(fitnesses)
        fronts     = [[]]
        rang       = [0] * n
        domine_par = [0] * n
        domine_set = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._domine(fitnesses[i], fitnesses[j]):
                    domine_set[i].append(j)
                elif self._domine(fitnesses[j], fitnesses[i]):
                    domine_par[i] += 1
            if domine_par[i] == 0:
                rang[i] = 0
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            suivant = []
            for i in fronts[k]:
                for j in domine_set[i]:
                    domine_par[j] -= 1
                    if domine_par[j] == 0:
                        rang[j] = k + 1
                        suivant.append(j)
            k += 1
            fronts.append(suivant)
        return fronts[:-1], rang

    def _crowding_distance(self, front, fitnesses):
        n_f = len(front)
        if n_f <= 2:
            return [float('inf')] * n_f
        dist  = [0.0] * n_f
        n_obj = len(fitnesses[0])
        for m in range(n_obj):
            vals_m = [(fitnesses[front[i]][m], i) for i in range(n_f)]
            vals_m.sort()
            dist[vals_m[0][1]]  = float('inf')
            dist[vals_m[-1][1]] = float('inf')
            f_min = vals_m[0][0]
            f_max = vals_m[-1][0]
            span  = f_max - f_min + 1e-12
            for k in range(1, n_f - 1):
                dist[vals_m[k][1]] += (vals_m[k+1][0] - vals_m[k-1][0]) / span
        return dist

    def _tournoi(self, rangs, crowding, k=2):
        candidats = np.random.choice(len(rangs), k, replace=False)
        best = candidats[0]
        for c in candidats[1:]:
            if rangs[c] < rangs[best]:
                best = c
            elif rangs[c] == rangs[best] and crowding[c] > crowding[best]:
                best = c
        return best

    def _sbx(self, p1, p2, eta_c=15.0, p_c=0.9):
        if np.random.random() > p_c:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.n_var):
            if self.scenario_batterie == 'sans' and i == 3:
                continue   # ✅ Ne pas modifier BAT dans scénario sans batterie
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) < 1e-10:
                    continue
                y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                u      = np.random.random()
                beta1  = 1 + 2 * y1 / (y2 - y1 + 1e-12)
                beta2  = 1 + 2 * (1 - y2) / (y2 - y1 + 1e-12)
                alpha1 = 2 - beta1**(-(eta_c + 1))
                alpha2 = 2 - beta2**(-(eta_c + 1))
                if u <= 1 / alpha1:
                    betaq = (u * alpha1) ** (1 / (eta_c + 1))
                else:
                    betaq = (1 / (2 - u * alpha1)) ** (1 / (eta_c + 1))
                c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

    def _mutation(self, x, eta_m=20.0):
        x   = x.copy()
        p_m = 1.0 / self.n_var
        for i in range(self.n_var):
            if self.scenario_batterie == 'sans' and i == 3:
                continue   # ✅ Pas de mutation sur BAT si scénario sans batterie
            if np.random.random() < p_m:
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                x[i] = np.clip(x[i] + delta, 0, 1)
        return x

    def optimiser(self, verbose=True):
        """Exécute NSGA-II et retourne le front de Pareto."""
        t_debut = time.time()
        label_bat = "AVEC batterie" if self.scenario_batterie == 'avec' else "SANS batterie"
        print(f"\n{'═'*65}")
        print(f"  NSGA-II — {self.region} | {self.technologie} | {label_bat}")
        print(f"  pop={self.pop_size} | gen={self.n_gen}")
        print(f"{'═'*65}")

        print("  [1/4] Initialisation population...")
        population = self._init_population()

        print(f"  [2/4] Évaluation génération 0 ({len(population)} individus)...")
        fitnesses = [self.evaluer(x) for x in population]

        print(f"  [3/4] Évolution ({self.n_gen} générations)...")
        for gen in range(self.n_gen):
            fronts, rangs = self._fast_non_dominated_sort(fitnesses)
            crowding_all  = [0.0] * len(population)
            for front in fronts:
                cd = self._crowding_distance(front, fitnesses)
                for i, idx_ in enumerate(front):
                    crowding_all[idx_] = cd[i]

            enfants = []
            fitness_e = []
            while len(enfants) < self.pop_size:
                i1 = self._tournoi(rangs, crowding_all)
                i2 = self._tournoi(rangs, crowding_all)
                c1, c2 = self._sbx(population[i1], population[i2])
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                enfants.append(c1)
                fitness_e.append(self.evaluer(c1))
                if len(enfants) < self.pop_size:
                    enfants.append(c2)
                    fitness_e.append(self.evaluer(c2))

            pop_combinee = population + enfants
            fit_combinee = fitnesses + fitness_e
            fronts_c, rangs_c = self._fast_non_dominated_sort(fit_combinee)

            population_new = []
            fitnesses_new  = []
            for front in fronts_c:
                if len(population_new) + len(front) <= self.pop_size:
                    for idx_ in front:
                        population_new.append(pop_combinee[idx_])
                        fitnesses_new.append(fit_combinee[idx_])
                else:
                    cd    = self._crowding_distance(front, fit_combinee)
                    ordre = sorted(range(len(front)), key=lambda k: -cd[k])
                    reste = self.pop_size - len(population_new)
                    for k in ordre[:reste]:
                        population_new.append(pop_combinee[front[k]])
                        fitnesses_new.append(fit_combinee[front[k]])
                    break

            population = population_new
            fitnesses  = fitnesses_new

            lcoh_min = min(f[0] for f in fitnesses)
            fib_max  = max(-f[1] for f in fitnesses)
            self.historique.append({
                'generation'        : gen + 1,
                'LCOH_min'          : round(lcoh_min, 3),
                'Fiab_max'          : round(fib_max, 3),
                'N_front1'          : len(fronts_c[0]) if fronts_c else 0,
                'scenario_batterie' : self.scenario_batterie,
            })

            if verbose and (gen + 1) % 5 == 0:
                print(f"    Gen {gen+1:3d}/{self.n_gen} | "
                      f"LCOH_min={lcoh_min:.2f} $/kg | "
                      f"Fiab_max={fib_max*100:.1f}% | "
                      f"Front1={len(fronts_c[0])} pts")

        print("  [4/4] Extraction front de Pareto final...")
        fronts_f, _ = self._fast_non_dominated_sort(fitnesses)
        front_pareto = []
        for idx_ in fronts_f[0]:
            x_dec = self._decode(population[idx_])
            lcoh  = fitnesses[idx_][0]
            fib   = -fitnesses[idx_][1]
            # ✅ NOUVEAU : calcul LCOH détaillé pour chaque solution Pareto
            res_sim = simuler_pypsa(
                self.profils_df, x_dec[0], x_dec[1], x_dec[2], x_dec[3],
                self.technologie, self.region, use_cache=True)
            lcoh_val, lcoh_detail = calculer_LCOH(
                x_dec[0], x_dec[1], x_dec[2], x_dec[3],
                res_sim['H2_prod_kg_an'], self.technologie, detail=True)

            front_pareto.append({
                'x'                : population[idx_].tolist(),
                'f'                : (lcoh, fib),
                'LCOH'             : round(lcoh, 3),
                'Fiabilite'        : round(fib, 3),
                'PV_MW'            : round(x_dec[0], 1),
                'EOL_MW'           : round(x_dec[1], 1),
                'ELEC_MW'          : round(x_dec[2], 1),
                'BAT_MWH'          : round(x_dec[3], 1),
                'technologie'      : self.technologie,
                'region'           : self.region,
                'scenario_batterie': self.scenario_batterie,
                'H2_prod_kt_an'    : round(res_sim.get('H2_vert_kg_an', res_sim['H2_prod_kg_an']) / 1e6, 3),
                'taux_curtailment' : round(res_sim['taux_curtailment'] * 100, 1),
                'heures_full_load' : res_sim['heures_full_load'],
                **lcoh_detail,
            })

        front_pareto.sort(key=lambda s: s['LCOH'])
        self.front_pareto_final = front_pareto

        duree = time.time() - t_debut
        print(f"\n  ✅ {label_bat} — {duree:.0f}s | {len(front_pareto)} solutions Pareto")
        print(f"     LCOH : [{min(s['LCOH'] for s in front_pareto):.2f} — "
              f"{max(s['LCOH'] for s in front_pareto):.2f}] $/kgH2")
        print(f"     Fiab : [{min(s['Fiabilite'] for s in front_pareto)*100:.1f} — "
              f"{max(s['Fiabilite'] for s in front_pareto)*100:.1f}]%")
        return front_pareto


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — ANALYSE DES RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════

def analyser_front_pareto(front_pareto, region, technologie, scenario_batterie='avec'):
    """
    Identifie 3 solutions clés sur le front de Pareto :
        - economique : LCOH minimal
        - equilibree : compromis optimal (distance utopie)
        - fiable     : fiabilité maximale
    """
    df = pd.DataFrame(front_pareto)

    lcoh_min, lcoh_max = df['LCOH'].min(), df['LCOH'].max()
    fib_min,  fib_max  = df['Fiabilite'].min(), df['Fiabilite'].max()
    df['LCOH_norm']  = (df['LCOH'] - lcoh_min) / (lcoh_max - lcoh_min + 1e-9)
    df['Fib_norm']   = 1 - (df['Fiabilite'] - fib_min) / (fib_max - fib_min + 1e-9)
    df['dist_utopie']= np.sqrt(df['LCOH_norm']**2 + df['Fib_norm']**2)

    solutions_cles = {
        'economique': df.loc[df['LCOH'].idxmin()].to_dict(),
        'equilibree': df.loc[df['dist_utopie'].idxmin()].to_dict(),
        'fiable'    : df.loc[df['Fiabilite'].idxmax()].to_dict(),
    }

    label = "AVEC" if scenario_batterie == 'avec' else "SANS"
    print(f"\n  ── Solutions clés — {region} | {technologie} | {label} batterie ──")
    for nom, sol in solutions_cles.items():
        print(f"  [{nom.upper():11s}] LCOH={sol['LCOH']:.2f} $/kg | "
              f"Fiab={sol['Fiabilite']*100:.1f}% | "
              f"PV={sol['PV_MW']:.0f}MW | EOL={sol['EOL_MW']:.0f}MW | "
              f"ELEC={sol['ELEC_MW']:.0f}MW | BAT={sol['BAT_MWH']:.0f}MWh")
    return df, solutions_cles


def exporter_resultats(front_pareto, historique, region, technologie,
                       scenario_batterie='avec'):
    """Exporte CSV + Excel multi-feuilles."""
    label    = scenario_batterie
    df_front = pd.DataFrame(front_pareto)
    df_hist  = pd.DataFrame(historique)

    # CSV
    df_front.to_csv(
        f"{OUTPUT_DIR2}/csv/Pareto_{region}_{technologie}_{label}.csv",
        index=False, encoding='utf-8-sig')
    df_hist.to_csv(
        f"{OUTPUT_DIR2}/csv/Convergence_{region}_{technologie}_{label}.csv",
        index=False, encoding='utf-8-sig')

    
    if EXCEL_OK:
        excel_path = f"{OUTPUT_DIR2}/csv/Resultats_{region}_{technologie}_{label}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_front.to_excel(writer, sheet_name='Front_Pareto', index=False)
            df_hist.to_excel(writer, sheet_name='Convergence', index=False)
            # Feuille résumé LCOH décomposé
            cols_lcoh = [c for c in df_front.columns if 'LCOH' in c or 'Fiab' in c
                         or c in ('PV_MW', 'EOL_MW', 'ELEC_MW', 'BAT_MWH')]
            df_front[cols_lcoh].to_excel(writer, sheet_name='LCOH_Detail', index=False)
        print(f"  ✓ Excel exporté : {excel_path}")

    print(f"  ✓ CSV exportés : Pareto ({len(df_front)} pts) + "
          f"Convergence ({len(df_hist)} gen)")
    return df_front, df_hist


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def fig9_profils_production(profils_df, region='Dakhla'):
    """Fig9 : Profils horaires 8760h."""
    print(f"  [Fig9] Profils — {region}...")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Profils Horaires 8760h — {region}\n"
                 f"Données d'entrée PyPSA (Étape 2)",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.28)

    for ax_idx, (semaine, titre) in enumerate(
        [(26, 'Semaine été (S26 — juin)'), (2, 'Semaine hiver (S2 — janvier)')]):
        ax   = fig.add_subplot(gs[0, ax_idx])
        h0   = semaine * 7 * 24
        hh   = range(168)
        ax.fill_between(hh, profils_df['CF_PV_h'].iloc[h0:h0+168]*100,
                        alpha=0.7, color=COLORS['PV'], label='PV')
        ax.fill_between(hh, profils_df['CF_eol_h'].iloc[h0:h0+168]*100,
                        alpha=0.6, color=COLORS['EOL'], label='Éolien')
        ax.set_title(titre)
        ax.set_xlabel('Heure'); ax.set_ylabel('CF (%)'); ax.set_ylim(0, 105)
        ax.set_xticks(range(0, 169, 24))
        ax.set_xticklabels(['L','M','M','J','V','S','D',''])
        ax.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[1, :])
    h_ann = np.arange(1, 8761)
    ax3.plot(h_ann, np.sort(profils_df['CF_PV_h'].values)[::-1]*100,
             color=COLORS['PV'], lw=2, label='PV')
    ax3.plot(h_ann, np.sort(profils_df['CF_eol_h'].values)[::-1]*100,
             color=COLORS['EOL'], lw=2, label='Éolien')
    hyb = (profils_df['CF_PV_h'].values + profils_df['CF_eol_h'].values) / 2
    ax3.plot(h_ann, np.sort(hyb)[::-1]*100,
             color=COLORS['accent'], lw=2.5, ls='--', label='Hybride moyen')
    ax3.axvline(4380, color='gray', ls=':', lw=1.5, label='50% du temps')
    ax3.set_xlabel('Heures classées'); ax3.set_ylabel('CF (%)')
    ax3.set_title('Courbe de durée annuelle')
    ax3.set_xlim(0, 8760); ax3.set_ylim(0, 105)
    ax3.legend(fontsize=9, ncol=4)

    plt.tight_layout()
    path = f"{OUTPUT_DIR2}/figures/Fig9_Profils_{region}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig9 → {path}")


def fig10_front_pareto(df_pareto, solutions_cles, region, technologie,
                       scenario_batterie='avec'):
    """Fig10 : Front de Pareto LCOH vs Fiabilité."""
    label = "AVEC batterie" if scenario_batterie == 'avec' else "SANS batterie"
    print(f"  [Fig10] Front Pareto — {region} | {technologie} | {label}...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Front de Pareto — {region} | {technologie} | {label}\n"
                 f"LCOH ($/kgH2) vs Fiabilité (%)",
                 fontsize=13, fontweight='bold')

    # Graphique Pareto
    ax = axes[0]
    ratio_pv_eol = df_pareto['PV_MW'] / (df_pareto['EOL_MW'] + 1)
    sc = ax.scatter(df_pareto['LCOH'], df_pareto['Fiabilite']*100,
                    c=ratio_pv_eol, cmap='RdYlGn',
                    s=60, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label='Ratio PV/Éolien')
    df_sort = df_pareto.sort_values('LCOH')
    ax.plot(df_sort['LCOH'], df_sort['Fiabilite']*100,
            color='gray', lw=1, alpha=0.5, zorder=2)

    styles = {
        'economique': ('*', COLORS['secondary'], 180, 'Économique'),
        'equilibree': ('D', COLORS['primary'],   120, 'Équilibrée'),
        'fiable'    : ('s', COLORS['accent'],    120, 'Fiable'),
    }
    for nom, (marker, color, size, lbl) in styles.items():
        sol = solutions_cles[nom]
        ax.scatter(sol['LCOH'], sol['Fiabilite']*100,
                   marker=marker, color=color, s=size, zorder=5,
                   edgecolors='black', linewidths=1.2, label=lbl)
        ax.annotate(f"  {lbl}\n  {sol['LCOH']:.2f} $/kg",
                    xy=(sol['LCOH'], sol['Fiabilite']*100), fontsize=8, color=color)

    ax.axvline(2.0, color='orange', ls='--', lw=1.5, alpha=0.7,
               label='Parité H2 gris (2 $/kg)')
    ax.axhline(80, color='blue', ls=':', lw=1.5, alpha=0.7, label='Cible 80%')
    ax.set_xlabel('LCOH ($/kgH2)'); ax.set_ylabel('Fiabilité (%)')
    ax.set_title('Front de Pareto'); ax.legend(fontsize=8, loc='lower right')

    # Dimensionnement solutions clés
    ax2   = axes[1]
    noms  = ['Économique', 'Équilibrée', 'Fiable']
    clefs = ['economique', 'equilibree', 'fiable']
    comps = ['PV_MW', 'EOL_MW', 'ELEC_MW', 'BAT_MWH']
    lbls  = ['PV (MW)', 'Éolien (MW)', 'Électrolyseur (MW)', 'Batterie (MWh)']
    cols  = [COLORS['PV'], COLORS['EOL'], COLORS['PEM'], COLORS['BAT']]
    x     = np.arange(len(noms))
    w     = 0.18
    for i, (comp, lbl, col) in enumerate(zip(comps, lbls, cols)):
        vals = [solutions_cles[k][comp] for k in clefs]
        bars = ax2.bar(x + i*w - 0.27, vals, w, label=lbl, color=col,
                       alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 3, f'{v:.0f}',
                     ha='center', fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(noms)
    ax2.set_ylabel('Capacité installée')
    ax2.set_title('Dimensionnement — Solutions clés')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    suf  = scenario_batterie
    path = f"{OUTPUT_DIR2}/figures/Fig10_Pareto_{region}_{technologie}_{suf}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig10 → {path}")


def fig11_convergence_multi_regions(historiques_dict):
    """Fig11 : Convergence NSGA-II multi-régions/scénarios."""
    print("  [Fig11] Convergence NSGA-II multi-configurations...")
    configs = list(historiques_dict.keys())
    n = len(configs)
    if n == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Convergence NSGA-II — LCOH et Fiabilité\n"
                 "Comparaison configurations avec/sans batterie",
                 fontsize=12, fontweight='bold')

    cmap = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (cfg, hist) in enumerate(historiques_dict.items()):
        df_h = pd.DataFrame(hist)
        # ✅ Style ligne selon scénario batterie
        ls = '-' if 'avec' in cfg else '--'
        ax1.plot(df_h['generation'], df_h['LCOH_min'],
                 color=cmap[i], lw=2, ls=ls, label=cfg)
        ax2.plot(df_h['generation'], df_h['Fiab_max']*100,
                 color=cmap[i], lw=2, ls=ls, label=cfg)

    ax1.axhline(2.0, color='orange', ls='--', lw=1.5, label='Cible 2030 (2 $/kg)')
    ax1.set_xlabel('Génération'); ax1.set_ylabel('LCOH minimal ($/kgH2)')
    ax1.set_title('Convergence LCOH'); ax1.legend(fontsize=7)

    ax2.axhline(80, color='blue', ls='--', lw=1.5, label='Cible 80%')
    ax2.set_xlabel('Génération'); ax2.set_ylabel('Fiabilité maximale (%)')
    ax2.set_title('Convergence Fiabilité'); ax2.legend(fontsize=7)

    #  Légende scénario batterie
    ligne_solid = Line2D([0], [0], color='gray', lw=2, ls='-')
    ligne_dash  = Line2D([0], [0], color='gray', lw=2, ls='--')
    ax2.legend(handles=[ligne_solid, ligne_dash],
               labels=['Avec batterie', 'Sans batterie'],
               fontsize=9, loc='lower right')

    plt.tight_layout()
    path = f"{OUTPUT_DIR2}/figures/Fig11_Convergence_Multi.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig11 → {path}")


def fig12_synthese_regions(resultats_dict):
    """Fig12 : Synthèse comparative LCOH et fiabilité par région."""
    print("  [Fig12] Synthèse comparative...")
    configs  = list(resultats_dict.keys())
    lcoh_equ = [resultats_dict[c]['equilibree']['LCOH'] for c in configs]
    fib_equ  = [resultats_dict[c]['equilibree']['Fiabilite']*100 for c in configs]
    lcoh_eco = [resultats_dict[c]['economique']['LCOH'] for c in configs]
    lcoh_fib = [resultats_dict[c]['fiable']['LCOH'] for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Synthèse Comparative — LCOH H2 Vert\n"
                 "Solutions Pareto-optimales (NSGA-II)", fontsize=12, fontweight='bold')
    x = np.arange(len(configs))
    ax = axes[0]
    ax.bar(x - 0.25, lcoh_eco, 0.25, label='Économique',
           color=COLORS['secondary'], alpha=0.85)
    ax.bar(x,        lcoh_equ, 0.25, label='Équilibrée',
           color=COLORS['primary'],   alpha=0.85)
    ax.bar(x + 0.25, lcoh_fib, 0.25, label='Fiable',
           color=COLORS['accent'],    alpha=0.85)
    ax.axhline(2.0, color='orange', ls='--', lw=1.5, label='Parité H2 gris 2030')
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('LCOH ($/kgH2)'); ax.set_title('Comparaison LCOH')
    ax.legend(fontsize=8)

    axes[1].bar(x, fib_equ, color=COLORS['primary'], alpha=0.85, edgecolor='white')
    axes[1].axhline(80, color='blue', ls='--', lw=1.5, label='Cible 80%')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    axes[1].set_ylabel('Fiabilité (%)'); axes[1].set_title('Fiabilité système')
    axes[1].set_ylim(0, 105); axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR2}/figures/Fig12_Synthese_Regions.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig12 → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  NOUVEAU MODULE 7 — FIGURE 13 : COMPARAISON AVEC/SANS BATTERIE
# ══════════════════════════════════════════════════════════════════════════════

def fig13_comparaison_scenarios_batterie(resultats_avec, resultats_sans,
                                          region, technologie):
    """
    Fig13 : Comparaison directe AVEC vs SANS batterie.

    Montre pour chaque région :
        - Gain LCOH apporté par la batterie (delta $/kg)
        - Gain fiabilité (delta %)
        - Capacité batterie optimale (MWh)
        - Taux curtailment avec/sans batterie

    Source méthode : analyse comparative scénarios — IEA (2022)
    """
    print(f"  [Fig13] Comparaison AVEC/SANS batterie — {region} | {technologie}...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparaison Scénarios Batterie — {region} | {technologie}\n"
                 f"Impact stockage sur LCOH et Fiabilité",
                 fontsize=13, fontweight='bold')

    noms_sol = ['Économique', 'Équilibrée', 'Fiable']
    clefs    = ['economique', 'equilibree', 'fiable']
    x        = np.arange(len(noms_sol))
    w        = 0.35

    # ── Graphique 1 : LCOH AVEC vs SANS ──────────────────────────────────────
    ax = axes[0, 0]
    lcoh_avec = [resultats_avec[k]['LCOH'] for k in clefs]
    lcoh_sans = [resultats_sans[k]['LCOH'] for k in clefs]
    bars1 = ax.bar(x - w/2, lcoh_avec, w, label='Avec batterie',
                   color=COLORS['avec_bat'], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, lcoh_sans, w, label='Sans batterie',
                   color=COLORS['sans_bat'], alpha=0.85, edgecolor='white')
    for bar, v in zip(bars1, lcoh_avec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{v:.2f}', ha='center', fontsize=8, color=COLORS['avec_bat'])
    for bar, v in zip(bars2, lcoh_sans):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{v:.2f}', ha='center', fontsize=8, color=COLORS['sans_bat'])
    ax.axhline(2.0, color='orange', ls='--', lw=1.5, label='Parité 2 $/kg')
    ax.set_xticks(x); ax.set_xticklabels(noms_sol)
    ax.set_ylabel('LCOH ($/kgH2)'); ax.set_title('Comparaison LCOH')
    ax.legend(fontsize=8)

    # ── Graphique 2 : Fiabilité AVEC vs SANS ─────────────────────────────────
    ax2 = axes[0, 1]
    fib_avec = [resultats_avec[k]['Fiabilite']*100 for k in clefs]
    fib_sans = [resultats_sans[k]['Fiabilite']*100 for k in clefs]
    ax2.bar(x - w/2, fib_avec, w, label='Avec batterie',
            color=COLORS['avec_bat'], alpha=0.85, edgecolor='white')
    ax2.bar(x + w/2, fib_sans, w, label='Sans batterie',
            color=COLORS['sans_bat'], alpha=0.85, edgecolor='white')
    ax2.axhline(80, color='blue', ls='--', lw=1.5, label='Cible 80%')
    ax2.set_xticks(x); ax2.set_xticklabels(noms_sol)
    ax2.set_ylabel('Fiabilité (%)'); ax2.set_title('Comparaison Fiabilité')
    ax2.set_ylim(0, 105); ax2.legend(fontsize=8)

    # ── Graphique 3 : Gain fiabilité (delta) ─────────────────────────────────
    ax3 = axes[1, 0]
    delta_fib  = [a - s for a, s in zip(fib_avec, fib_sans)]
    delta_lcoh = [s - a for a, s in zip(lcoh_avec, lcoh_sans)]  # positif = batterie moins chère
    colors_fib = [COLORS['avec_bat'] if d > 0 else COLORS['sans_bat']
                  for d in delta_fib]
    bars3 = ax3.bar(x, delta_fib, 0.5, color=colors_fib, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars3, delta_fib):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.5 if v >= 0 else -1.5),
                 f'+{v:.1f}%' if v >= 0 else f'{v:.1f}%',
                 ha='center', fontsize=9, fontweight='bold')
    ax3.axhline(0, color='black', lw=1)
    ax3.set_xticks(x); ax3.set_xticklabels(noms_sol)
    ax3.set_ylabel('Δ Fiabilité (pp)')
    ax3.set_title('Gain fiabilité apporté par la batterie')
    ax3.set_ylim(min(delta_fib) - 5, max(delta_fib) + 5)

    # ── Graphique 4 : Capacité batterie optimale (solution équilibrée) ────────
    ax4 = axes[1, 1]
    bat_vals = [resultats_avec[k]['BAT_MWH'] for k in clefs]
    bar_colors = [COLORS['avec_bat']] * 3
    bars4 = ax4.bar(x, bat_vals, 0.5, color=bar_colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars4, bat_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{v:.0f} MWh', ha='center', fontsize=9)
    ax4.set_xticks(x); ax4.set_xticklabels(noms_sol)
    ax4.set_ylabel('Capacité batterie (MWh)')
    ax4.set_title('Batterie optimale (scénario avec batterie)')

    plt.tight_layout()
    path = f"{OUTPUT_DIR2}/figures/Fig13_Comparaison_Batterie_{region}_{technologie}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig13 → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  NOUVEAU MODULE 8 — FIGURE 14 : DÉCOMPOSITION LCOH PAR COMPOSANTE (WATERFALL)
# ══════════════════════════════════════════════════════════════════════════════

def fig14_decomposition_lcoh(resultats_avec, resultats_sans,
                              region, technologie):
    """
    Fig14 : Décomposition LCOH en composantes (waterfall chart).

    Décompose le LCOH de la solution équilibrée en :
        CAPEX PV / Éolien / Électrolyseur / Batterie
        OPEX PV / Éolien / Électrolyseur / Batterie
        Coût eau

    Permet de visualiser quelle composante domine le coût.
    Source : IRENA (2020) — Green Hydrogen Cost Reduction, Fig. 2.11
    """
    print(f"  [Fig14] Décomposition LCOH waterfall — {region} | {technologie}...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Décomposition LCOH — {region} | {technologie}\n"
                 f"Solution équilibrée ($/kgH2)",
                 fontsize=13, fontweight='bold')

    composantes = [
        ('LCOH_PV_capex',   'CAPEX PV',    COLORS['PV']),
        ('LCOH_EOL_capex',  'CAPEX Éolien', COLORS['EOL']),
        ('LCOH_ELEC_capex', 'CAPEX Électrolyse', COLORS['PEM']),
        ('LCOH_BAT_capex',  'CAPEX Batterie', COLORS['BAT']),
        ('LCOH_PV_opex',    'O&M PV',       '#E6AC00'),
        ('LCOH_EOL_opex',   'O&M Éolien',   '#0097A7'),
        ('LCOH_ELEC_opex',  'O&M Électrolyse', '#1565C0'),
        ('LCOH_BAT_opex',   'O&M Batterie', '#6A1B9A'),
        ('LCOH_eau',        'Eau',          '#00796B'),
    ]

    for ax_i, (resultats, label_sc) in enumerate(
        [(resultats_avec, 'Avec batterie'), (resultats_sans, 'Sans batterie')]):
        ax    = axes[ax_i]
        sol   = resultats.get('equilibree', {})
        vals  = []
        lbls  = []
        cols  = []
        total = 0.0

        for key, nom, col in composantes:
            v = sol.get(key, 0.0)
            if v > 0.001:  # Ignorer composantes nulles (ex: batterie scénario sans)
                vals.append(v)
                lbls.append(nom)
                cols.append(col)
                total += v

        x_pos = np.arange(len(vals))
        bars  = ax.bar(x_pos, vals, color=cols, alpha=0.85,
                       edgecolor='white', linewidth=0.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=7.5, fontweight='bold')

        # Ligne LCOH total
        ax.axhline(total, color='red', ls='--', lw=2,
                   label=f'LCOH total = {total:.2f} $/kg')
        ax.axhline(2.0, color='orange', ls=':', lw=1.5,
                   label='Parité H2 gris (2 $/kg)')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(lbls, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Contribution LCOH ($/kgH2)')
        ax.set_title(f'{label_sc}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR2}/figures/Fig14_LCOH_Decompose_{region}_{technologie}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig14 → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL (mis à jour avec scénarios batterie)
# ══════════════════════════════════════════════════════════════════════════════

def lancer_etape2(configurations=None, pop_size=50, n_gen=30,
                  force_synthetic=False, verbose=True,
                  scenarios_batterie=('avec', 'sans')):   
    """
    Lance le pipeline complet de l'Étape 2.

    
        scenarios_batterie : tuple de scénarios à lancer
            ('avec', 'sans') → lance les deux et compare
            ('avec',)        → seulement avec batterie
            ('sans',)        → seulement sans batterie

    Retourne :
        resultats_finaux : dict {config_scenario: solutions_cles}
    """
    print("=" * 65)
    print("  ÉTAPE 2 — OPTIMISATION PRODUCTION H2 VERT (v2)")
    print("  PyPSA + NSGA-II | Scénarios AVEC/SANS batterie")
    print("=" * 65)

    if configurations is None:
        configurations = [
            ('Dakhla',      'AEL'),
            ('Ouarzazate',  'PEM'),
            ('Jorf_Lasfar', 'PEM'),
        ]

    resultats_finaux   = {}
    historiques_global = {}
    #  Stocker résultats par scénario pour Fig13/Fig14
    resultats_par_scenario = {}

    for region, technologie in configurations:
        print(f"\n{'─'*65}")
        print(f"  Configuration : {region} | {technologie}")
        print(f"{'─'*65}")

        # Chargement profils (une seule fois par région)
        print(f"\n  [A] Chargement profils horaires T10...")
        profils = charger_profils_T10(region, annee=2024,
                                      force_synthetic=force_synthetic)
        fig9_profils_production(profils, region)

        resultats_par_scenario[f"{region}_{technologie}"] = {}

        for scenario_bat in scenarios_batterie:
            cfg_label = f"{region}_{technologie}_{scenario_bat}"
            label_bat = "AVEC batterie" if scenario_bat == 'avec' else "SANS batterie"
            print(f"\n  ── {label_bat} ─────────────────────────────────────────")

            # ── NSGA-II ───────────────────────────────────────────────────────
            optimizer = NSGAII(profils, technologie=technologie,
                               region=region, pop_size=pop_size,
                               n_gen=n_gen, seed=42,
                               scenario_batterie=scenario_bat)
            front_pareto = optimizer.optimiser(verbose=verbose)

            # ── Analyse ───────────────────────────────────────────────────────
            df_pareto, solutions_cles = analyser_front_pareto(
                front_pareto, region, technologie, scenario_bat)

            # ── Export ────────────────────────────────────────────────────────
            exporter_resultats(front_pareto, optimizer.historique,
                               region, technologie, scenario_bat)

            # ── Figures par scénario ──────────────────────────────────────────
            fig10_front_pareto(df_pareto, solutions_cles,
                               region, technologie, scenario_bat)

            resultats_finaux[cfg_label]   = solutions_cles
            historiques_global[cfg_label] = optimizer.historique
            resultats_par_scenario[f"{region}_{technologie}"][scenario_bat] = solutions_cles

        # ── ✅ Figures de comparaison AVEC/SANS (si les deux scénarios lancés) ──
        if 'avec' in scenarios_batterie and 'sans' in scenarios_batterie:
            key_reg = f"{region}_{technologie}"
            fig13_comparaison_scenarios_batterie(
                resultats_par_scenario[key_reg]['avec'],
                resultats_par_scenario[key_reg]['sans'],
                region, technologie)
            fig14_decomposition_lcoh(
                resultats_par_scenario[key_reg]['avec'],
                resultats_par_scenario[key_reg]['sans'],
                region, technologie)

    # ── Figures de synthèse ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Figures de synthèse multi-configurations...")
    fig11_convergence_multi_regions(historiques_global)
    fig12_synthese_regions(resultats_finaux)

    # ── Rapport de synthèse ───────────────────────────────────────────────────
    _generer_rapport(resultats_finaux, resultats_par_scenario)

    print(f"\n{'='*65}")
    print(f"  ✅ ÉTAPE 2 COMPLÈTE")
    print(f"     Résultats dans : {OUTPUT_DIR2}")
    print(f"{'='*65}")

    return resultats_finaux


def _generer_rapport(resultats_finaux, resultats_par_scenario=None):
    """Génère un rapport JSON enrichi de l'Étape 2."""
    rapport = {
        'etape'          : 2,
        'version'        : 'v2',
        'description'    : 'Optimisation Production H2 — PyPSA + NSGA-II',
        'nouveautes_v2'  : [
            'Scénarios avec/sans batterie',
            'PyPSA modèle corrigé (curtailment, min_load)',
            'Électrolyseur : ramping + min_load + dégradation',
            'LCOH décomposé par composante',
            'Cache simulation NSGA-II',
            'Export Excel multi-feuilles',
        ],
        'configurations' : {},
        'comparaison_batterie': {},
    }

    for cfg, sols in resultats_finaux.items():
        rapport['configurations'][cfg] = {
            nom: {
                'LCOH_USD_kg'        : sol['LCOH'],
                'Fiabilite_pct'      : round(sol['Fiabilite'] * 100, 1),
                'PV_MW'              : sol['PV_MW'],
                'EOL_MW'             : sol['EOL_MW'],
                'ELEC_MW'            : sol['ELEC_MW'],
                'BAT_MWH'            : sol['BAT_MWH'],
                'scenario_batterie'  : sol.get('scenario_batterie', 'N/A'),
            }
            for nom, sol in sols.items()
        }

    # ✅ NOUVEAU : résumé comparaison avec/sans batterie
    if resultats_par_scenario:
        for key_reg, scenarios in resultats_par_scenario.items():
            if 'avec' in scenarios and 'sans' in scenarios:
                sol_av = scenarios['avec'].get('equilibree', {})
                sol_sa = scenarios['sans'].get('equilibree', {})
                if sol_av and sol_sa:
                    rapport['comparaison_batterie'][key_reg] = {
                        'LCOH_avec_bat'    : sol_av.get('LCOH', 0),
                        'LCOH_sans_bat'    : sol_sa.get('LCOH', 0),
                        'delta_LCOH'       : round(sol_sa.get('LCOH', 0)
                                                   - sol_av.get('LCOH', 0), 3),
                        'Fiab_avec_bat_pct': round(sol_av.get('Fiabilite', 0)*100, 1),
                        'Fiab_sans_bat_pct': round(sol_sa.get('Fiabilite', 0)*100, 1),
                        'delta_fiabilite_pp': round(
                            (sol_av.get('Fiabilite', 0)
                             - sol_sa.get('Fiabilite', 0)) * 100, 1),
                        'BAT_optimale_MWh' : sol_av.get('BAT_MWH', 0),
                    }

    with open(f"{OUTPUT_DIR2}/reports/rapport_etape2.json", 'w',
              encoding='utf-8') as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Rapport JSON v2 exporté")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# INTERMITTENCE — Stochasticité AR1 + HMM + Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 1 — STOCHASTICITÉ INTRA-JOURNALIÈRE (AR1 + événements discrets)
# ══════════════════════════════════════════════════════════════════════════════

def ajouter_intermittence_AR1(profils_df,
                               phi_pv=0.85,   sigma_pv=0.04,
                               phi_eol=0.70,  sigma_eol=0.06,
                               n_rafales=20,  duree_rafale_h=3,
                               n_calmes=15,   duree_calme_h=4,
                               seed=None):
    """
    Ajoute une intermittence réaliste aux profils CF_PV et CF_eol.

    Composantes :
        1. AR(1) : corrélation temporelle heure par heure
           CF(t) = mu + phi*(CF(t-1)-mu) + eps(t)
           phi élevé → inertie (bon temps persistant)
           phi faible → changements rapides (turbulence)

        2. Rafales éoliennes : fronts météo côtiers (Dakhla, Tanger)
           Profil trapézoïdal : montée | plateau | descente

        3. Calmes éoliens : anticyclones sahariens (été)
           Transition douce vers production quasi-nulle

        4. Passages nuageux PV : stratus, poussières sahariennes
           Uniquement pendant les heures de production

    Paramètres :
        phi_pv/phi_eol : autocorrélation [0-1]
                         PV  : 0.80-0.92 (inertie thermique atmosphère)
                         EOL : 0.65-0.78 (turbulence plus rapide)
        sigma_pv/sigma_eol : écart-type bruit AR1
        n_rafales      : nombre de rafales éoliennes par an  (15-30)
        duree_rafale_h : durée moyenne d'une rafale (2-6h)
        n_calmes       : nombre de calmes éoliens par an (10-20)
        duree_calme_h  : durée moyenne d'un calme (3-8h)

    Source :
        Bludszuweit et al. (2008) — Statistical analysis of PV output.
        Lorenz et al. (2011) — ECMWF solar irradiance forecasting.
    """
    if seed is not None:
        np.random.seed(seed)

    df = profils_df.copy()
    n  = len(df)

    CF_PV_orig  = df['CF_PV_h'].values.copy()
    CF_EOL_orig = df['CF_eol_h'].values.copy()

    # ── 1. Processus AR(1) ────────────────────────────────────────────────────
    def appliquer_AR1(cf_orig, phi, sigma):
        cf_new = cf_orig.copy()
        mu     = cf_orig.mean()
        eps    = np.random.normal(0, sigma, n)
        for t in range(1, n):
            cf_new[t] = mu + phi * (cf_new[t-1] - mu) + eps[t]
            # Pondération : 60% AR1 + 40% signal original (garde la forme)
            cf_new[t] = 0.6 * cf_new[t] + 0.4 * cf_orig[t]
        return np.clip(cf_new, 0, 1)

    CF_PV_ar1  = appliquer_AR1(CF_PV_orig,  phi_pv,  sigma_pv)
    CF_EOL_ar1 = appliquer_AR1(CF_EOL_orig, phi_eol, sigma_eol)

    # ── 2. Rafales éoliennes ──────────────────────────────────────────────────
    for _ in range(n_rafales):
        t_debut   = np.random.randint(0, n - duree_rafale_h * 3)
        duree     = np.random.randint(duree_rafale_h, duree_rafale_h * 2)
        intensite = np.random.uniform(0.15, 0.35)

        montee   = max(1, duree // 4)
        descente = max(1, duree // 4)
        plateau  = duree - montee - descente

        profil_rafale = np.concatenate([
            np.linspace(0, intensite, montee),
            np.full(plateau, intensite),
            np.linspace(intensite, 0, descente),
        ])

        t_fin = min(t_debut + len(profil_rafale), n)
        l     = t_fin - t_debut
        CF_EOL_ar1[t_debut:t_fin] = np.clip(
            CF_EOL_ar1[t_debut:t_fin] + profil_rafale[:l], 0, 1)

    # ── 3. Calmes éoliens prolongés ───────────────────────────────────────────
    for _ in range(n_calmes):
        t_debut   = np.random.randint(0, n - duree_calme_h * 2)
        duree     = np.random.randint(duree_calme_h, duree_calme_h * 2)
        niveau_bas = np.random.uniform(0.0, 0.04)
        t_fin     = min(t_debut + duree, n)
        transition = 6

        for h in range(t_debut, t_fin):
            if h < t_debut + transition:
                alpha = (h - t_debut) / transition
            elif h > t_fin - transition:
                alpha = (t_fin - h) / transition
            else:
                alpha = 1.0
            CF_EOL_ar1[h] = CF_EOL_ar1[h] * (1 - alpha) + niveau_bas * alpha

    CF_EOL_ar1 = np.clip(CF_EOL_ar1, 0, 1)

    # ── 4. Passages nuageux PV ────────────────────────────────────────────────
    heures_jour = CF_PV_orig > 0.05
    n_nuages    = int(n_rafales * 1.5)

    for _ in range(n_nuages):
        t_debut    = np.random.randint(0, n - 5)
        duree      = np.random.randint(1, 4)
        attenuation = np.random.uniform(0.30, 0.75)
        t_fin      = min(t_debut + duree, n)
        mask       = heures_jour[t_debut:t_fin]
        CF_PV_ar1[t_debut:t_fin][mask] *= (1 - attenuation)

    CF_PV_ar1 = np.clip(CF_PV_ar1, 0, 1)

    df['CF_PV_h']    = CF_PV_ar1
    df['CF_eol_h']   = CF_EOL_ar1
    df['CF_PV_orig'] = CF_PV_orig
    df['CF_eol_orig']= CF_EOL_orig

    delta_pv  = (CF_PV_ar1  - CF_PV_orig).std()
    delta_eol = (CF_EOL_ar1 - CF_EOL_orig).std()
    print(f"    AR1 : CF_PV={CF_PV_ar1.mean()*100:.1f}% (Δstd={delta_pv*100:.2f}%) | "
          f"CF_eol={CF_EOL_ar1.mean()*100:.1f}% (Δstd={delta_eol*100:.2f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 2 — CHAÎNE DE MARKOV CACHÉE (HMM) : régimes météo
# ══════════════════════════════════════════════════════════════════════════════

def generer_profil_HMM(profils_df, region='Dakhla', seed=None):
    """
    Modèle de Markov Caché (HMM) à 3 états pour les régimes météo.

    États cachés :
        0 — Ensoleillé/Venteux  : production maximale
        1 — Nuageux/Modéré      : production réduite
        2 — Couvert/Calme       : production minimale

    Matrice de transition (probabilité horaire de changer d'état) :
        Diagonale élevée (> 0.80) → régimes persistants (anticyclone, front actif)

    Facteurs multiplicatifs par état :
        PV  : [1.05, 0.65, 0.20]  — très sensible aux nuages
        EOL : [1.10, 0.80, 0.35]  — calme = production quasi-nulle

    Source :
        Papaefthymiou & Klöckl (2008) — MCMC for wind power assessment.
        Rodrigues et al. (2011) — HMM for renewable energy.

    Retourne :
        df_hmm, etats : DataFrame modifié + array états cachés (0/1/2)
    """
    if seed is not None:
        np.random.seed(seed)

    df = profils_df.copy()
    n  = len(df)

    # Matrices de transition calibrées par région (données ERA5 Maroc)
    TRANSITION = {
        'Dakhla'     : np.array([[0.93, 0.06, 0.01],
                                  [0.12, 0.80, 0.08],
                                  [0.03, 0.18, 0.79]]),
        'Ouarzazate' : np.array([[0.91, 0.08, 0.01],
                                  [0.10, 0.81, 0.09],
                                  [0.02, 0.16, 0.82]]),
        'Jorf_Lasfar': np.array([[0.88, 0.10, 0.02],
                                  [0.09, 0.79, 0.12],
                                  [0.02, 0.14, 0.84]]),
        'Tanger'     : np.array([[0.85, 0.12, 0.03],
                                  [0.08, 0.76, 0.16],
                                  [0.02, 0.12, 0.86]]),
        'Agadir'     : np.array([[0.89, 0.09, 0.02],
                                  [0.10, 0.80, 0.10],
                                  [0.02, 0.15, 0.83]]),
    }
    P = TRANSITION.get(region, TRANSITION['Dakhla'])

    FACT_PV  = [1.05, 0.65, 0.20]
    FACT_EOL = [1.10, 0.80, 0.35]

    # Simulation chaîne de Markov
    etats  = np.zeros(n, dtype=int)
    pi0    = np.array([0.65, 0.25, 0.10])
    etats[0] = np.random.choice(3, p=pi0)

    for t in range(1, n):
        etats[t] = np.random.choice(3, p=P[etats[t-1]])

    # Application des facteurs
    CF_PV_orig  = df['CF_PV_h'].values.copy()
    CF_EOL_orig = df['CF_eol_h'].values.copy()
    CF_PV_hmm   = CF_PV_orig.copy()
    CF_EOL_hmm  = CF_EOL_orig.copy()

    for e, (f_pv, f_eol) in enumerate(zip(FACT_PV, FACT_EOL)):
        mask = (etats == e)
        CF_PV_hmm[mask]  = np.clip(CF_PV_orig[mask]  * f_pv,  0, 1)
        CF_EOL_hmm[mask] = np.clip(CF_EOL_orig[mask] * f_eol, 0, 1)

    # Transitions douces (filtre moyenneur sur 2h aux changements d'état)
    changements = np.where(np.diff(etats) != 0)[0]
    for tc in changements:
        debut = max(0, tc - 1)
        fin   = min(n, tc + 3)
        CF_PV_hmm[debut:fin]  = pd.Series(CF_PV_hmm[debut:fin]).rolling(
            2, min_periods=1, center=True).mean().values
        CF_EOL_hmm[debut:fin] = pd.Series(CF_EOL_hmm[debut:fin]).rolling(
            2, min_periods=1, center=True).mean().values

    CF_PV_hmm  = np.clip(CF_PV_hmm,  0, 1)
    CF_EOL_hmm = np.clip(CF_EOL_hmm, 0, 1)

    df['CF_PV_h']     = CF_PV_hmm
    df['CF_eol_h']    = CF_EOL_hmm
    df['CF_PV_orig']  = CF_PV_orig
    df['CF_eol_orig'] = CF_EOL_orig
    df['etat_HMM']    = etats

    labels_etat = ['Ensoleillé/Venteux', 'Nuageux/Modéré', 'Couvert/Calme']
    for e in range(3):
        n_e   = (etats == e).sum()
        n_max = _duree_max_consecutives(etats, e)
        print(f"    HMM état {e} ({labels_etat[e]}) : "
              f"{n_e}h ({n_e/8760*100:.1f}%) | max consécutif : {n_max}h")

    return df, etats


def _duree_max_consecutives(etats, etat_cible):
    """Durée maximale consécutive d'un état donné."""
    max_dur = cur = 0
    for e in etats:
        cur = cur + 1 if e == etat_cible else 0
        max_dur = max(max_dur, cur)
    return max_dur


# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 3 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

def lancer_monte_carlo(region='Dakhla', technologie='AEL',
                        scenario_batterie='avec',
                        n_tirages=20,
                        pop_size=20, n_gen=15,
                        niveau_intermittence=1,
                        seed=0, verbose=False):
    """
    Analyse Monte Carlo : distribution du LCOH sous intermittence stochastique.

    Algorithme :
        Pour k = 1 → N :
            1. profil_k = profils_base + intermittence(seed=k)
            2. NSGA-II(profil_k) → front Pareto_k
            3. sol_k = solution_equilibree(Pareto_k)
            → LCOH_k, Fiabilite_k

        Résultat : distribution sur N tirages
            IC 90% = [P5, P95] du LCOH
            CV     = std/mean × 100% (coefficient de variation)

    Recommandation :
        n_tirages = 20-30 pour résultats publiables
        pop_size=20, n_gen=15 pour MC (réduit par rapport à l'optimisation finale)

    Source :
        IEA (2023) — Uncertainty in renewable energy assessments, p. 45.
    """
    if _STANDALONE:
        print("Mode autonome : Monte Carlo nécessite etape2_ameliore.py importé")
        return {}

    print(f"\n{'═'*65}")
    print(f"  MONTE CARLO — {region} | {technologie} | N={n_tirages}")
    print(f"  Niveau intermittence : {niveau_intermittence}")
    print(f"{'═'*65}")

    LCOH_list = []
    Fiab_list = []
    PV_list   = []
    EOL_list  = []
    ELEC_list = []
    BAT_list  = []

    for k in range(n_tirages):
        print(f"  Tirage {k+1:3d}/{n_tirages}...", end='\r')

        profils_base = charger_profils_T10(region, annee=2024, force_synthetic=True)

        if niveau_intermittence == 1:
            profils_k = ajouter_intermittence_AR1(profils_base, seed=seed + k)
        elif niveau_intermittence == 2:
            profils_k, _ = generer_profil_HMM(profils_base, region, seed=seed + k)
        else:
            profils_k, _ = generer_profil_HMM(profils_base, region, seed=seed + k)
            profils_k    = ajouter_intermittence_AR1(profils_k, seed=seed + k + 1000)

        optimizer = NSGAII(profils_k, technologie=technologie,
                           region=region, pop_size=pop_size,
                           n_gen=n_gen, seed=seed + k,
                           scenario_batterie=scenario_batterie)
        front_pareto = optimizer.optimiser(verbose=False)

        if not front_pareto:
            continue

        _, solutions_cles = analyser_front_pareto(
            front_pareto, region, technologie, scenario_batterie)
        sol = solutions_cles['equilibree']

        LCOH_list.append(sol['LCOH'])
        Fiab_list.append(sol['Fiabilite'] * 100)
        PV_list.append(sol['PV_MW'])
        EOL_list.append(sol['EOL_MW'])
        ELEC_list.append(sol['ELEC_MW'])
        BAT_list.append(sol['BAT_MWH'])

    print(f"\n  OK — {len(LCOH_list)}/{n_tirages} tirages valides")

    if not LCOH_list:
        return {}

    LCOH_arr = np.array(LCOH_list)
    Fiab_arr = np.array(Fiab_list)

    resultats_mc = {
        'region'             : region,
        'technologie'        : technologie,
        'scenario_batterie'  : scenario_batterie,
        'n_tirages'          : len(LCOH_list),
        'niveau_intermittence': niveau_intermittence,
        'LCOH_mean'          : float(LCOH_arr.mean()),
        'LCOH_std'           : float(LCOH_arr.std()),
        'LCOH_P5'            : float(np.percentile(LCOH_arr, 5)),
        'LCOH_P50'           : float(np.percentile(LCOH_arr, 50)),
        'LCOH_P95'           : float(np.percentile(LCOH_arr, 95)),
        'LCOH_CV_pct'        : float(LCOH_arr.std() / LCOH_arr.mean() * 100),
        'LCOH_list'          : LCOH_list,
        'Fiab_mean'          : float(Fiab_arr.mean()),
        'Fiab_std'           : float(Fiab_arr.std()),
        'Fiab_P5'            : float(np.percentile(Fiab_arr, 5)),
        'Fiab_P95'           : float(np.percentile(Fiab_arr, 95)),
        'Fiab_list'          : Fiab_list,
        'PV_mean'            : float(np.mean(PV_list)),
        'EOL_mean'           : float(np.mean(EOL_list)),
        'ELEC_mean'          : float(np.mean(ELEC_list)),
        'BAT_mean'           : float(np.mean(BAT_list)),
    }

    print(f"\n  LCOH : {resultats_mc['LCOH_mean']:.2f} $/kg "
          f"[IC90%: {resultats_mc['LCOH_P5']:.2f} — {resultats_mc['LCOH_P95']:.2f}] "
          f"CV={resultats_mc['LCOH_CV_pct']:.1f}%")
    print(f"  Fiab : {resultats_mc['Fiab_mean']:.1f}% "
          f"[IC90%: {resultats_mc['Fiab_P5']:.1f} — {resultats_mc['Fiab_P95']:.1f}]%")

    return resultats_mc


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — VISUALISATION INTERMITTENCE
# ══════════════════════════════════════════════════════════════════════════════

def fig15_intermittence(profils_orig, profils_inter, etats_hmm=None,
                         region='Dakhla'):
    """Fig15 : Comparaison profils originaux vs avec intermittence."""
    print(f"  [Fig15] Visualisation intermittence — {region}...")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Intermittence Sources ENR — {region}\n"
        f"Profils originaux vs modèle stochastique",
        fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    h0    = 26 * 7 * 24
    heures = range(168)

    # PV semaine été
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(heures, profils_orig['CF_PV_h'].iloc[h0:h0+168]*100,
                     alpha=0.5, color='#FFD54F', label='Original')
    ax1.fill_between(heures, profils_inter['CF_PV_h'].iloc[h0:h0+168]*100,
                     alpha=0.6, color=COLORS['PV'], label='Avec intermittence')
    ax1.set_title('PV — Semaine été (S26)')
    ax1.set_xlabel('Heure'); ax1.set_ylabel('CF (%)')
    ax1.set_ylim(0, 105)
    ax1.set_xticks(range(0, 169, 24))
    ax1.set_xticklabels(['L','M','M','J','V','S','D',''])
    ax1.legend(fontsize=8)

    # Éolien semaine été
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(heures, profils_orig['CF_eol_h'].iloc[h0:h0+168]*100,
                     alpha=0.5, color='#80DEEA', label='Original')
    ax2.fill_between(heures, profils_inter['CF_eol_h'].iloc[h0:h0+168]*100,
                     alpha=0.6, color=COLORS['EOL'], label='Avec intermittence')
    diff_eol = (profils_inter['CF_eol_h'].iloc[h0:h0+168].values
                - profils_orig['CF_eol_h'].iloc[h0:h0+168].values)
    ax2.fill_between(heures, 0, np.maximum(diff_eol, 0)*100,
                     alpha=0.4, color='red', label='Rafales')
    ax2.fill_between(heures, 0, np.minimum(diff_eol, 0)*100,
                     alpha=0.4, color='gray', label='Calmes')
    ax2.set_title('Éolien — Semaine été (S26)')
    ax2.set_xlabel('Heure'); ax2.set_ylabel('CF (%)')
    ax2.set_ylim(-10, 105)
    ax2.set_xticks(range(0, 169, 24))
    ax2.set_xticklabels(['L','M','M','J','V','S','D',''])
    ax2.legend(fontsize=7)

    # Courbe de durée annuelle
    ax3 = fig.add_subplot(gs[1, :])
    h_ann = np.arange(1, 8761)
    for profil, ls, alpha, lbl_pv, lbl_eol in [
        (profils_orig,  '-',  0.5, 'PV original',        'Éolien original'),
        (profils_inter, '--', 0.9, 'PV + intermittence',  'Éolien + intermittence'),
    ]:
        ax3.plot(h_ann, np.sort(profil['CF_PV_h'].values)[::-1]*100,
                 color=COLORS['PV'], lw=2, ls=ls, alpha=alpha, label=lbl_pv)
        ax3.plot(h_ann, np.sort(profil['CF_eol_h'].values)[::-1]*100,
                 color=COLORS['EOL'], lw=2, ls=ls, alpha=alpha, label=lbl_eol)
    ax3.axvline(4380, color='gray', ls=':', lw=1.5, label='50% du temps')
    ax3.set_xlabel('Heures classées'); ax3.set_ylabel('CF (%)')
    ax3.set_title('Courbe de durée annuelle')
    ax3.set_xlim(0, 8760); ax3.set_ylim(0, 105)
    ax3.legend(fontsize=8, ncol=3)

    # États HMM ou différence annuelle
    ax4 = fig.add_subplot(gs[2, :])
    if etats_hmm is not None:
        h_sem      = np.arange(168)
        etat_sem   = etats_hmm[h0:h0+168]
        cols_etat  = {0: '#27AE60', 1: '#F39C12', 2: '#E74C3C'}
        lbls_etat  = {0: 'Ensoleillé/Venteux', 1: 'Nuageux/Modéré', 2: 'Couvert/Calme'}
        for e in range(3):
            ax4.fill_between(h_sem, 0, (etat_sem == e).astype(float),
                             alpha=0.65, color=cols_etat[e], label=lbls_etat[e])
        ax4.set_title('États HMM — Semaine été (S26)')
        ax4.set_xlabel('Heure')
        ax4.set_yticks([0, 1]); ax4.set_yticklabels(['', 'Actif'])
        ax4.set_xticks(range(0, 169, 24))
        ax4.set_xticklabels(['L','M','M','J','V','S','D',''])
        ax4.legend(fontsize=8, ncol=3)
        for e in range(3):
            n_e   = (etats_hmm == e).sum()
            n_max = _duree_max_consecutives(etats_hmm, e)
            ax4.text(0.01 + e*0.33, 0.88,
                     f'État {e}: {n_e}h ({n_e/8760*100:.0f}%) | max {n_max}h',
                     transform=ax4.transAxes, fontsize=8, color=cols_etat[e])
    else:
        diff_ann = profils_inter['CF_eol_h'].values - profils_orig['CF_eol_h'].values
        ax4.plot(np.arange(8760), diff_ann*100, color=COLORS['EOL'], lw=0.5, alpha=0.6)
        ax4.axhline(0, color='black', lw=1)
        ax4.fill_between(np.arange(8760), 0, diff_ann*100,
                         where=diff_ann > 0, alpha=0.4, color='green', label='Surplus')
        ax4.fill_between(np.arange(8760), 0, diff_ann*100,
                         where=diff_ann < 0, alpha=0.4, color='red', label='Déficit')
        ax4.set_xlabel('Heure de l\'année'); ax4.set_ylabel('ΔCF éolien (%)')
        ax4.set_title('Différence annuelle éolien (intermittence - original)')
        ax4.legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR2}/figures/Fig15_Intermittence_{region}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig15 → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16 — RÉSULTATS MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

def fig16_monte_carlo(resultats_mc):
    """Fig16 : Distribution LCOH et fiabilité sous incertitude Monte Carlo."""
    if not resultats_mc:
        return
    region   = resultats_mc['region']
    tech     = resultats_mc['technologie']
    n        = resultats_mc['n_tirages']
    LCOH_arr = np.array(resultats_mc['LCOH_list'])
    Fiab_arr = np.array(resultats_mc['Fiab_list'])

    print(f"  [Fig16] Monte Carlo — {region} | {tech}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Monte Carlo — {region} | {tech} | N={n} tirages\n"
                 f"LCOH et Fiabilité sous intermittence stochastique",
                 fontsize=13, fontweight='bold')

    # Distribution LCOH
    ax1 = axes[0, 0]
    ax1.hist(LCOH_arr, bins=max(8, n//3), color=COLORS['primary'],
             alpha=0.75, edgecolor='white')
    ax1.axvline(resultats_mc['LCOH_mean'], color='black', lw=2,
                label=f"Moy = {resultats_mc['LCOH_mean']:.2f} $/kg")
    ax1.axvline(resultats_mc['LCOH_P5'],  color='blue', lw=1.5, ls='--',
                label=f"P5  = {resultats_mc['LCOH_P5']:.2f} $/kg")
    ax1.axvline(resultats_mc['LCOH_P95'], color='red',  lw=1.5, ls='--',
                label=f"P95 = {resultats_mc['LCOH_P95']:.2f} $/kg")
    ax1.axvline(2.0, color='orange', lw=1.5, ls=':', label='Parité 2 $/kg')
    ax1.axvspan(resultats_mc['LCOH_P5'], resultats_mc['LCOH_P95'],
                alpha=0.15, color='blue', label='IC 90%')
    ax1.set_xlabel('LCOH ($/kgH2)'); ax1.set_ylabel('Fréquence')
    ax1.set_title(f"Distribution LCOH | CV={resultats_mc['LCOH_CV_pct']:.1f}%")
    ax1.legend(fontsize=8)

    # Distribution Fiabilité
    ax2 = axes[0, 1]
    ax2.hist(Fiab_arr, bins=max(8, n//3), color=COLORS['accent'],
             alpha=0.75, edgecolor='white')
    ax2.axvline(resultats_mc['Fiab_mean'], color='black', lw=2,
                label=f"Moy = {resultats_mc['Fiab_mean']:.1f}%")
    ax2.axvline(resultats_mc['Fiab_P5'],  color='blue', lw=1.5, ls='--',
                label=f"P5  = {resultats_mc['Fiab_P5']:.1f}%")
    ax2.axvline(resultats_mc['Fiab_P95'], color='red',  lw=1.5, ls='--',
                label=f"P95 = {resultats_mc['Fiab_P95']:.1f}%")
    ax2.axvline(80, color='blue', ls=':', lw=1.5, label='Cible 80%')
    ax2.axvspan(resultats_mc['Fiab_P5'], resultats_mc['Fiab_P95'],
                alpha=0.15, color='blue', label='IC 90%')
    ax2.set_xlabel('Fiabilité (%)'); ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution Fiabilité')
    ax2.legend(fontsize=8)

    # Nuage LCOH vs Fiabilité
    ax3 = axes[1, 0]
    sc = ax3.scatter(LCOH_arr, Fiab_arr, c=np.arange(n),
                     cmap='plasma', s=60, alpha=0.8,
                     edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=ax3, label='Numéro tirage')
    ax3.scatter(resultats_mc['LCOH_mean'], resultats_mc['Fiab_mean'],
                marker='X', s=200, color='red', zorder=5,
                edgecolors='black', linewidths=1, label='Moyenne MC')
    ax3.axvline(2.0, color='orange', ls='--', lw=1.5, alpha=0.7, label='2 $/kg')
    ax3.axhline(80,  color='blue',   ls=':',  lw=1.5, alpha=0.7, label='80%')
    ax3.set_xlabel('LCOH ($/kgH2)'); ax3.set_ylabel('Fiabilité (%)')
    ax3.set_title('Nuage LCOH vs Fiabilité')
    ax3.legend(fontsize=8)

    # Convergence de l'IC 90% selon N
    ax4 = axes[1, 1]
    n_min   = 3
    n_range = np.arange(n_min, n + 1)
    p5_c    = [np.percentile(LCOH_arr[:k], 5)  for k in n_range]
    p95_c   = [np.percentile(LCOH_arr[:k], 95) for k in n_range]
    moy_c   = [np.mean(LCOH_arr[:k])           for k in n_range]
    ax4.fill_between(n_range, p5_c, p95_c, alpha=0.3, color='blue', label='IC 90%')
    ax4.plot(n_range, moy_c, color='black', lw=2, label='Moyenne')
    ax4.plot(n_range, p5_c,  color='blue',  lw=1.5, ls='--', label='P5')
    ax4.plot(n_range, p95_c, color='red',   lw=1.5, ls='--', label='P95')
    ax4.axhline(2.0, color='orange', ls=':', lw=1.5, label='Parité 2 $/kg')
    ax4.set_xlabel('Nombre de tirages Monte Carlo')
    ax4.set_ylabel('LCOH ($/kgH2)')
    ax4.set_title('Convergence IC 90% selon N')
    ax4.legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR2}/figures/Fig16_MonteCarlo_{region}_{tech}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     OK : Fig16 → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL — INTÉGRATION COMPLÈTE
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_avec_intermittence(region='Dakhla', technologie='AEL',
                                  scenario_batterie='avec',
                                  niveau=2,
                                  pop_size=30, n_gen=20,
                                  n_tirages_mc=20,
                                  verbose=True):
    """
    Pipeline complet d'analyse de l'intermittence.

    Usage dans __main__ de etape2_ameliore.py :

        from module_intermittence import pipeline_avec_intermittence

        resultats_inter = pipeline_avec_intermittence(
            region='Dakhla', technologie='AEL',
            scenario_batterie='avec',
            niveau=2,               # 1=AR1, 2=HMM, 3=AR1+HMM
            pop_size=30, n_gen=20,
            n_tirages_mc=20,
        )

    Étapes :
        A. Chargement profils de base
        B. Application intermittence (niveau 1, 2 ou 3)
        C. Visualisation Fig15
        D. NSGA-II sur profils perturbés
        E. Comparaison avec/sans intermittence
        F. Monte Carlo N tirages → Fig16
    """
    if _STANDALONE:
        print("Nécessite etape2_ameliore.py importé pour NSGA-II + Monte Carlo")
        return {}

    print(f"\n{'='*65}")
    print(f"  PIPELINE INTERMITTENCE — {region} | {technologie}")
    print(f"  Niveau {niveau} | Scénario : {scenario_batterie}")
    print(f"{'='*65}")

    # A. Profils de base
    print("\n  [A] Profils de base...")
    profils_base = charger_profils_T10(region, annee=2024, force_synthetic=False)

    # B. Intermittence
    print(f"\n  [B] Intermittence niveau {niveau}...")
    etats_hmm = None
    if niveau == 1:
        profils_inter = ajouter_intermittence_AR1(profils_base, seed=42)
    elif niveau == 2:
        profils_inter, etats_hmm = generer_profil_HMM(profils_base, region, seed=42)
    else:
        profils_inter, etats_hmm = generer_profil_HMM(profils_base, region, seed=42)
        profils_inter = ajouter_intermittence_AR1(profils_inter, seed=43)

    # C. Fig15
    fig15_intermittence(profils_base, profils_inter, etats_hmm, region)

    # D. NSGA-II sur profils perturbés
    print(f"\n  [D] NSGA-II sur profils avec intermittence...")
    opt_inter = NSGAII(profils_inter, technologie=technologie, region=region,
                       pop_size=pop_size, n_gen=n_gen, seed=42,
                       scenario_batterie=scenario_batterie)
    front_inter = opt_inter.optimiser(verbose=verbose)
    _, sols_inter = analyser_front_pareto(front_inter, region, technologie,
                                           scenario_batterie)

    # E. Comparaison original vs intermittent
    print(f"\n  [E] Comparaison original vs intermittent...")
    opt_orig  = NSGAII(profils_base, technologie=technologie, region=region,
                       pop_size=pop_size, n_gen=n_gen, seed=42,
                       scenario_batterie=scenario_batterie)
    front_orig = opt_orig.optimiser(verbose=False)
    _, sols_orig = analyser_front_pareto(front_orig, region, technologie,
                                          scenario_batterie)

    sol_i = sols_inter['equilibree']
    sol_o = sols_orig['equilibree']
    print(f"\n  ── Impact intermittence ────────────────────────────────────")
    print(f"  Original      : LCOH={sol_o['LCOH']:.2f} $/kg | "
          f"Fiab={sol_o['Fiabilite']*100:.1f}%")
    print(f"  Intermittent  : LCOH={sol_i['LCOH']:.2f} $/kg | "
          f"Fiab={sol_i['Fiabilite']*100:.1f}%")
    print(f"  Impact        : ΔLCOH={sol_i['LCOH']-sol_o['LCOH']:+.2f} $/kg | "
          f"ΔFiab={(sol_i['Fiabilite']-sol_o['Fiabilite'])*100:+.1f}%")

    # F. Monte Carlo
    print(f"\n  [F] Monte Carlo ({n_tirages_mc} tirages)...")
    resultats_mc = lancer_monte_carlo(
        region=region, technologie=technologie,
        scenario_batterie=scenario_batterie,
        n_tirages=n_tirages_mc,
        pop_size=pop_size, n_gen=n_gen,
        niveau_intermittence=niveau,
        seed=0, verbose=False)

    fig16_monte_carlo(resultats_mc)

    return {
        'profils_base'    : profils_base,
        'profils_inter'   : profils_inter,
        'etats_hmm'       : etats_hmm,
        'solutions_orig'  : sols_orig,
        'solutions_inter' : sols_inter,
        'resultats_mc'    : resultats_mc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST AUTONOME
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE — PIPELINE COMPLET AVEC INTERMITTENCE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*65)
    print("  H2 MAROC — ETAPE 2 v3 : PRODUCTION + INTERMITTENCE")
    print("  PyPSA + NSGA-II + AR1/HMM + Monte Carlo")
    print("="*65)

    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------

    CONFIGURATIONS = [
        ('Dakhla',      'AEL'),   # Meilleur potentiel eolien Maroc
        ('Ouarzazate',  'PEM'),   # Meilleur potentiel solaire Maroc
        ('Jorf_Lasfar', 'PEM'),   # Hub industriel cote atlantique
    ]

    SCENARIOS_BATTERIE = ('avec', 'sans')

    # pop_size=20/n_gen=10 pour test rapide, 60/50 pour resultats fins
    POP_SIZE = 30
    N_GEN    = 20
    FORCE_SYNTHETIC = False

    # -------------------------------------------------------------------------
    # MODE EXECUTION
    # -------------------------------------------------------------------------
    # MODE 1 : Optimisation standard (sans intermittence)
    # MODE 2 : Intermittence AR1 (stochasticite intra-journaliere)
    # MODE 3 : Intermittence HMM (regimes meteo Markov)
    # MODE 4 : AR1 + HMM combines (le plus complet)
    # MODE 5 : Monte Carlo complet (N tirages -> intervalles de confiance LCOH)
    MODE = 2   # <- changer ici : 1, 2, 3, 4 ou 5

    N_TIRAGES_MC = 20   # Pour MODE 5

    # -------------------------------------------------------------------------
    # ETAPE A — OPTIMISATION STANDARD
    # -------------------------------------------------------------------------
    print("\n[ETAPE A] Optimisation standard (profils deterministes)...")
    resultats = lancer_etape2(
        configurations     = CONFIGURATIONS,
        pop_size           = POP_SIZE,
        n_gen              = N_GEN,
        force_synthetic    = FORCE_SYNTHETIC,
        verbose            = True,
        scenarios_batterie = SCENARIOS_BATTERIE,
    )

    print("\n=== Resume solutions equilibrees (standard) ===")
    for cfg, sols in resultats.items():
        sol = sols["equilibree"]
        bat_label = f"+BAT={sol["BAT_MWH"]:.0f}MWh" if sol["BAT_MWH"] > 0 else "+SANS_BAT"
        print(f"  {cfg:30s} -> LCOH={sol["LCOH"]:.2f} $/kg | "
              f"Fiab={sol["Fiabilite"]*100:.1f}% | "
              f"PV={sol["PV_MW"]:.0f}MW+EOL={sol["EOL_MW"]:.0f}MW+"
              f"ELEC={sol["ELEC_MW"]:.0f}MW{bat_label}")

    if len(SCENARIOS_BATTERIE) == 2:
        print("\n=== Comparaison AVEC vs SANS batterie ===")
        regions_vues = set()
        for cfg in resultats:
            parts = cfg.rsplit("_", 1)
            if len(parts) == 2:
                base, sc = parts[0], parts[1]
                if base not in regions_vues:
                    clef_avec = f"{base}_avec"
                    clef_sans = f"{base}_sans"
                    if clef_avec in resultats and clef_sans in resultats:
                        sol_av = resultats[clef_avec]["equilibree"]
                        sol_sa = resultats[clef_sans]["equilibree"]
                        delta_lcoh = sol_sa["LCOH"] - sol_av["LCOH"]
                        delta_fib  = (sol_av["Fiabilite"] - sol_sa["Fiabilite"]) * 100
                        print(f"  {base:25s} | DELTA_LCOH={delta_lcoh:+.2f} $/kg | "
                              f"DELTA_Fiab={delta_fib:+.1f}% | BAT={sol_av["BAT_MWH"]:.0f}MWh")
                        regions_vues.add(base)

    if MODE == 1:
        print(f"\nTermine (MODE 1) — resultats dans : {OUTPUT_DIR2}")
    else:
        # -------------------------------------------------------------------
        # ETAPE B — ANALYSE INTERMITTENCE
        # -------------------------------------------------------------------
        print(f"\n[ETAPE B] Analyse intermittence (MODE {MODE})...")

        REGION_REF   = "Dakhla"
        TECHNO_REF   = "AEL"
        SCENARIO_REF = "avec"

        print(f"  Region de reference : {REGION_REF} | {TECHNO_REF}")

        profils_base = charger_profils_T10(REGION_REF, annee=2024,
                                            force_synthetic=FORCE_SYNTHETIC)
        etats_hmm = None

        if MODE == 2:
            print("\n  Niveau 1 : Stochasticite AR1...")
            profils_inter = ajouter_intermittence_AR1(profils_base, seed=42)
        elif MODE == 3:
            print("\n  Niveau 2 : HMM (regimes meteo)...")
            profils_inter, etats_hmm = generer_profil_HMM(
                profils_base, region=REGION_REF, seed=42)
        else:
            print("\n  Niveau 3 : AR1 + HMM combines...")
            profils_inter, etats_hmm = generer_profil_HMM(
                profils_base, region=REGION_REF, seed=42)
            profils_inter = ajouter_intermittence_AR1(profils_inter, seed=43)

        fig15_intermittence(profils_base, profils_inter, etats_hmm, REGION_REF)

        print("\n  NSGA-II sur profils avec intermittence...")
        _SIM_CACHE.clear()
        opt_inter = NSGAII(profils_inter, technologie=TECHNO_REF,
                           region=REGION_REF, pop_size=POP_SIZE,
                           n_gen=N_GEN, seed=42,
                           scenario_batterie=SCENARIO_REF)
        front_inter = opt_inter.optimiser(verbose=True)
        _, sols_inter = analyser_front_pareto(
            front_inter, REGION_REF, TECHNO_REF, SCENARIO_REF)

        key_ref = f"{REGION_REF}_{TECHNO_REF}_{SCENARIO_REF}"
        sol_ref = resultats.get(key_ref, {})
        sol_i   = sols_inter["equilibree"]
        sol_o   = sol_ref.get("equilibree", sol_i)
        print("\n  Impact intermittence (solution equilibree) :")
        print(f"  Sans : LCOH={sol_o["LCOH"]:.2f} $/kg | Fiab={sol_o["Fiabilite"]*100:.1f}%")
        print(f"  Avec : LCOH={sol_i["LCOH"]:.2f} $/kg | Fiab={sol_i["Fiabilite"]*100:.1f}%")
        print(f"  Delta LCOH : {sol_i["LCOH"]-sol_o["LCOH"]:+.2f} $/kg | "
              f"Delta Fiab : {(sol_i["Fiabilite"]-sol_o["Fiabilite"])*100:+.1f}%")

        if MODE == 5:
            # ---------------------------------------------------------------
            # ETAPE C — MONTE CARLO
            # ---------------------------------------------------------------
            print(f"\n[ETAPE C] Monte Carlo ({N_TIRAGES_MC} tirages)...")
            resultats_mc = lancer_monte_carlo(
                region            = REGION_REF,
                technologie       = TECHNO_REF,
                scenario_batterie = SCENARIO_REF,
                n_tirages         = N_TIRAGES_MC,
                pop_size          = max(15, POP_SIZE // 2),
                n_gen             = max(10, N_GEN // 2),
                niveau_intermittence = 3,
                seed=0, verbose=False)

            fig16_monte_carlo(resultats_mc)

            print("\n  === Resume Monte Carlo ===")
            print(f"  LCOH : {resultats_mc["LCOH_mean"]:.2f} $/kg "
                  f"[IC90%: {resultats_mc["LCOH_P5"]:.2f} -- {resultats_mc["LCOH_P95"]:.2f}] "
                  f"CV={resultats_mc["LCOH_CV_pct"]:.1f}%")
            print(f"  Fiab : {resultats_mc["Fiab_mean"]:.1f}% "
                  f"[IC90%: {resultats_mc["Fiab_P5"]:.1f} -- {resultats_mc["Fiab_P95"]:.1f}]%")
            print(f"  Dim. moy : PV={resultats_mc["PV_mean"]:.0f}MW | "
                  f"EOL={resultats_mc["EOL_mean"]:.0f}MW | "
                  f"ELEC={resultats_mc["ELEC_mean"]:.0f}MW | "
                  f"BAT={resultats_mc["BAT_mean"]:.0f}MWh")

        print(f"\nTermine — resultats dans : {OUTPUT_DIR2}")
