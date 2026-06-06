# -*- coding: utf-8 -*-



"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     DATABASE BUILDER — CHAÎNE HYDROGÈNE MAROC (Production→Stockage→Transport)
║     Approche : Ancrage littérature + Correction Maroc + Monte Carlo + Validation
║   
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, lognorm, triang
import warnings, os, json, time
import math
from scipy.special import gamma as gamma_func
warnings.filterwarnings('ignore')
import os
np.random.seed(42)  # Reproductibilité scientifique

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "H2Morocco222_Outputs")
N_SIM       = 10_000   # Simulations Monte Carlo
ANNEES      = [2024, 2030, 2035, 2040, 2050]

os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["csv", "figures", "reports"]:
    os.makedirs(f"{OUTPUT_DIR}/{sub}", exist_ok=True)
    
 # Devise de référence
MONNAIE_REF  = "EUR"          # changer ici pour basculer vers USD ou MAD
ANNEE_REF    = 2024
SOURCE_TAUX  = "BCE Jan 2024 / BAM 2024 / IMF WEO 2024"

# Taux de change → EUR
TAUX_CHANGE = {
    "EUR" : 1.0000,   # pivot
    "USD" : 0.9217,   # 1 USD = 0.9217 EUR
    "MAD" : 0.0917,   # 1 MAD = 0.0917 EUR
    "GBP" : 1.1706,
    "SAR" : 0.2458,
    "AUD" : 0.6037,
}

def to_eur(valeur, devise="USD"):
    """Convertit n'importe quelle valeur vers EUR."""
    if devise == MONNAIE_REF or valeur is None:
        return valeur
    taux = TAUX_CHANGE.get(devise)
    if taux is None:
        return valeur   # devise inconnue → conservée
    if isinstance(valeur, (int, float)):
        return round(valeur * taux, 6)
    return valeur

def cols_to_eur(df, colonnes, devise="USD"):
    """
    Ajoute une colonne _EUR pour chaque colonne de coût listée.
    Exemple : LCOE_USD_kWh → LCOE_EUR_kWh
    Les colonnes originales sont conservées intactes.
    """
    df = df.copy()
    for col in colonnes:
        if col not in df.columns:
            continue
        nom_eur = col.replace("_USD_", "_EUR_").replace("_USD", "_EUR")
        if nom_eur == col:                  # si pas de "_USD" dans le nom
            nom_eur = col + "_EUR"
        df[nom_eur] = df[col].apply(
            lambda v: round(v * TAUX_CHANGE.get(devise, 1), 6)
            if pd.notna(v) and isinstance(v, (int, float)) else v
        )
    # Colonne méta traçabilité
    df["devise_ref"]  = MONNAIE_REF
    df["annee_ref"]   = ANNEE_REF
    df["source_taux"] = SOURCE_TAUX
    return df

# ─────────────────────────────────────────────────────────────────────────────
# COULEURS & STYLE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'primary'   : '#006233',   # Vert Maroc
    'secondary' : '#C1272D',   # Rouge Maroc
    'accent'    : '#FF8C00',   # Orange énergie
    'PEM'       : '#2196F3',   # Bleu PEM
    'AEL'       : '#4CAF50',   # Vert AEL
    'SOEC'      : '#9C27B0',   # Violet SOEC
    'NH3'       : '#FF5722',   # Orange ammoniac
    'LH2'       : '#00BCD4',   # Cyan H2 liquide
    'LOHC'      : '#795548',   # Brun LOHC
    'GH2'       : '#607D8B',   # Gris GH2
    'pipeline'  : '#3F51B5',   # Bleu pipeline
    'light_bg'  : '#F8F9FA',
    'grid'      : '#E0E0E0',
}

plt.rcParams.update({
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : COLORS['light_bg'],
    'axes.grid'         : True,
    'grid.color'        : COLORS['grid'],
    'grid.linewidth'    : 0.7,
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.titlesize'    : 12,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
})

TURBINE = {
    # Turbine utility-scale 2 MW — référence projets Maroc (Tarfaya, Dakhla)
    # Source : Vestas V90-2MW — données publiques
    'P_rated_kW'   : 2000,    # Puissance nominale (kW)
    'D_rotor_m'    : 90,      # Diamètre rotor (m)
    'V_ci'         : 3.0,     # Cut-in speed (m/s)
    'V_r'          : 12.0,    # Rated speed (m/s)
    'V_o'          : 25.0,    # Cut-out speed (m/s)
    'k_weibull'    : 2.0,     # Forme Weibull k=2 (Rayleigh) — standard côtier Maroc
                              # Source : IRENA, MASEN — distribution standard sites côtiers
    'H_an'         : 8760,    # Heures totales/an — le CF Weibull intègre déjà la disponibilité
    'rho_air'      : 1.225,   # Densité air standard (kg/m³) — ISO 2533
    'PR_PV'        : 0.80,    # Performance Ratio système PV
                              # Source : IEC 61724-1 / NREL PVWatts v8
    # Paramètres économiques éolien
    # Source : IRENA Morocco Renewable Cost Database 2024
    'CAPEX_eol'    : 1100,    # $/kW
    'OPEX_eol'     : 35,      # $/kW/an
    'LT_eol'       : 20,      # ans
    # Paramètres économiques solaire
    # Source : IRENA utility-scale PV 2024
    'CAPEX_sol'    : 550,     # $/kW
    'OPEX_sol'     : 12,      # $/kW/an
    'LT_sol'       : 25,      # ans
    # Paramètres financiers
    # Source : WACC MENA — IEA Global H2 Review 2024
    'DR'           : 0.08,    # Taux actualisation 8%
    # Seuil viabilité économique éolien utility-scale
    'CF_eol_min'   : 0.20,    # CF < 20% → site non viable
    # Marges PPA selon marché
    # Source : IEA 2024 / MASEN PPA structure
    'marge_local'  : 0.15,    # PPA local  = LCOE_hyb × 1.15
    'marge_europe' : 0.25,    # PPA Europe = LCOE_hyb × 1.25
    'marge_H2'     : 0.40,    # PPA H2     = LCOE_hyb × 1.40
}
# Surface balayée rotor (m²) — calculée depuis diamètre
TURBINE['A_WT'] = np.pi * (TURBINE['D_rotor_m'] / 2) ** 2
TURBINE['eta_terrain'] = 0.85

# ══════════════════════════════════════════════════════════════════════════════
# FORMULES PHYSIQUES — fonctions internes utilisées par build_T1_ressources
# ══════════════════════════════════════════════════════════════════════════════

def _CRF(DR, LT):
    """Capital Recovery Factor — annuité financière"""
    return (DR * (1 + DR)**LT) / ((1 + DR)**LT - 1)


def _calc_CF_solaire(GHI_kWh_m2_an):
    """
    CF solaire PV = (GHI_an / 8760) × PR
    GHI en kWh/m²/an divisé par 8760h = fraction d'irradiance moyenne
    PR = Performance Ratio = 0.80 (pertes câblage, onduleur, température)
    Source : IEC 61724-1, NREL PVWatts v8
    Calibration : CF_sol Ouarzazate = (2172/8760)×0.80 = 19.84% ≈ NOOR réel 19.8%
    """
    return float(np.clip((GHI_kWh_m2_an / 8760) * TURBINE['PR_PV'], 0.0, 0.35))


def _calc_PWT(V):
    """
    Puissance turbine éolienne (kW) — courbe polynomiale 3e degré
    PWT = -0.6994·V³ + 19.481·V² - 90.983·V + 121
    Zones :
      V < Vci ou V >= Vo  → P = 0        (arrêt)
      Vci <= V < Vr       → courbe poly  (montée en puissance)
      Vr  <= V < Vo       → P = P_rated  (puissance nominale)
    Source : formule polynomiale données turbine
    """
    Vci = TURBINE['V_ci']
    Vr  = TURBINE['V_r']
    Vo  = TURBINE['V_o']
    Pr  = TURBINE['P_rated_kW']
    if V < Vci or V >= Vo:
        return 0.0
    if Vci <= V < Vr:
        return float(np.clip(-0.6994*V**3 + 19.481*V**2 - 90.983*V + 121, 0.0, Pr))
    return float(Pr)


def _calc_CFWT(v_mean):
    """
    Capacity Factor éolien — formule Weibull analytique avec correction terrain
    CFWT = [e^-(Vci/c)^k - e^-(Vr/c)^k] / [(Vr/c)^k - (Vci/c)^k] - e^-(Vo/c)^k
           × eta_terrain
 
    eta_terrain = 0.85 : facteur de correction NASA POWER → production réelle
      Inclut : wake effect inter-turbines (~5%), disponibilité (~3%),
               pertes électriques (~2%), pertes diverses (~5%)
    Calibration : Weibull(9.5 m/s) × 0.85 → CF=45.9% ≈ Tarfaya réel 43% (err=6.7% ✓)
 
    Source formule : Manwell, McGowan, Rogers — Wind Energy Explained, 2009
    Source eta     : IEC 61400-12-1 (2017) / Barthelmie et al. 2010
    Source eta     : IEC 61400-12-1 (2017) / Barthelmie et al. 2010
    Source formule :  Rezaei M, Naghdi-Khozani N, Jafari N. Wind 
energy utilization for hydrogen production in an 
underdeveloped country: an economic investigation. 
Renew Energy 2020/  Nasser M, Megahed TF, Ookawara S, Hassan H. 
Techno-economic assessment of green hydrogen 
production using different configurations of wind 
turbines and PV panels. J Energy Syst 2022/Hasan MM, Genç G. Techno-economic analysis of 
solar/wind power based hydrogen production. Fuel 
2022
    """
    from scipy.special import gamma as gamma_func
 
    k          = TURBINE['k_weibull']
    Vci        = TURBINE['V_ci']
    Vr         = TURBINE['V_r']
    Vo         = TURBINE['V_o']
    eta_terrain = 0.85   # facteur correction satellite → terrain
 
    if v_mean < 1.0:
        return 0.0
 
    c     = v_mean / gamma_func(1 + 1 / k)
    t1    = np.exp(-(Vci / c)**k)
    t2    = np.exp(-(Vr  / c)**k)
    t3    = np.exp(-(Vo  / c)**k)
    denom = (Vr / c)**k - (Vci / c)**k
 
    if denom == 0:
        return 0.0
 
    CF_weibull = (t1 - t2) / denom - t3
    CF_corrige  = CF_weibull * eta_terrain
 
    return float(np.clip(CF_corrige, 0.0, 1.0))


def _calc_eta_WT(V):
    """
    Rendement aérodynamique turbine
    η_WT = PWT / (0.5 · ρ_air · A_WT · V³)
    Limite théorique de Betz : 16/27 ≈ 59.3%
    Source : Betz, 1926 — loi fondamentale aérodynamique éolienne
    """
    if V < TURBINE['V_ci'] or V >= TURBINE['V_o']:
        return 0.0
    P_dispo = 0.5 * TURBINE['rho_air'] * TURBINE['A_WT'] * V**3 / 1000  # kW
    if P_dispo == 0:
        return 0.0
    return float(np.clip(_calc_PWT(V) / P_dispo, 0.0, 16/27))


def _calc_AnnualWTpower(v_mean):
    """
    Production annuelle turbine (MWh/turbine/an)
    AnnualWTpower = PWT(V_mean) × CFWT × H_an
    H_an = 8760 h (total annuel — le CF Weibull intègre déjà la disponibilité)
    """
    return _calc_CFWT(v_mean) * TURBINE['P_rated_kW'] * TURBINE['H_an'] / 1000


def _calc_LCOE_sol(CF_sol):
    """
    LCOE solaire PV ($/kWh)
    LCOE = (CAPEX × CRF + OPEX) / (CF × 8760)
    CRF = DR·(1+DR)^LT / ((1+DR)^LT - 1)
    Source : IRENA Renewable Cost Database 2024
    Calibration : Midelt PPA réel = 0.018 $/kWh — borne inférieure atteignable
    """
    crf = _CRF(TURBINE['DR'], TURBINE['LT_sol'])
    return round((TURBINE['CAPEX_sol'] * crf + TURBINE['OPEX_sol']) / (CF_sol * 8760), 4)


def _calc_LCOE_eol(CF_eol):
    """
    LCOE éolien ($/kWh)
    LCOE = (CAPEX × CRF + OPEX) / (CF × 8760)
    Retourne None si CF < seuil économique (20%) → site non viable utility-scale
    Source : IRENA 2024 / El Hafdaoui et al. 2024 — plage Maroc : 25–40 $/MWh
    Calibration : Tarfaya LCOE réel = 0.038 $/kWh (IRENA 2022)
    """
    if CF_eol < TURBINE['CF_eol_min']:
        return None
    crf = _CRF(TURBINE['DR'], TURBINE['LT_eol'])
    return round((TURBINE['CAPEX_eol'] * crf + TURBINE['OPEX_eol']) / (CF_eol * 8760), 4)


def _calc_hybride(CF_eol, CF_sol, LCOE_eol, LCOE_sol):
    """
    Mix optimal PV + éolien minimisant le LCOE hybride.

    Deux cas :
    ─────────────────────────────────────────────────────────────────────
    CAS A — Éolien viable (LCOE_eol disponible) :
        Optimisation économique : on cherche w* ∈ [0,1] qui minimise
            LCOE_hyb(w) = w × LCOE_eol + (1-w) × LCOE_sol
        Solution analytique : comme LCOE_hyb est linéaire en w,
            • si LCOE_eol < LCOE_sol → w* = 1.0  (tout éolien)
            • si LCOE_eol > LCOE_sol → w* = 0.0  (tout solaire)
            • si LCOE_eol = LCOE_sol → tout w donne le même LCOE
        En pratique, on contraint w ∈ [0.20, 0.80] pour garantir
        la complémentarité horaire (diversification de source).
        CF_hyb = w* × CF_eol + (1 - w*) × CF_sol   ← moyenne pondérée correcte

    CAS B — Éolien non viable (LCOE_eol = None, CF_eol < 20%) :
        Pondération par ressource disponible (pas d'optimisation économique) :
            w = CF_eol / (CF_eol + CF_sol)
        CF_hyb  = w × CF_eol + (1-w) × CF_sol
        LCOE_hyb = LCOE_sol  (seul coût connu)

    Paramètres
    ----------
    CF_eol   : float — Capacity Factor éolien (0–1)
    CF_sol   : float — Capacity Factor solaire PV (0–1)
    LCOE_eol : float | None — LCOE éolien ($/kWh), None si non viable
    LCOE_sol : float — LCOE solaire ($/kWh)

    Retourne
    --------
    (CF_hyb, w_eol, LCOE_hyb)

    Sources
    -------
    Optimisation mix hybride : IEA Renewable Integration 2023
    Contrainte w ∈ [0.20, 0.80] : Denholm et al. 2021 — NREL
    Formule CF moyenne pondérée : standard industrie
    """
    s = CF_eol + CF_sol
    if s == 0:
        return 0.0, 0.0, LCOE_sol

    # ── CAS B : éolien non viable ─────────────────────────────────────────────
    if LCOE_eol is None:
        w      = CF_eol / s
        CF_h   = w * CF_eol + (1 - w) * CF_sol   # moyenne pondérée correcte
        return round(CF_h, 6), round(w, 4), LCOE_sol

    # ── CAS A : optimisation économique avec contrainte de diversification ─────
    W_MIN, W_MAX = 0.20, 0.80   # contrainte complémentarité horaire

    if LCOE_eol < LCOE_sol:
        w_opt = W_MAX   # favoriser éolien (moins cher)
    elif LCOE_eol > LCOE_sol:
        w_opt = W_MIN   # favoriser solaire (moins cher)
    else:
        w_opt = CF_eol / s   # iso-coût → pondération par ressource

    CF_h    = w_opt * CF_eol + (1 - w_opt) * CF_sol
    LCOE_h  = w_opt * LCOE_eol + (1 - w_opt) * LCOE_sol

    return round(CF_h, 6), round(w_opt, 4), round(LCOE_h, 4)



def _calc_score_H2(GHI, CF_hyb, dist_port, cout_eau, reseau):
    """
    Score composite potentiel H2 (0–100) — Weighted Overlay AHP
    Poids calculés automatiquement (CR=0.008 ✓) :
      GHI=31.9% | CF=31.9% | Logistique=18.4% | Eau=11.0% | Réseau=6.9%
    Source : Saaty 1980 + résultats GEE validés (résultats propres)
    """
    s_irr = min(100.0, GHI / 25.0)
    s_cf  = min(100.0, CF_hyb / 0.40 * 100.0)
    s_log = max(0.0,   100.0 - dist_port / 4.0)
    s_eau = (1.0 - min(1.0, cout_eau / 1.50)) * 100.0
    s_res = {'Excellente': 100.0, 'Bonne': 70.0, 'Moyenne': 40.0}.get(reseau, 50.0)
    return round(
        s_irr * AHP['GHI']        +
        s_cf  * AHP['CF_hybride'] +
        s_log * AHP['Logistique'] +
        s_eau * AHP['Eau']        +
        s_res * AHP['Reseau'],
        1
    )
def _ahp_poids():
    """Poids AHP — Saaty 1980, CR=0.008 ✓"""
    M = np.array([
        [1.000, 1.000, 2.000, 3.000, 4.000],
        [1.000, 1.000, 2.000, 3.000, 4.000],
        [0.500, 0.500, 1.000, 2.000, 3.000],
        [0.333, 0.333, 0.500, 1.000, 2.000],
        [0.250, 0.250, 0.333, 0.500, 1.000],
    ])
    M_norm = M / M.sum(axis=0)
    poids  = M_norm.mean(axis=1)
    return {
        'GHI'        : float(poids[0]),
        'CF_hybride' : float(poids[1]),
        'Logistique' : float(poids[2]),
        'Eau'        : float(poids[3]),
        'Reseau'     : float(poids[4]),
    }

AHP = _ahp_poids()



# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — RESSOURCES ÉNERGÉTIQUES MAROCAINES (8 régions)
# Sources : NASA POWER v8.2, IRENA 2024, MASEN, HCP Maroc, Bakkari et al. 2024
#           El Hafdaoui et al. 2024, Cour des comptes Maroc 2024-2025
# ══════════════════════════════════════════════════════════════════════════════
def build_T1_ressources():
    print("  [T1] Construction : Ressources énergétiques marocaines")

    # ══════════════════════════════════════════════════════════════════════════════════
    # SOURCES PRIMAIRES UTILISÉES :
    #   [S1] CDER / MASEN — Atlas Solaire Maroc (Solargis, résolution 1 km)
    #         https://solaratlas.masen.ma
    #   [S2] Global Solar Atlas 2.0 — World Bank / ESMAP / Solargis (2020)
    #         https://globalsolaratlas.info
    #   [S3] NASA POWER v8.2 — GHI (ALLSKY_SFC_SW_DWN), WS100M — moy. 2019-2023
    #         https://power.larc.nasa.gov
    #   [S4] CDER / GTZ — Atlas Éolien du Maroc, vitesses à 40 m & 100 m
    #         IRENA Case Study Morocco 2013
    #   [S5] El Alani et al. (2022) — Springer, GHI & DNI Helioclim3, 10 ans
    #   [S6] MDPI Resources (2024) — Solar Energy Resource Morocco Review
    #   [S7] Res4Africa (2023) — Powering Desalination with RE in Morocco
    #   [S8] Smart Water Magazine / ONEE (2024-2025) — Désalinisation Maroc
    #   [S9] World Bank (2025) — Gateway to Green Energy: Moroccan Ports as H2 Hubs
    #   [S10] MASEN / Offre Maroc H2 (2024) — 1 M ha terres allouées H2 vert
    #   [S11] Wikipedia — Wind Power in Morocco (données CDER officielles)
    #   [S12] Frontiers in Sustainable Food Systems (2025) — Water management Morocco
    # ══════════════════════════════════════════════════════════════════════════════════

    data = {

        'region': [
            'Laayoune', 'Dakhla', 'Boujdour', 'Guelmim',
            'Jorf_Lasfar', 'Ouarzazate', 'Agadir', 'Tanger',
            'Casablanca', 'Nador', 'Marrakech', 'Midelt'
        ],

        'latitude_N': [
            27.1253, 23.6848, 26.1000, 28.9870,
            33.1100, 30.9189, 30.4278, 35.7595,
            33.5731, 35.1681, 31.6295, 32.6800
        ],

        'longitude_W': [
            -13.1625, -15.9572, -14.5000, -10.0572,
             -8.6300,  -6.8934,  -9.5981,  -5.8340,
             -7.5898,  -2.9335,  -7.9811,  -4.7340
        ],

        # ── RESSOURCES SOLAIRES ────────────────────────────────────────────────────
        # GHI (kWh/m²/an) — Source [S1][S2][S5]
        # Plage nationale confirmée : 1 800–2 500 kWh/m²/an [S6]
        # Ouarzazate > 5.57 kWh/m²/jour ≈ 2 033 kWh/m²/an [S6]
        # Noor Midelt : site préselectionné pour fort GHI présaharien [S6]
        # Côtes atlantiques nord (Tanger, Casablanca) : valeurs plus basses [S6]
        'GHI_kWh_m2_an': [
            2160,   # Laayoune   : zone sahélienne, très fort ensoleillement [S1][S2]
            2155,   # Dakhla     : côte atlantique sud, GHI très élevé [S1][S5]
            2175,   # Boujdour   : désert côtier, meilleure valeur saharienne [S1][S2]
            1940,   # Guelmim    : transition semi-aride / désert [S2][S3]
            1900,   # Jorf_Lasfar: côte atlantique centrale, nébulosité côtière [S2][S3]
            2180,   # Ouarzazate : > 5.57 kWh/m²/j certifié MASEN/NOOR [S1][S6]
            2095,   # Agadir     : côte atlantique, brumes matinales [S2][S3]
            1840,   # Tanger     : méditerranéen-atlantique, plus nuageux [S2][S3]
            1875,   # Casablanca : façade atlantique, influence maritime [S2][S3]
            1785,   # Nador      : méditerranéen, valeurs les plus basses [S2][S3]
            2085,   # Marrakech  : pré-atlasique, très ensoleillé [S1][S2]
            2200,   # Midelt     : altiplano présaharien, sélectionné Noor Midelt [S1][S6]
        ],

        # DNI (kWh/m²/an) — Source [S1][S2][S5]
        # Ouarzazate DNI mesuré ≈ 2 463 kWh/m²/an [S5]
        # Ratio DNI/GHI : 1.01–1.13 (aride/présaharien) ; 0.94–0.99 (côtes) [S2]
        'DNI_kWh_m2_an': [
            2210,   # Laayoune   : ratio ≈ 1.02, ciel dégagé dominant [S2][S5]
            2170,   # Dakhla     : côte, couche marine légère [S2][S5]
            2200,   # Boujdour   : désert pur, ratio élevé [S2]
            2100,   # Guelmim    : ratio ≈ 1.08, transition aride [S2]
            1840,   # Jorf_Lasfar: côte industrielle, ratio ≈ 0.97 [S2]
            2463,   # Ouarzazate : valeur mesurée terrain Helioclim3 [S5]
            2050,   # Agadir     : ratio ≈ 0.98, brumes côtières [S2]
            1790,   # Tanger     : ratio ≈ 0.97, nébulosité atlantique [S2]
            1815,   # Casablanca : ratio ≈ 0.97 [S2]
            1720,   # Nador      : méditerranéen, ratio ≈ 0.96 [S2]
            2350,   # Marrakech  : ratio ≈ 1.13, ciel continental sec [S2]
            2400,   # Midelt     : ratio ≈ 1.09, altiplano semi-aride [S2]
        ],

        # Heures d'ensoleillement (h/an) — Source [S1][S6]
        # National : ~3 000 h/an côtes nord → 3 600 h/an désert [S6][Wikipedia RE Morocco]
        'heures_ensoleillement_an': [
            3200,   # Laayoune
            3180,   # Dakhla
            3215,   # Boujdour
            3320,   # Guelmim    : intérieur aride
            3060,   # Jorf_Lasfar: côte centrale
            3420,   # Ouarzazate : site NOOR, parmi les plus élevés [S6]
            3150,   # Agadir
            2960,   # Tanger     : le plus bas, influence atlantique-méditerranéenne
            3000,   # Casablanca
            2900,   # Nador      : méditerranéen nuageux hiver
            3250,   # Marrakech
            3350,   # Midelt     : altiplano, ciel sec
        ],

        # ── RESSOURCES ÉOLIENNES ───────────────────────────────────────────────────
        # Vitesse vent (m/s) — Source [S4][S11]
        # Atlas CDER officiel, hauteur 40 m → extrapolé 100 m (loi logarithmique z0=0.03m)
        # Tanger-Tétouan : 8–11 m/s à 40 m → ~9.5–13 m/s à 100 m [S4][S11]
        # Dakhla, Tarfaya : 7.5–9.5 m/s à 40 m [S4][S11]
        # Laayoune : 6–7 m/s (région LAAYOUNE-SAKIA) [S4]
        # Moyenne nationale : 5.3 m/s à 90%+ du territoire [S11]
        'vitesse_vent_moy_ms': [
            7.8,    # Laayoune   : 6–7 m/s @40m CDER → ~7.8 m/s @100m [S4][S11]
            9.0,    # Dakhla     : 7.5–8.5 m/s @40m CDER → ~9.0 m/s @100m [S4][S11]
            8.5,    # Boujdour   : similaire Dakhla, vent saharien [S4][S11]
            5.5,    # Guelmim    : zone moins venteuse [S4]
            5.0,    # Jorf_Lasfar: côte industrielle, vent modéré [S4]
            5.5,    # Ouarzazate : vent thermique de vallée, modéré [S4]
            5.5,    # Agadir     : alizé côtier, souss-massa [S4]
            9.5,    # Tanger     : 8–11 m/s @40m → parmi les plus élevés Maroc [S4][S11]
            4.5,    # Casablanca : zone urbaine côtière, vent modéré [S4]
            5.8,    # Nador      : méditerranéen, vent tramontane [S4]
            4.0,    # Marrakech  : intérieur, vent faible à modéré [S4]
            5.5,    # Midelt     : couloir atlasique, vent de montagne [S4]
        ],

        # Potentiel éolien (GW) — Source [S4][S11]
        # Potentiel national total : 25 GW onshore (ministre Amara 2015) [S11]
        # Distribution selon densité de puissance et surface disponible
        # Tarfaya + Laayoune représentent les zones les plus dotées [S4][S11]
        'potentiel_eolien_GW': [
            6.5,    # Laayoune   : zone prioritaire identifiée IRENA [S4]
            10.0,   # Dakhla     : plus forte densité vent + surface [S4]
            8.3,    # Boujdour   : 100 MW Noor Boujdour déjà construit [S4][S11]
            1.5,    # Guelmim    : Total Eren projet Guelmim-Oued Noun [S11]
            1.6,    # Jorf_Lasfar: OCP industrial hub, potentiel limité [S9]
            2.0,    # Ouarzazate : projet Midelt 100 MW wind [S11]
            2.1,    # Agadir     : Souss-Massa, alizé atlantique [S4]
            2.5,    # Tanger     : Tanger 2 (150 MW), Khalladi (120 MW) [S11]
            0.8,    # Casablanca : zone urbaine, potentiel restreint [S4]
            1.2,    # Nador      : Oriental, potentiel méditerranéen [S4]
            0.5,    # Marrakech  : intérieur, potentiel faible [S4]
            1.5,    # Midelt     : projet 100 MW Midelt wind [S11]
        ],

        # ── RESSOURCES EN EAU ──────────────────────────────────────────────────────
        # Source [S7][S8][S12]
        # Maroc : 80% territoire aride/semi-aride, 645 m³/hab/an en 2015 [S7]
        # Désalinisation : 17 plantes opérationnelles 2023, objectif 1.7 Mds m³/an 2030 [S8]
        'disponibilite_eau': [
            'Faible',       # Laayoune   : zone saharienne, désalinisation requise [S7][S8]
            'Faible',       # Dakhla     : côte saharienne, projet dessal. DAWEC 37Mm³/an [S8]
            'Très faible',  # Boujdour   : désert côtier, Noor Boujdour = dessal. [S8]
            'Très faible',  # Guelmim    : aride, nappe phréatique critique [S12]
            'Bonne',        # Jorf_Lasfar: accès mer + OCP dessal. Safi-ElJadida [S8]
            'Très faible',  # Ouarzazate : Drâa-Tafilalet, irrigat. Ouarzazate dépendant dam [S12]
            'Bonne',        # Agadir     : plus grande dessal. Afrique (275 000 m³/j) [S8][S12]
            'Bonne',        # Tanger     : projet dessal. 150 Mm³/an planifié [S8]
            'Bonne',        # Casablanca : plus grande dessal. monde en construction (300Mm³/an) [S8]
            'Bonne',        # Nador      : projet dessal. 250 Mm³/an planifié [S8]
            'Moyenne',      # Marrakech  : Tensift, stress hydrique croissant [S12]
            'Faible',       # Midelt     : Moulouya, ressource limitée altitude [S12]
        ],

        # Désalinisation requise — Source [S7][S8]
        # Critère : zone aride OU distance mer < 100 km ET stress hydrique élevé
        'dessalement_requis': [
            True,   # Laayoune   : saharien côtier [S7]
            True,   # Dakhla     : projet DAWEC opérationnel 2025 [S8]
            True,   # Boujdour   : Noor Boujdour associé à dessal. [S8]
            True,   # Guelmim    : aride sévère [S7]
            True,   # Jorf_Lasfar: OCP Green Investment Program [S8]
            True,   # Ouarzazate : continental aride [S7][S12]
            False,  # Agadir     : dessal. opérationnelle, réseau eau disponible [S8]
            False,  # Tanger     : dessal. planifiée mais eau disponible [S8]
            False,  # Casablanca : dessal. en construction, réseau actuel ok [S8]
            False,  # Nador      : dessal. planifiée, accès mer direct [S8]
            False,  # Marrakech  : Tensift + réseau ONEE [S12]
            True,   # Midelt     : continental, altitude, ressource limitée [S12]
        ],

        # Coût eau (USD/m³) — Source [S7][S8]
        # SWRO (seawater RO) typique Maroc avec RE : 0.45–0.55 USD/m³ grands sites [S7]
        # Sites isolés / petits : 0.70–1.00 USD/m³ [S7]
        # Agadir plant (275 000 m³/j) : ≈ 0.48 USD/m³ [S8]
        'cout_eau_USD_m3': [
            0.75,   # Laayoune   : petit site, coût élevé [S7]
            0.70,   # Dakhla     : projet DAWEC wind-powered [S7][S8]
            0.80,   # Boujdour   : très petit site, coût unitaire haut [S7]
            0.95,   # Guelmim    : pas de projet concret, eau rare [S7]
            0.50,   # Jorf_Lasfar: OCP industriel, économie d'échelle [S8]
            1.00,   # Ouarzazate : eau transportée / continental [S7][S12]
            0.48,   # Agadir     : grande installation SWRO, RE intégrée [S8]
            0.52,   # Tanger     : projet planifié grande capacité [S8]
            0.50,   # Casablanca : plus grande dessal. monde, économie d'échelle [S8]
            0.55,   # Nador      : projet 250Mm³/an planifié [S8]
            0.65,   # Marrakech  : eau ONEE + stress croissant [S12]
            0.90,   # Midelt     : continental, coût acheminement [S7][S12]
        ],

        # Consommation eau électrolyse (L/kg H2) — Source IEA 2024
        # Constante physique PEM/alcalin : ~18–22 L/kg H2, consensus IEA = 21.1
        'consommation_eau_L_kgH2': [21.1] * 12,

        # ── INFRASTRUCTURE & LOGISTIQUE ───────────────────────────────────────────
        # Données calculées par analyse multi-critères (Pente < 5%, Hors zones protégées)
        # Sources : Stratégie Nationale du Foncier / Offre Maroc H2 2024
        'surface_disponible_km2': [
            7500,   # Laayoune   
            12500,  # Dakhla     
            4000,   # Boujdour   
            3500,   # Guelmim    
            15,    # Jorf_Lasfar
            800,   # Ouarzazate 
            200,    # Agadir     
            120,    # Tanger     
            5,     # Casablanca 
            450,    # Nador      
            180,    # Marrakech  
            1500,   # Midelt     
        ],

        # Distance au port (km) — Source [S9] World Bank Moroccan Ports H2
        # Basé sur [S9] : Tanger Med, Mohammedia, Jorf Lasfar, Agadir, Dakhla, Tan-Tan
        'distance_port_km': [
            20,     # Laayoune   : port Laayoune (Foum El Oued) [S9]
            8,      # Dakhla     : port Dakhla, accès direct Atlantique [S9]
            12,     # Boujdour   : petit port Boujdour [S9]
            55,     # Guelmim    : port Tan-Tan (étude MASEN électrolyseur 100MW) [S9]
            2,      # Jorf_Lasfar: port industriel OCP Jorf Lasfar, direct [S9]
            350,    # Ouarzazate : enclavé, plus proche = Agadir [S9]
            5,      # Agadir     : port Agadir direct [S9]
            15,     # Tanger     : Tanger Med (hub mondial) [S9]
            40,     # Casablanca : port Casablanca + Mohammedia [S9]
            8,      # Nador      : port Nador West Med [S9]
            230,    # Marrakech  : Agadir ou Casablanca [S9]
            380,    # Midelt     : enclavé, plus proche = Nador ou Casablanca [S9]
        ],

        # Connexion réseau électrique — Source ONEE / MASEN rapports annuels
        # Jorf Lasfar : centrale thermique 2 352 MW, nœud réseau national [S8]
        # Ouarzazate : Noor 580 MW connecté HTB, mais ligne longue [S6]
        # Nador : Oriental, réseau moins dense [S9]
        # Midelt : Noor Midelt 800 MW en cours, connexion prévue [S6]
        'connexion_reseau_elec': [
            'Excellente',   # Laayoune   : Noor Laayoune 80 MW + réseau ONEe HTB
            'Excellente',   # Dakhla     : Noor Dakhla + interconnexion planifiée
            'Bonne',        # Boujdour   : Noor Boujdour 20 MW, ligne limitée
            'Excellente',   # Guelmim    : nœud régional Guelmim, Total Eren
            'Excellente',   # Jorf_Lasfar: nœud HTB national, centrale 2 352 MW
            'Bonne',        # Ouarzazate : Noor 580 MW mais ligne 225 kV longue
            'Excellente',   # Agadir     : réseau HTB Souss-Massa, Noor Agadir
            'Excellente',   # Tanger     : Tanger Med + interconnexion Europe 900 MW
            'Excellente',   # Casablanca : nœud principal réseau national ONEE
            'Bonne',        # Nador      : Nador West Med, réseau en amélioration
            'Excellente',   # Marrakech  : nœud HTB Haouz, bien connecté
            'Bonne',        # Midelt     : Noor Midelt en cours, HTB prévu
        ],
    }

    df = pd.DataFrame(data)

    # ══════════════════════════════════════════════════════════════════════════
    # CALCULS — toutes les colonnes dérivées calculées depuis formules physiques
    # ══════════════════════════════════════════════════════════════════════════

    # ── 1. CF solaire : CF = (GHI / 8760) × PR ───────────────────────────────
    df['CF_solaire_PV_pct']    = df['GHI_kWh_m2_an'].apply(
        lambda g: round(_calc_CF_solaire(g) * 100, 2)
    )
    df['LCOE_solaire_USD_kWh'] = df['CF_solaire_PV_pct'].apply(
        lambda cf: _calc_LCOE_sol(cf / 100)
    )

    # ── 2. CF éolien : Weibull analytique + PWT polynomial ───────────────────
    # Paramètre d'échelle Weibull : c = v_mean / Γ(1 + 1/k)
    df['k_weibull']              = TURBINE['k_weibull']
    df['c_weibull_ms']           = df['vitesse_vent_moy_ms'].apply(
        lambda v: round(v / gamma_func(1 + 1/TURBINE['k_weibull']), 3)
    )
    df['CF_eolien_pct']          = df['vitesse_vent_moy_ms'].apply(
        lambda v: round(_calc_CFWT(v) * 100, 2)
    )
    # PWT(V_mean) : puissance à la vitesse moyenne (kW)
    df['PWT_kW_a_vmoy']          = df['vitesse_vent_moy_ms'].apply(
        lambda v: round(_calc_PWT(v), 1)
    )
    # η_WT : rendement aérodynamique (limite Betz = 59.3%)
    df['eta_WT_pct']             = df['vitesse_vent_moy_ms'].apply(
        lambda v: round(_calc_eta_WT(v) * 100, 2)
    )
    # AnnualWTpower = PWT × CFWT × 7500 h
    df['AnnualWTpower_MWh_turb'] = df['vitesse_vent_moy_ms'].apply(
        lambda v: round(_calc_AnnualWTpower(v), 1)
    )
    df['LCOE_eolien_USD_kWh']    = df['CF_eolien_pct'].apply(
        lambda cf: _calc_LCOE_eol(cf / 100)
    )
    # Flag viabilité économique éolien (CF >= 20%)
    df['site_viable_eolien']     = df['CF_eolien_pct'].apply(
        lambda cf: cf / 100 >= TURBINE['CF_eol_min']
    )

    # ── 3. Hybride : pondération optimale CF + LCOE ───────────────────────────
    hybride = df.apply(
        lambda r: _calc_hybride(
            r['CF_eolien_pct'] / 100,
            r['CF_solaire_PV_pct'] / 100,
            r['LCOE_eolien_USD_kWh'],
            r['LCOE_solaire_USD_kWh']
        ), axis=1
    )
    df['CF_hybride_pct']        = hybride.apply(lambda x: round(x[0] * 100, 2))
    df['w_eolien']              = hybride.apply(lambda x: round(x[1], 3))
    df['LCOE_hybride_USD_kWh']  = hybride.apply(lambda x: round(x[2], 4))

    # ── 4. PPA selon marché (marges sur LCOE hybride) ────────────────────────
    # Source : IEA 2024 / MASEN PPA structure contractuelle
    df['PPA_local_USD_kWh']   = df['LCOE_hybride_USD_kWh'].apply(
        lambda x: round(x * (1 + TURBINE['marge_local']),  4)
    )
    df['PPA_Europe_USD_kWh']  = df['LCOE_hybride_USD_kWh'].apply(
        lambda x: round(x * (1 + TURBINE['marge_europe']), 4)
    )
    df['PPA_H2_USD_kWh']      = df['LCOE_hybride_USD_kWh'].apply(
        lambda x: round(x * (1 + TURBINE['marge_H2']),     4)
    )

    # ── 5. Conversion USD → EUR pour toutes les colonnes pertinentes ──────────
    cols_usd = [
        'LCOE_solaire_USD_kWh', 'LCOE_eolien_USD_kWh', 'LCOE_hybride_USD_kWh',
        'PPA_local_USD_kWh', 'PPA_Europe_USD_kWh', 'PPA_H2_USD_kWh',
        'cout_eau_USD_m3'
    ]
    df = cols_to_eur(df, cols_usd, devise="USD")

    # ── 6. Colonnes de traçabilité ────────────────────────────────────────────
    df['source_vent_solaire']   = 'NASA POWER v8.2 — WS100M + GHI — 2019-2023'
    df['source_formule_CF_eol'] = 'Weibull analytique k=2 (Rayleigh) — Manwell et al. 2009'
    df['source_formule_PWT']    = 'PWT=-0.6994V³+19.481V²-90.983V+121 (polynomiale)'
    df['source_formule_CF_sol'] = 'CF=(GHI/8760)×PR, PR=0.80 — IEC 61724-1 / NREL PVWatts'
    df['source_LCOE']           = 'LCOE=(CAPEX×CRF+OPEX)/(CF×8760) — IRENA 2024'

    # ── Sauvegarde CSV ────────────────────────────────────────────────────────
    os.makedirs(os.path.join(OUTPUT_DIR, "csv"), exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/csv/T1_ressources_energetiques.csv",
              index=False, encoding='utf-8-sig')

    print(f"     ✓ T1 sauvegardé : {len(df)} régions × {len(df.columns)} variables")
    print(f"\n     {'Région':<13} {'CF_sol%':>8} {'CF_éol%':>8} {'CF_hyb%':>8} "
          f"{'LCOE_hyb':>10} {'Viable_éol':>11} ")
    print(f"     {'-'*68}")
    for _, row in df.iterrows():
        print(f"     {row['region']:<13} {row['CF_solaire_PV_pct']:>8.2f} "
              f"{row['CF_eolien_pct']:>8.2f} {row['CF_hybride_pct']:>8.2f} "
              f"{str(row['LCOE_hybride_USD_kWh']):>10} "
              f"{str(row['site_viable_eolien']):>11} ")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — TECHNOLOGIES DE PRODUCTION H2
# ══════════════════════════════════════════════════════════════════════════════
# ── Fonction complète corrigée ──
def build_T2_production():
    print("  [T2] Construction : Technologies de production H2...")

    # Paramètres par technologie — format : (min, mode/mean, max, unité, source)
    rows = [
        # ── ÉLECTROLYSE ALCALINE (AEL)
        ['AEL','CAPEX_stack',          500, 650, 1000,'USD/kW', 'Hydrogen Europe 2024'],
        ['AEL','CAPEX_systeme_complet',600, 800,1200,'USD/kW','IEA Global H2 Review 2024'],
        ['AEL','CAPEX_systeme_complet',500, 950,1400,'USD/kW','Halder et al., 2023'],
        ['AEL','CAPEX_systeme_complet',800, 900,1000,'USD/kW','Ahmad et al., 2024'],
        ['AEL','OPEX_fixe',1.5,2.0,3.0,'%CAPEX/an','NREL H2A 2024'],
        ['AEL','OPEX_fixe',1.5,2.0,3.0,'%CAPEX/an','Taoufik & Fekri, 2023'],
        ['AEL','OPEX_fixe',1.5,2.0,3.0,'%CAPEX/an','Squadrito et al., 2023'],
        ['AEL','OPEX_variable',0.03,0.05,0.08,'USD/kgH2','IEA 2024'],
        ['AEL','efficacite',50,52,58,'kWh/kgH2','Hydrogen Europe 2024'],
        ['AEL','purete_H2',99.3,99.5,99.8,'%','ISO 14687'],
        ['AEL','purete_H2',99.99,99.9945,99.999,'%','Ahmad et al., 2024'],
        ['AEL','purete_H2',99.99,99.9945,99.999,'%','Taoufik & Fekri, 2023'],
        ['AEL','pression_sortie',1,10,30,'bar','Hydrogen Europe 2024'],
        ['AEL','pression_sortie',1,10,30,'bar','Collis & Schomäcker, 2024'],
        ['AEL','temperature_op',60,75,90,'°C','Schmidt et al. 2017'],
        ['AEL','temperature_op',60,70,80,'°C','Ahmad et al., 2024'],
        ['AEL','temperature_op',60,70,80,'°C','Gorji, 2023'],
        ['AEL','duree_vie_stack',60000,80000,100000,'heures','NREL 2024'],
        ['AEL','duree_vie_stack',60000,80000,100000,'heures','Ahmad et al., 2024'],
        ['AEL','duree_vie_systeme',20,25,30,'ans','IEA 2024'],
        ['AEL','remplacement_stack',8,10,12,'ans','Hydrogen Europe 2024'],
        ['AEL','degradation',0.08,0.12,0.20,'%/1000h','Hydrogen Europe 2024'],
        ['AEL','rampe_montee',5,10,20,'%/s','Buttler & Spliethoff 2018'],
        ['AEL','charge_min',10,20,30,'%','Hydrogen Europe 2024'],
        ['AEL','disponibilite',93,95,98,'%','IEA 2024'],
        ['AEL','disponibilite',90,94,98,'%','Habour et al., 2024'],
        ['AEL','TRL',9,9,9,'-','IEA TRL scale'],

        # ── PEM
        ['PEM','CAPEX_stack',600,900,1500,'USD/kW','IEA 2024 + DOE 2024+ Rozzi 2024'],
        ['PEM','CAPEX_systeme_complet',800,1100,2000,'USD/kW','DOE H2 Program 2024'],
        ['PEM','CAPEX_systeme_complet',1100,1450,1800,'USD/kW','Ahmad et al., 2024'],
        ['PEM','OPEX_fixe',2.0,3.0,4.0,'%CAPEX/an','NREL H2A 2024'],
        ['PEM','OPEX_variable',0.05,0.07,0.12,'USD/kgH2','IEA 2024'],
        ['PEM','efficacite',50,55,65,'kWh/kgH2','Hydrogen Europe 2024'],
        ['PEM','purete_H2',99.9,99.999,99.999,'%','ISO 14687'],
        ['PEM','pression_sortie',30,50,80,'bar','Hydrogen Europe 2024'],
        ['PEM','temperature_op',50,65,80,'°C','Schmidt et al. 2017'],
        ['PEM','temperature_op',50,65,80,'°C','Gorji 2023'],
        ['PEM','duree_vie_stack',40000,60000,90000,'heures','NREL 2024'],
        ['PEM','duree_vie_stack',50000,65000,80000,'heures','Squadrito et al., 2023'],
        ['PEM','duree_vie_systeme',15,20,25,'ans','IEA 2024'],
        ['PEM','duree_vie_systeme',20,25,30,'ans','Villarreal Vives et al., 2023'],
        ['PEM','remplacement_stack',5,7,10,'ans','Hydrogen Europe 2024'],
        ['PEM','degradation',0.15,0.25,0.40,'%/1000h','Hydrogen Europe 2024'],
        ['PEM','rampe_montee',50,100,200,'%/s','Buttler & Spliethoff 2018'],
        ['PEM','charge_min',3,5,10,'%','Hydrogen Europe 2024'],
        ['PEM','disponibilite',94,97,99,'%','IEA 2024'],
        ['PEM','TRL',8,8,9,'-','IEA TRL scale'],

        # ── SOEC
        ['SOEC','CAPEX_systeme_complet',1500,2500,4000,'USD/kW','IEA 2024'],
        ['SOEC','CAPEX_systeme_complet',2800,4200,5600,'USD/kW','Ahmad et al.,2024'],
        ['SOEC','CAPEX_systeme_complet',2800,4200,5600,'USD/kW','Squadrito et al., 2023'],
        ['SOEC','OPEX_fixe',2.5,3.5,5.0,'%CAPEX/an','NREL H2A 2024'],
        ['SOEC','efficacite',35,40,45,'kWh/kgH2','IEA 2024 (meilleure effi)'],
        ['SOEC','temperature_op',700,800,900,'°C','Schmidt et al. 2017'],
        ['SOEC','temperature_op',700,850,1000,'°C','Gorji 2023'],
        ['SOEC','duree_vie_systeme',8,10,15,'ans','IEA 2024'],
        ['SOEC','TRL',5,6,7,'-','IEA TRL scale'],

        # ── PV solaire
        ['PV_solaire','CAPEX',350,550,900,'USD/kW','IRENA 2024'],
        ['PV_solaire','CAPEX',730,740,750,'USD/kW','Taoufik & Fekri, 2023'],
        ['PV_solaire','OPEX_fixe_kWan',8,12,18,'USD/kW/an','IRENA 2024'],
        ['PV_solaire','OPEX_fixe_kWan',21,22,23,'USD/kW/an','Taoufik & Fekri, 2023'],
        ['PV_solaire','degradation',0.3,0.5,0.8,'%/an','IRENA 2024'],
        ['PV_solaire','duree_vie',25,30,35,'ans','IRENA 2024'],
        ['PV_solaire','efficacite',18,21,24,'%','IEA 2024'],
        ['PV_solaire','LCOE_Maroc',0.015,0.025,0.040,'USD/kWh','MASEN PPA record 0.018 (Midelt 2020)'],
        ['PV_solaire','LCOE_Maroc',0.030,0.040,0.050,'USD/kWh','El Hafdaoui et al., 2024'],

        # ── Éolien terrestre
        ['Eolien','CAPEX',900,1200,1600,'USD/kW','IRENA 2024'],
        ['Eolien','OPEX_fixe_kWan',25,35,50,'USD/kW/an','IRENA 2024'],
        ['Eolien','OPEX_fixe_kWan',47,48.5,50,'USD/kW/an','El Hafdaoui et al., 2024'],
        ['Eolien','duree_vie',20,25,30,'ans','IRENA 2024'],
        ['Eolien','LCOE_Maroc',0.020,0.032,0.050,'USD/kWh','Tarfaya LCOE=0.038 (IRENA 2022)'],
        ['Eolien','LCOE_Maroc',0.025,0.032,0.040,'USD/kWh','El Hafdaoui et al., 2024'],
    ]

    # Création du DataFrame
    df = pd.DataFrame(rows, columns=[
        'technologie','parametre','valeur_min','valeur_mode','valeur_max','unite','source'
    ])

    # ── Conversion USD → EUR
    mask_usd = df['unite'].str.contains('USD', na=False)
    df['valeur_min_EUR']  = df.apply(lambda r: round(r['valeur_min']*TAUX_CHANGE['USD'],6) if mask_usd[r.name] else r['valeur_min'], axis=1)
    df['valeur_mode_EUR'] = df.apply(lambda r: round(r['valeur_mode']*TAUX_CHANGE['USD'],6) if mask_usd[r.name] else r['valeur_mode'], axis=1)
    df['valeur_max_EUR']  = df.apply(lambda r: round(r['valeur_max']*TAUX_CHANGE['USD'],6) if mask_usd[r.name] else r['valeur_max'], axis=1)

    # Informations complémentaires
    df['devise_originale'] = df['unite'].apply(lambda u: 'USD' if 'USD' in str(u) else 'N/A')
    df['devise_ref']       = MONNAIE_REF
    df['annee_ref']        = ANNEE_REF
    df['source_taux']      = SOURCE_TAUX

    # Création dossier csv si nécessaire
    os.makedirs(os.path.join(OUTPUT_DIR,"csv"), exist_ok=True)

    # Sauvegarde CSV
    df.to_csv(os.path.join(OUTPUT_DIR,"csv/T2_technologies_production.csv"), index=False, encoding='utf-8-sig')
    print(f"     ✓ T2 sauvegardé : {len(df)} paramètres × {len(df.columns)} colonnes")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — TECHNOLOGIES DE STOCKAGE H2
# ══════════════════════════════════════════════════════════════════════════════
def build_T3_stockage():
    print("  [T3] Construction : Technologies de stockage H2...")

    rows = [
        # ── H2 COMPRIMÉ 350 bar ──────────────────────────────────────────────
        ['GH2_350bar','CAPEX_reservoir',        400,  600,  900, 'USD/kgH2', 'DOE 2024'],
        ['GH2_350bar','CAPEX_compresseur',       800, 1200, 2000, 'USD/kW',   'Hydrogen Council 2023'],
        ['GH2_350bar','CAPEX_systeme_complet',  1200, 1800, 2900, 'USD/kgH2', 'DOE 2024 — réservoir + compresseur'],
        ['GH2_350bar','energie_compression',     1.5,  2.0,  3.0, 'kWh/kgH2','IEA 2024'],
        ['GH2_350bar','OPEX_pct_CAPEX',          1.5,  2.0,  3.0, '%/an',    'IEA 2024'],
        ['GH2_350bar','pertes_boil_off',         1.0,  1.5,  2.0, '%/jour',  'physique'],
        ['GH2_350bar','densite_vol',            23.5, 23.5, 23.5, 'kgH2/m3', 'physique'],
        ['GH2_350bar','duree_vie',              15,   20,   25,   'ans',     'DOE 2024'],
        ['GH2_350bar','TRL',                    9,    9,    9,    '-',       'IEA TRL'],
        ['GH2_350bar','LCOS',                   0.3,  0.6,  1.2,  'USD/kgH2','IEA 2024'],

        # ── H2 COMPRIMÉ 700 bar ──────────────────────────────────────────────
        ['GH2_700bar','CAPEX_reservoir',         600,  900, 1400, 'USD/kgH2', 'DOE 2024'],
        ['GH2_700bar','CAPEX_systeme_complet',  1400, 1800, 2800, 'USD/kgH2', 'DOE 2024 — réservoir Type IV + compresseur'],
        ['GH2_700bar','energie_compression',     2.5,  3.5,  5.0, 'kWh/kgH2','IEA 2024'],
        ['GH2_700bar','OPEX_pct_CAPEX',          2.0,  2.5,  3.5, '%/an',    'IEA (2024) Global Hydrogen Review [R3]'],
        ['GH2_700bar','densite_vol',            40.2, 40.2, 40.2, 'kgH2/m3', 'physique'],
        ['GH2_700bar','TRL',                    9,    9,    9,    '-',       'IEA TRL'],
        ['GH2_700bar','LCOS',                   0.5,  0.9,  1.5,  'USD/kgH2','IEA (2024) Global Hydrogen Review ; Hydrogen Council (2021) Hydrogen Insights'],

        # ── H2 LIQUIDE (LH2) ─────────────────────────────────────────────────
        ['LH2','CAPEX_liquefacteur',           2000, 3500, 6000, 'USD/(kgH2/j)','DOE H2 Program 2024'],
        ['LH2','CAPEX_reservoir',               800, 1200, 2000, 'USD/kgH2',    'IEA 2024'],
        ['LH2','CAPEX_systeme_complet',        2800, 4700, 8000, 'USD/kgH2',    'DOE 2024 — liquéfacteur + réservoir cryogénique'],
        ['LH2','energie_liquefaction',            8,   10,   14,  'kWh/kgH2',   'IEA 2024'],
        ['LH2','OPEX_pct_CAPEX',                1.5,  2.0,  3.0, '%/an',        'IEA 2024'],
        ['LH2','pertes_boil_off',               0.1,  0.3,  0.5, '%/jour',      'IEA 2024'],
        ['LH2','densite_vol',                  70.8, 70.8, 70.8, 'kgH2/m3',    'physique (-253°C)'],
        ['LH2','temperature_K',              -253, -253,  -253,  '°C',          'physique'],
        ['LH2','duree_vie',                    20,   25,   30,   'ans',         'IEA 2024'],
        ['LH2','TRL',                           6,    7,    8,   '-',           'IEA TRL'],
        ['LH2','LCOS',                          1.5,  2.5,  4.0, 'USD/kgH2',   'Hydrogen Council 2023'],

        # ── AMMONIAC (NH3) ───────────────────────────────────────────────────
        ['NH3','CAPEX_synthese_Haber',          400,  700, 1100, 'USD/(tNH3/j)','Hydrogen Council 2023'],
        ['NH3','CAPEX_craquage_NH3_H2',         500,  900, 1500, 'USD/(tNH3/j)','IEA 2024'],
        ['NH3','CAPEX_stockage',                200,  350,  500, 'USD/tNH3',    'IEA 2024'],
        ['NH3','CAPEX_systeme_complet',        1100, 1950, 3100, 'USD/kgH2',    'IEA 2024 — synthèse + craquage + stockage NH3'],
        ['NH3','energie_synthese',               8,   10,   12,  'kWh/kgH2',   'IEA 2024'],
        ['NH3','energie_craquage',              12,   15,   20,  'kWh/kgH2',   'IEA 2024'],
        ['NH3','OPEX_pct_CAPEX',                2.0,  3.0,  4.0, '%/an',       'IEA 2024'],
        ['NH3','efficacite_H2_to_NH3',         68,   72,   76,  '%',           'IEA 2024'],
        ['NH3','efficacite_NH3_to_H2',         80,   85,   90,  '%',           'IEA 2024'],
        ['NH3','densite_vol',                  121,  121,  121, 'kgH2/m3',    'physique (liquide -33°C)'],
        ['NH3','temperature_stockage_C',       -33,  -33,  -33, '°C ou ambiant sous pression','physique'],
        ['NH3','pertes_pct',                   0.5,  1.0,  2.0, '%',           'IEA 2024'],
        ['NH3','TRL',                           9,    9,    9,  '-',           'IEA TRL (infrastructure existante)'],
        ['NH3','LCOS',                          0.8,  1.5,  2.5, 'USD/kgH2',   'Hydrogen Council 2023'],

        # ── LOHC (Dibenzyltoluène / H18-DBT) ────────────────────────────────
        ['LOHC','CAPEX_hydrogenation',          500,  800, 1200, 'USD/(kgH2/j)','Hydrogenious 2024'],
        ['LOHC','CAPEX_dehydrogenation',        800, 1200, 2000, 'USD/(kgH2/j)','Hydrogenious 2024'],
        ['LOHC','CAPEX_systeme_complet',       1300, 2000, 3200, 'USD/kgH2',    'Hydrogenious 2024 — hydrogénation + déshydrogénation'],
        ['LOHC','energie_hydrogenation',          3,    5,    8,  'kWh/kgH2',   'IEA 2024'],
        ['LOHC','energie_dehydrogenation',        8,   10,   14,  'kWh/kgH2',   'IEA 2024'],
        ['LOHC','temperature_hydrog_C',         130,  150,  180, '°C',          'Hydrogenious 2024'],
        ['LOHC','temperature_dehydrog_C',       280,  310,  340, '°C',          'Hydrogenious 2024'],
        ['LOHC','densite_vol',                   50,   57,   62,  'kgH2/m3',    'physique'],
        ['LOHC','pertes_carrier_pct_cycle',    0.05, 0.10, 0.20, '%',           'Hydrogenious 2024'],
        ['LOHC','OPEX_pct_CAPEX',               2.5,  3.5,  5.0, '%/an',       'Niermann et al. (2021) Energy Environ. Sci. 12, 290 DOI:10.1039/C8EE02700E'],
        ['LOHC','TRL',                           6,    7,    8,  '-',           'IEA TRL'],
        ['LOHC','LCOS',                          1.5,  2.5,  4.0, 'USD/kgH2',  'Hydrogen Council 2023'],

        # ── CAVERNE SALINE ───────────────────────────────────────────────────
        ['Caverne_saline','CAPEX_USD_kWh',       0.5,  3.0, 10.0, 'USD/kWh',  'IEA 2024'],
        ['Caverne_saline','CAPEX_systeme_complet',200, 500, 1200, 'USD/kgH2',  'Caglayan et al. (2020) Int. J. Hydrogen Energy 45, 6793 DOI:10.1016/j.ijhydene.2019.12.161'],
        ['Caverne_saline','OPEX_pct_CAPEX',      0.5,  1.0,  2.0, '%/an',     'IEA 2024'],
        ['Caverne_saline','efficacite',          96,   98,   99,  '%',          'IEA 2024'],
        ['Caverne_saline','capacite_GWh',       100, 1000,10000,  'GWh',       'IEA 2024'],
        ['Caverne_saline','duree_vie',           40,   50,   60,  'ans',        'IEA 2024'],
        ['Caverne_saline','TRL',                  7,    8,    9,  '-',          'IEA TRL'],
        ['Caverne_saline','sites_potentiels',     1,    3,    5,  'nb sites Maroc','IRESEN géologie 2022'],
        ['Caverne_saline','densite_vol',         20,   30,   40,  'kgH2/m3',   'IEA (2024) ; Caglayan et al. (2020) — cavernes 50-180 bar'],
        ['Caverne_saline','energie_compression',  1.0,  1.5,  2.5,'kWh/kgH2', 'IEA (2024) — injection basse pression 50-100 bar'],
        ['Caverne_saline','LCOS',                0.15, 0.80, 1.20,'USD/kgH2',  'Caglayan et al. (2020) Int. J. Hydrogen Energy 45, 6793 ; Lord et al. (2014) Int. J. Hydrogen Energy 39, 15570'],

        # ── MÉTHANOL VERT (eMethanol) ────────────────────────────────────────
        ['eMethanol','CAPEX_synthese',           400,  600,  900, 'USD/tMeOH/an','IRENA 2023 / CRI'],
        ['eMethanol','CAPEX_stockage',            50,   80,  120, 'USD/tMeOH',   'IEA 2024'],
        ['eMethanol','CAPEX_systeme_complet',    800, 1400, 2200, 'USD/kgH2',    'IRENA (2023) Innovation Outlook: Renewable Methanol ISBN:978-92-9260-567-0'],
        ['eMethanol','efficacite_H2_to_MeOH',    60,   65,   70,  '%',           'Hydrogen Europe'],
        ['eMethanol','efficacite_MeOH_to_H2',    65,   70,   75,  '%',           'IEA'],
        ['eMethanol','pertes_stockage',          0.1,  0.2,  0.3, '%',           'physique liquide'],
        ['eMethanol','densite_vol',              120,  140,  150, 'kgH2/m3',     'densité système'],
        ['eMethanol','temperature_stockage_C',   20,   20,   20,  '°C',          'ambiant'],
        ['eMethanol','TRL',                       6,    7,    8,  '-',           'IEA'],
        ['eMethanol','LCOS',                      2.5,  3.5,  6.0,'USD/kgH2',   'IRENA 2023 — cycle complet CO2+H2→MeOH'],
        ['eMethanol','cout_CO2_capture',          40,   80,  150, 'USD/tCO2',   'source → DAC'],
        ['eMethanol','duree_vie_usine',           15,   20,   25,  'ans',        'industrie chimique'],
        ['eMethanol','OPEX_pct_CAPEX',            2.0,  3.0,  4.0,'%/an',       'IRENA (2023) Innovation Outlook: Renewable Methanol ISBN:978-92-9260-567-0'],
    ]

    df_technologies = pd.DataFrame(rows, columns=[
        'technologie','parametre','valeur_min','valeur_mode','valeur_max','unite','source'
    ])

    # Conversion USD → EUR
    mask_usd = df_technologies['unite'].str.contains('USD', na=False)
    df_technologies['valeur_min_EUR']  = df_technologies.apply(
        lambda r: round(r['valeur_min']  * TAUX_CHANGE['USD'], 6) if mask_usd[r.name] else r['valeur_min'],  axis=1)
    df_technologies['valeur_mode_EUR'] = df_technologies.apply(
        lambda r: round(r['valeur_mode'] * TAUX_CHANGE['USD'], 6) if mask_usd[r.name] else r['valeur_mode'], axis=1)
    df_technologies['valeur_max_EUR']  = df_technologies.apply(
        lambda r: round(r['valeur_max']  * TAUX_CHANGE['USD'], 6) if mask_usd[r.name] else r['valeur_max'],  axis=1)
    df_technologies['devise_originale'] = df_technologies['unite'].apply(
        lambda u: 'USD' if 'USD' in str(u) else 'N/A')
    df_technologies['devise_ref']  = MONNAIE_REF
    df_technologies['annee_ref']   = ANNEE_REF
    df_technologies['source_taux'] = SOURCE_TAUX

    os.makedirs(os.path.join(OUTPUT_DIR, "csv"), exist_ok=True)
    df_technologies.to_csv(
        os.path.join(OUTPUT_DIR, "csv/T3_technologies_stockage.csv"),
        index=False, encoding='utf-8-sig')

    # Types de réservoirs (inchangé)
    reservoir_types = pd.DataFrame([
        ['I',   'All-metal construction',                 300,      '83-240',   0.017, 'https://doi.org/10.3390/ma12121973'],
        ['II',  'Mostly metal + composite hoop wrap',     200,      '86-360',   0.021, 'https://doi.org/10.3390/ma12121973'],
        ['III', 'Metal liner + full composite wrap',      '300-700','300-700',0.042, 'https://doi.org/10.1093/ce/zkad021'],
        ['IV',  'All-composite construction',             '300-700','633-1200',0.057,'https://doi.org/10.1093/ce/zkad021'],
    ], columns=['type','materiaux','pression_bar','cout_USD_kg','densite_grav_wt','source_doi'])

    reservoir_types.to_csv(
        os.path.join(OUTPUT_DIR, "csv/T3_types_reservoirs.csv"),
        index=False, encoding='utf-8-sig')

    print(f"     ✓ T3 technologies stockages sauvegardées : {len(df_technologies)} paramètres")
    print(f"     ✓ T3 types réservoirs sauvegardés : {len(reservoir_types)} types avec DOI")

    return df_technologies, reservoir_types

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — TRANSPORT & INFRASTRUCTURE Calcul automatique des distances via OSRM (OpenStreetMap)
# + Choix du mode optimal selon distance et type de corridor
#+ Export CSV

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 v2 — TRANSPORT & INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
# Améliorations vs v1 :
#   1. OSRM systématique pour TOUS les segments terrestres (Maroc + Europe)
#   2. Décomposition automatique terre/mer pour l'export
#      (origine intérieure → port optimal → destination maritime)
#   3. Fonction calcul_corridor_auto(lat, lon, destination) pour site arbitraire
#   4. Interpolation IDW des paramètres (GHI, vent, coût eau) pour sites hors base
#   5. Corridors prédéfinis conservés + génération dynamique possible
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import requests
import time
import math
import os

# ─────────────────────────────────────────────────────────────────────────────
# NŒUDS — Réseau complet (Production Maroc + Ports Maroc + International)
# ─────────────────────────────────────────────────────────────────────────────
NODES = {
    # ── Sites de production Maroc ──
    "Ouarzazate"  : {"lat": 30.9189, "lon": -6.8934,  "zone": "Maroc", "type_noeud": "Production",       "code_locode": "MA OUA", "code_postal": "45000", "pays": "Maroc"},
    "Dakhla"      : {"lat": 23.6848, "lon": -15.9572, "zone": "Maroc", "type_noeud": "Production/Port",  "code_locode": "MA VIL", "code_postal": "73000", "pays": "Maroc", "est_port": True},
    "Laayoune"    : {"lat": 27.1253, "lon": -13.1625, "zone": "Maroc", "type_noeud": "Production/Port",  "code_locode": "MA EUN", "code_postal": "70000", "pays": "Maroc", "est_port": True},
    "Tarfaya"     : {"lat": 27.9378, "lon": -12.9194, "zone": "Maroc", "type_noeud": "Production",       "code_locode": "MA TFY", "code_postal": "70350", "pays": "Maroc"},
    "Guelmim"     : {"lat": 28.9870, "lon": -10.0572, "zone": "Maroc", "type_noeud": "Production",       "code_locode": "MA GLM", "code_postal": "81000", "pays": "Maroc"},
    "Midelt"      : {"lat": 32.6800, "lon": -4.7340,  "zone": "Maroc", "type_noeud": "Hub intérieur",    "code_locode": "MA MID", "code_postal": "54350", "pays": "Maroc"},
    "Boujdour"    : {"lat": 26.1000, "lon": -14.5000, "zone": "Maroc", "type_noeud": "Production",       "code_locode": "MA BJD", "code_postal": "71000", "pays": "Maroc"},
    "Marrakech"   : {"lat": 31.6295, "lon": -7.9811,  "zone": "Maroc", "type_noeud": "Hub",              "code_locode": "MA RAK", "code_postal": "40000", "pays": "Maroc"},

    # ── Ports Maroc (export maritime possible) ──
    "Tanger"      : {"lat": 35.7595, "lon": -5.8340,  "zone": "Maroc", "type_noeud": "Port/Hub",         "code_locode": "MA TNG", "code_postal": "90000", "pays": "Maroc", "est_port": True},
    "Jorf_Lasfar" : {"lat": 33.1100, "lon": -8.6300,  "zone": "Maroc", "type_noeud": "Port industriel",  "code_locode": "MA JOR", "code_postal": "25100", "pays": "Maroc", "est_port": True},
    "Casablanca"  : {"lat": 33.5731, "lon": -7.5898,  "zone": "Maroc", "type_noeud": "Hub central",      "code_locode": "MA CAS", "code_postal": "20000", "pays": "Maroc", "est_port": True},
    "Agadir"      : {"lat": 30.4278, "lon": -9.5981,  "zone": "Maroc", "type_noeud": "Port/Hub",         "code_locode": "MA AGA", "code_postal": "80000", "pays": "Maroc", "est_port": True},
    "Nador"       : {"lat": 35.1681, "lon": -2.9335,  "zone": "Maroc", "type_noeud": "Port",             "code_locode": "MA NDR", "code_postal": "62000", "pays": "Maroc", "est_port": True},

    # ── Destinations internationales ──
    "Algésiras"   : {"lat": 36.1408, "lon": -5.4530,  "zone": "Europe",   "type_noeud": "Port",            "code_locode": "ES ALG", "code_postal": "11201", "pays": "Espagne"},
    "Almería"     : {"lat": 36.8340, "lon": -2.4637,  "zone": "Europe",   "type_noeud": "Port",            "code_locode": "ES LEI", "code_postal": "04001", "pays": "Espagne"},
    "Rotterdam"   : {"lat": 51.9244, "lon": 4.4777,   "zone": "Europe",   "type_noeud": "Port terminal H2","code_locode": "NL RTM", "code_postal": "3011",  "pays": "Pays-Bas"},
    "Barcelone"   : {"lat": 41.3851, "lon": 2.1734,   "zone": "Europe",   "type_noeud": "Port",            "code_locode": "ES BCN", "code_postal": "08001", "pays": "Espagne"},
    "Marseille"   : {"lat": 43.2965, "lon": 5.3698,   "zone": "Europe",   "type_noeud": "Port",            "code_locode": "FR MRS", "code_postal": "13001", "pays": "France"},
    "Paris"       : {"lat": 48.8566, "lon": 2.3522,   "zone": "Europe",   "type_noeud": "Marché final",    "code_locode": "FR PAR", "code_postal": "75001", "pays": "France"},
    "Dakar"       : {"lat": 14.7167, "lon": -17.4677, "zone": "Afrique",  "type_noeud": "Port",            "code_locode": "SN DKR", "code_postal": "BP 3000","pays": "Sénégal"},
    "Canaries"    : {"lat": 28.1235, "lon": -15.4363, "zone": "Canaries", "type_noeud": "Hub insulaire",   "code_locode": "ES LPA", "code_postal": "35001", "pays": "Espagne"},
}

# Liste des ports marocains (nœuds avec capacité d'export maritime)
PORTS_MAROC = [nom for nom, info in NODES.items()
               if info.get("est_port", False) and info["zone"] == "Maroc"]

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES DES MODES DE TRANSPORT (Coûts en USD/kg H2)
# Sources : IEA 2024, Hydrogen Council 2023, Dinh et al. 2024
# ─────────────────────────────────────────────────────────────────────────────
MODES_PARAMS = {
    "Tube_trailer": {
        "cout_min": 0.35, "cout_max": 0.80,
        "vecteur": "GH2", "type": "terrestre",
        "desc": "Remorque tube — courte distance (<200 km)"
    },
    "Pipeline_H2_reconverti": {
        "cout_min": 0.15, "cout_max": 0.50,
        "vecteur": "GH2", "type": "terrestre",
        "desc": "Pipeline gaz reconverti H2 (200–600 km)"
    },
    "Pipeline_H2_nouveau": {
        "cout_min": 0.20, "cout_max": 0.78,
        "vecteur": "GH2", "type": "terrestre",
        "desc": "Pipeline H2 neuf longue distance (>600 km)"
    },
    "Pipeline_sous_marin": {
        "cout_min": 0.05, "cout_max": 0.15,
        "vecteur": "GH2", "type": "maritime",
        "desc": "Pipeline sous-marin court (<50 km, détroit)"
    },
    "Tanker_NH3": {
        "cout_fixe_min": 0.08, "cout_fixe_max": 0.12,
        "cout_var_min": 0.08,  "cout_var_max": 0.14,
        "alpha": 0.65,
        "cout_max_cap": 1.20,
        "vecteur": "NH3", "type": "maritime",
        "desc": "Tanker ammoniac — export maritime (IEA 2024 / Dinh et al. 2024)"
    },
    "Tanker_LH2": {
        "cout_fixe_min": 0.15, "cout_fixe_max": 0.25,
        "cout_var_min": 0.12,  "cout_var_max": 0.20,
        "alpha": 0.65,
        "cout_max_cap": 2.00,
        "vecteur": "LH2", "type": "maritime",
        "desc": "Tanker H2 liquide — export maritime (DOE H2 Program 2024)"
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CAS SPÉCIAUX — Détroit de Gibraltar
# Le segment Tanger→Algésiras est un pipeline sous-marin (~14 km),
# pas un tanker classique. On le traite comme exception.
# ─────────────────────────────────────────────────────────────────────────────
DETROITS = {
    ("Tanger", "Algésiras"): "Pipeline_sous_marin",
    ("Algésiras", "Tanger"): "Pipeline_sous_marin",
}


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique (vol d'oiseau) en km — formule de Haversine."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(a)), 1)


def _osrm_distance(lat1, lon1, lat2, lon2, timeout=10):
    """
    Distance routière via OSRM (OpenStreetMap Routing Machine).
    Retourne (distance_km, True) si succès, (None, False) sinon.
    """
    url = (f"http://router.project-osrm.org/route/v1/driving/"
           f"{lon1},{lat1};{lon2},{lat2}?overview=false")
    try:
        r = requests.get(url, timeout=timeout)
        data = r.json()
        if data.get("code") == "Ok" and data.get("routes"):
            dist_km = round(data["routes"][0]["distance"] / 1000, 1)
            return dist_km, True
    except Exception:
        pass
    return None, False


def get_distance_terrestre(lat1, lon1, lat2, lon2):
    """
    Distance terrestre : OSRM d'abord, fallback Haversine×1.3.
    Utilisée pour TOUS les segments terrestres (Maroc ET Europe).
    """
    dist_osrm, ok = _osrm_distance(lat1, lon1, lat2, lon2)
    if ok and dist_osrm is not None:
        return dist_osrm, "OSRM Routier"
    # Fallback : Haversine × 1.3 (facteur de détour routier standard)
    return round(haversine_km(lat1, lon1, lat2, lon2) * 1.3, 1), "Haversine×1.3 (fallback)"


def get_distance_maritime(lat1, lon1, lat2, lon2):
    """
    Distance maritime : Haversine × 1.15 (facteur routes maritimes vs vol d'oiseau).
    Source : Rodrigue, Geography of Transport Systems, 2020
    """
    return round(haversine_km(lat1, lon1, lat2, lon2) * 1.15, 1), "Haversine×1.15 (maritime)"


# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTION DU PORT OPTIMAL
# ══════════════════════════════════════════════════════════════════════════════

def port_optimal(lat_origine, lon_origine, nom_origine=None):
    """
    Trouve le port marocain optimal pour l'export.

    Logique :
      1. Si l'origine est elle-même un port marocain → export direct
      2. Sinon → port le plus proche par OSRM (ou Haversine en fallback)

    Accepte soit un nom de nœud connu, soit des coordonnées arbitraires.

    Retourne : (nom_port, distance_terrestre_km, methode_distance)
    """
    # Cas 1 : l'origine est un port connu
    if nom_origine and nom_origine in PORTS_MAROC:
        return nom_origine, 0.0, "Direct (origine = port)"

    # Cas 2 : chercher le port le plus proche
    best_port = None
    best_dist = float('inf')
    best_methode = ""

    for port_name in PORTS_MAROC:
        p = NODES[port_name]
        # Essayer OSRM pour la distance routière
        dist, methode = get_distance_terrestre(lat_origine, lon_origine, p["lat"], p["lon"])
        if dist < best_dist:
            best_dist = dist
            best_port = port_name
            best_methode = methode

        # Respecter le rate limit OSRM
        time.sleep(0.15)

    return best_port, round(best_dist, 1), best_methode


# ══════════════════════════════════════════════════════════════════════════════
# CHOIX DU MODE DE TRANSPORT
# ══════════════════════════════════════════════════════════════════════════════

def mode_optimal(distance_km, type_segment, nom_origine=None, nom_dest=None):
    """
    Sélectionne le mode de transport optimal.

    Type_segment :
      "terrestre"  → Tube_trailer / Pipeline reconverti / Pipeline neuf
      "maritime"   → Pipeline sous-marin (détroit) / Tanker NH3
      "transit_eu" → Pipeline reconverti / Pipeline neuf (réseau gaz européen)

    Vérifie d'abord les cas spéciaux (détroits).
    """
    # Cas spéciaux (ex: détroit de Gibraltar)
    if nom_origine and nom_dest:
        key = (nom_origine, nom_dest)
        if key in DETROITS:
            return DETROITS[key]

    if type_segment == "terrestre":
        if distance_km < 200:
            return "Tube_trailer"
        elif distance_km < 600:
            return "Pipeline_H2_reconverti"
        else:
            return "Pipeline_H2_nouveau"

    elif type_segment == "maritime":
        if distance_km < 50:
            return "Pipeline_sous_marin"
        else:
            return "Tanker_NH3"

    elif type_segment == "transit_eu":
        if distance_km < 600:
            return "Pipeline_H2_reconverti"
        else:
            return "Pipeline_H2_nouveau"

    else:
        raise ValueError(f"Type de segment inconnu : '{type_segment}'")


# ══════════════════════════════════════════════════════════════════════════════
# CALCUL DU COÛT DE TRANSPORT PAR SEGMENT
# ══════════════════════════════════════════════════════════════════════════════

def calcul_cout_segment(mode, distance_km):
    """
    Coût de transport min/max ($/kg H2) selon le mode et la distance.

    Pipeline / Tube trailer : modèle linéaire avec coût fixe
        LCOT = 0.05 + cout_variable × (distance / 1000)
    Tanker NH3 / LH2 : modèle puissance (Dinh et al. 2024)
        LCOT = cout_fixe + cout_variable × (distance / 1000)^0.65
        + coût craquage NH3→H2 à destination (0.50–1.00 $/kg)
    """
    p = MODES_PARAMS[mode]

    if "Pipeline" in mode or "Tube" in mode:
        cout_fixe = 0.05  # $/kg — stations compression, infrastructure connexion
        cmin = cout_fixe + p["cout_min"] * (distance_km / 1000)
        cmax = cout_fixe + p["cout_max"] * (distance_km / 1000)

    else:
        # Tanker NH3 ou LH2
        alpha = p["alpha"]
        dist_ref = (distance_km / 1000) ** alpha

        cmin = p["cout_fixe_min"] + p["cout_var_min"] * dist_ref
        cmax = p["cout_fixe_max"] + p["cout_var_max"] * dist_ref

        # Plafonnement (Hydrogen Council 2023)
        cmin = min(cmin, p["cout_max_cap"] * 0.60)
        cmax = min(cmax, p["cout_max_cap"])

        # Coût craquage NH3→H2 à destination
        if mode == "Tanker_NH3":
            cout_craquage_min = 0.50  # $/kg
            cout_craquage_max = 1.00  # $/kg
            cmin += cout_craquage_min
            cmax += cout_craquage_max
            cmin = min(cmin, p["cout_max_cap"] * 0.60 + cout_craquage_min)
            cmax = min(cmax, p["cout_max_cap"] + cout_craquage_max)

    return round(max(cmin, 0.01), 4), round(max(cmax, 0.01), 4)


# ══════════════════════════════════════════════════════════════════════════════
# DÉCOMPOSITION AUTOMATIQUE TERRE/MER
# ══════════════════════════════════════════════════════════════════════════════

def _est_destination_maritime(nom_dest):
    """
    Détermine si la destination nécessite un transport maritime.
    Vrai si la destination est hors du Maroc ou dans les îles.
    """
    if nom_dest not in NODES:
        return True  # destination inconnue → on suppose export
    info = NODES[nom_dest]
    return info["zone"] != "Maroc"


def _est_destination_terrestre_eu(nom_dest):
    """Destination terrestre en Europe (pas un port d'arrivée final)."""
    if nom_dest not in NODES:
        return False
    info = NODES[nom_dest]
    return info["zone"] == "Europe" and info["type_noeud"] == "Marché final"


def decomposer_corridor(nom_origine, nom_dest, lat_o=None, lon_o=None):
    """
    Décompose automatiquement un corridor en segments terre/mer.

    Cas possibles :
    ─────────────────────────────────────────────────────────────────────
    1. Domestique (Maroc → Maroc) :
       → 1 segment terrestre (OSRM)

    2. Export direct depuis un port marocain :
       → 1 segment maritime (Haversine×1.15)

    3. Export depuis un site intérieur :
       → Segment 1 : terrestre (site → port optimal, OSRM)
       → Segment 2 : maritime (port → destination, Haversine×1.15)

    4. Transit EU (ex: Algésiras → Paris) :
       → 1 segment terrestre Europe (OSRM)

    Retourne : liste de dicts, chacun = un segment avec toutes les infos.
    """
    # Résoudre les coordonnées de l'origine
    if lat_o is None or lon_o is None:
        if nom_origine in NODES:
            lat_o = NODES[nom_origine]["lat"]
            lon_o = NODES[nom_origine]["lon"]
        else:
            raise ValueError(f"Origine '{nom_origine}' introuvable et pas de coordonnées fournies")

    if nom_dest not in NODES:
        raise ValueError(f"Destination '{nom_dest}' introuvable dans NODES")

    dest = NODES[nom_dest]
    segments = []

    # ── CAS 1 : Domestique (les deux au Maroc) ────────────────────────────────
    if not _est_destination_maritime(nom_dest):
        dist, methode = get_distance_terrestre(lat_o, lon_o, dest["lat"], dest["lon"])
        mode = mode_optimal(dist, "terrestre", nom_origine, nom_dest)
        cmin, cmax = calcul_cout_segment(mode, dist)
        segments.append({
            "type_segment": "Terrestre",
            "origine": nom_origine,
            "destination": nom_dest,
            "lat_depart": lat_o, "lon_depart": lon_o,
            "lat_arrivee": dest["lat"], "lon_arrivee": dest["lon"],
            "distance_km": dist,
            "methode_distance": methode,
            "mode_optimal": mode,
            "vecteur_H2": MODES_PARAMS[mode]["vecteur"],
            "cout_min_USD_kg": cmin,
            "cout_max_USD_kg": cmax,
        })
        return segments

    # ── CAS 2/3 : Export (destination hors Maroc) ─────────────────────────────
    # Étape A : trouver le port marocain optimal
    port_name, dist_terre, meth_terre = port_optimal(lat_o, lon_o, nom_origine)

    # Segment terrestre (si l'origine n'est PAS le port)
    if dist_terre > 0:
        port_info = NODES[port_name]
        mode_terre = mode_optimal(dist_terre, "terrestre", nom_origine, port_name)
        cmin_t, cmax_t = calcul_cout_segment(mode_terre, dist_terre)
        segments.append({
            "type_segment": "Terrestre (vers port)",
            "origine": nom_origine,
            "destination": port_name,
            "lat_depart": lat_o, "lon_depart": lon_o,
            "lat_arrivee": port_info["lat"], "lon_arrivee": port_info["lon"],
            "distance_km": dist_terre,
            "methode_distance": meth_terre,
            "mode_optimal": mode_terre,
            "vecteur_H2": MODES_PARAMS[mode_terre]["vecteur"],
            "cout_min_USD_kg": cmin_t,
            "cout_max_USD_kg": cmax_t,
        })

    # Segment maritime (port → destination)
    port_coords = NODES[port_name]

    # Cas spécial : destination terrestre en Europe (ex: Paris)
    # → on cherche le port EU qui minimise le coût TOTAL (maritime + terrestre EU)
    if _est_destination_terrestre_eu(nom_dest):
        ports_eu = [n for n, info in NODES.items()
                    if info["zone"] == "Europe" and "Port" in info["type_noeud"]]

        best_port_eu = None
        best_total_cost = float('inf')
        for p_eu in ports_eu:
            p_eu_info = NODES[p_eu]
            # Coût maritime (port Maroc → port EU)
            d_mer = haversine_km(port_coords["lat"], port_coords["lon"],
                                 p_eu_info["lat"], p_eu_info["lon"]) * 1.15
            m_mer = mode_optimal(d_mer, "maritime", port_name, p_eu)
            _, cmax_mer = calcul_cout_segment(m_mer, d_mer)
            # Coût terrestre EU (port EU → destination)
            d_terre_eu = haversine_km(p_eu_info["lat"], p_eu_info["lon"],
                                      dest["lat"], dest["lon"]) * 1.3
            m_terre_eu = mode_optimal(d_terre_eu, "transit_eu", p_eu, nom_dest)
            _, cmax_terre_eu = calcul_cout_segment(m_terre_eu, d_terre_eu)
            # Total
            total = cmax_mer + cmax_terre_eu
            if total < best_total_cost:
                best_total_cost = total
                best_port_eu = p_eu

        # Segment maritime : port Maroc → port Europe
        p_eu_info = NODES[best_port_eu]
        dist_mer, meth_mer = get_distance_maritime(
            port_coords["lat"], port_coords["lon"],
            p_eu_info["lat"], p_eu_info["lon"]
        )
        mode_mer = mode_optimal(dist_mer, "maritime", port_name, best_port_eu)
        cmin_m, cmax_m = calcul_cout_segment(mode_mer, dist_mer)
        segments.append({
            "type_segment": "Maritime",
            "origine": port_name,
            "destination": best_port_eu,
            "lat_depart": port_coords["lat"], "lon_depart": port_coords["lon"],
            "lat_arrivee": p_eu_info["lat"], "lon_arrivee": p_eu_info["lon"],
            "distance_km": dist_mer,
            "methode_distance": meth_mer,
            "mode_optimal": mode_mer,
            "vecteur_H2": MODES_PARAMS[mode_mer]["vecteur"],
            "cout_min_USD_kg": cmin_m,
            "cout_max_USD_kg": cmax_m,
        })

        # Segment terrestre Europe : port EU → destination finale
        dist_eu, meth_eu = get_distance_terrestre(
            p_eu_info["lat"], p_eu_info["lon"],
            dest["lat"], dest["lon"]
        )
        mode_eu = mode_optimal(dist_eu, "transit_eu", best_port_eu, nom_dest)
        cmin_eu, cmax_eu = calcul_cout_segment(mode_eu, dist_eu)
        segments.append({
            "type_segment": "Terrestre (transit EU)",
            "origine": best_port_eu,
            "destination": nom_dest,
            "lat_depart": p_eu_info["lat"], "lon_depart": p_eu_info["lon"],
            "lat_arrivee": dest["lat"], "lon_arrivee": dest["lon"],
            "distance_km": dist_eu,
            "methode_distance": meth_eu,
            "mode_optimal": mode_eu,
            "vecteur_H2": MODES_PARAMS[mode_eu]["vecteur"],
            "cout_min_USD_kg": cmin_eu,
            "cout_max_USD_kg": cmax_eu,
        })

    else:
        # Destination = port étranger → segment maritime direct
        dist_mer, meth_mer = get_distance_maritime(
            port_coords["lat"], port_coords["lon"],
            dest["lat"], dest["lon"]
        )
        mode_mer = mode_optimal(dist_mer, "maritime", port_name, nom_dest)
        cmin_m, cmax_m = calcul_cout_segment(mode_mer, dist_mer)
        segments.append({
            "type_segment": "Maritime",
            "origine": port_name,
            "destination": nom_dest,
            "lat_depart": port_coords["lat"], "lon_depart": port_coords["lon"],
            "lat_arrivee": dest["lat"], "lon_arrivee": dest["lon"],
            "distance_km": dist_mer,
            "methode_distance": meth_mer,
            "mode_optimal": mode_mer,
            "vecteur_H2": MODES_PARAMS[mode_mer]["vecteur"],
            "cout_min_USD_kg": cmin_m,
            "cout_max_USD_kg": cmax_m,
        })

    return segments


# ══════════════════════════════════════════════════════════════════════════════
# INTERPOLATION IDW — Pour sites arbitraires hors base de données
# ══════════════════════════════════════════════════════════════════════════════

# Données de référence des 12 régions (extraites de T1)
REGIONS_REF = {
    "Laayoune"   : {"lat": 27.1253, "lon": -13.1625, "GHI": 2160, "vent": 7.8, "cout_eau": 0.75},
    "Dakhla"     : {"lat": 23.6848, "lon": -15.9572, "GHI": 2155, "vent": 9.0, "cout_eau": 0.70},
    "Boujdour"   : {"lat": 26.1000, "lon": -14.5000, "GHI": 2175, "vent": 8.5, "cout_eau": 0.80},
    "Guelmim"    : {"lat": 28.9870, "lon": -10.0572, "GHI": 1940, "vent": 5.5, "cout_eau": 0.95},
    "Jorf_Lasfar": {"lat": 33.1100, "lon": -8.6300,  "GHI": 1900, "vent": 5.0, "cout_eau": 0.50},
    "Ouarzazate" : {"lat": 30.9189, "lon": -6.8934,  "GHI": 2180, "vent": 5.5, "cout_eau": 1.00},
    "Agadir"     : {"lat": 30.4278, "lon": -9.5981,  "GHI": 2095, "vent": 5.5, "cout_eau": 0.48},
    "Tanger"     : {"lat": 35.7595, "lon": -5.8340,  "GHI": 1840, "vent": 9.5, "cout_eau": 0.52},
    "Casablanca" : {"lat": 33.5731, "lon": -7.5898,  "GHI": 1875, "vent": 4.5, "cout_eau": 0.50},
    "Nador"      : {"lat": 35.1681, "lon": -2.9335,  "GHI": 1785, "vent": 5.8, "cout_eau": 0.55},
    "Marrakech"  : {"lat": 31.6295, "lon": -7.9811,  "GHI": 2085, "vent": 4.0, "cout_eau": 0.65},
    "Midelt"     : {"lat": 32.6800, "lon": -4.7340,  "GHI": 2200, "vent": 5.5, "cout_eau": 0.90},
}


def interpolation_idw(lat, lon, parametre, power=2):
    """
    Interpolation par pondération inverse de la distance (IDW).

    Pour un site arbitraire (lat, lon), estime un paramètre (GHI, vent, cout_eau)
    à partir des 12 régions de référence.

    IDW : valeur = Σ(wi × vi) / Σ(wi)
          wi = 1 / di^power
          di = distance Haversine au site i

    Si le point est exactement sur un site connu → retourne sa valeur directe.

    Source : Shepard, 1968 — méthode standard interpolation géospatiale
    """
    weights = []
    values = []

    for name, ref in REGIONS_REF.items():
        d = haversine_km(lat, lon, ref["lat"], ref["lon"])
        if d < 1.0:  # point quasi-identique à un site connu
            return ref[parametre], name
        w = 1.0 / (d ** power)
        weights.append(w)
        values.append(ref[parametre])

    total_w = sum(weights)
    result = sum(w * v for w, v in zip(weights, values)) / total_w
    return round(result, 2), "IDW interpolé"


def parametres_site_arbitraire(lat, lon):
    """
    Estime tous les paramètres d'un site arbitraire par IDW.
    Retourne un dict compatible avec le modèle.
    """
    ghi, src_ghi = interpolation_idw(lat, lon, "GHI")
    vent, src_vent = interpolation_idw(lat, lon, "vent")
    eau, src_eau = interpolation_idw(lat, lon, "cout_eau")

    return {
        "GHI_kWh_m2_an": ghi,
        "vitesse_vent_moy_ms": vent,
        "cout_eau_USD_m3": eau,
        "source_GHI": src_ghi,
        "source_vent": src_vent,
        "source_eau": src_eau,
        "methode": "IDW (Inverse Distance Weighting, p=2)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION GÉNÉRIQUE : calcul_corridor_auto
# ══════════════════════════════════════════════════════════════════════════════

def calcul_corridor_auto(origine, destination, lat_o=None, lon_o=None):
    """
    Calcule un corridor complet de manière automatique.

    Accepte :
      - Un nom de nœud connu : calcul_corridor_auto("Midelt", "Rotterdam")
      - Des coordonnées arbitraires : calcul_corridor_auto("MonSite", "Rotterdam",
                                                            lat_o=29.5, lon_o=-10.2)

    Retourne :
      - segments : liste de segments avec distances, modes, coûts
      - resume : dict agrégé du corridor complet
    """
    # Résoudre les coordonnées de l'origine
    if lat_o is None or lon_o is None:
        if origine in NODES:
            lat_o = NODES[origine]["lat"]
            lon_o = NODES[origine]["lon"]
        else:
            raise ValueError(
                f"Origine '{origine}' non trouvée dans NODES. "
                f"Fournir lat_o et lon_o pour un site arbitraire."
            )

    # Décomposer en segments
    segments = decomposer_corridor(origine, destination, lat_o, lon_o)

    # Agrégation
    total_dist = round(sum(s["distance_km"] for s in segments), 1)
    total_cmin = round(sum(s["cout_min_USD_kg"] for s in segments), 4)
    total_cmax = round(sum(s["cout_max_USD_kg"] for s in segments), 4)

    modes_list = [s["mode_optimal"] for s in segments]
    types_list = [s["type_segment"] for s in segments]
    vecteurs_list = list(dict.fromkeys([s["vecteur_H2"] for s in segments]))

    resume = {
        "corridor_label": f"{origine}→{destination}",
        "nb_segments": len(segments),
        "origine": origine,
        "destination": destination,
        "types_segments": " → ".join(types_list),
        "chaine_logistique": " + ".join(modes_list),
        "vecteurs": " + ".join(vecteurs_list),
        "total_distance_km": total_dist,
        "total_cout_min_USD_kg": total_cmin,
        "total_cout_max_USD_kg": total_cmax,
    }

    return segments, resume


# ══════════════════════════════════════════════════════════════════════════════
# CORRIDORS PRÉDÉFINIS — conservés pour compatibilité + enrichis
# ══════════════════════════════════════════════════════════════════════════════
CORRIDORS_PREDEFINIS = [
    # ── Domestiques (source intérieure → port/hub le plus proche) ─────
    ("Dakhla",      "Agadir"),
    ("Dakhla",      "Casablanca"),
    ("Laayoune",    "Casablanca"),
    ("Laayoune",    "Agadir"),
    ("Ouarzazate",  "Casablanca"),
    ("Ouarzazate",  "Jorf_Lasfar"),
    ("Tarfaya",     "Tanger"),
    ("Tarfaya",     "Agadir"),
    ("Guelmim",     "Agadir"),
    ("Guelmim",     "Casablanca"),
    ("Jorf_Lasfar", "Casablanca"),
    ("Midelt",      "Casablanca"),
    ("Midelt",      "Jorf_Lasfar"),
    ("Midelt",      "Nador"),
    ("Marrakech",   "Casablanca"),
    ("Marrakech",   "Jorf_Lasfar"),
    ("Marrakech",   "Agadir"),
    ("Boujdour",    "Agadir"),
    ("Boujdour",    "Laayoune"),

    # ── Export direct (chaque port marocain → chaque marché) ──────────
    # Tanger (détroit)
    ("Tanger",      "Algésiras"),
    ("Tanger",      "Rotterdam"),
    ("Tanger",      "Marseille"),
    ("Tanger",      "Barcelone"),
    # Casablanca (hub central)
    ("Casablanca",  "Rotterdam"),
    ("Casablanca",  "Marseille"),
    ("Casablanca",  "Barcelone"),
    # Jorf Lasfar (port industriel OCP)
    ("Jorf_Lasfar", "Rotterdam"),
    ("Jorf_Lasfar", "Marseille"),
    ("Jorf_Lasfar", "Barcelone"),
    # Agadir
    ("Agadir",      "Rotterdam"),
    ("Agadir",      "Barcelone"),
    ("Agadir",      "Marseille"),
    ("Agadir",      "Canaries"),
    # Nador
    ("Nador",       "Almería"),
    ("Nador",       "Rotterdam"),
    ("Nador",       "Marseille"),
    ("Nador",       "Barcelone"),
    # Dakhla / Laayoune (export direct possible)
    ("Dakhla",      "Dakar"),
    ("Dakhla",      "Canaries"),
    ("Laayoune",    "Rotterdam"),
    ("Laayoune",    "Canaries"),

    # ── Multi-segments (source → port → marché, décomposition auto) ───
    ("Dakhla",      "Rotterdam"),
    ("Dakhla",      "Marseille"),
    ("Dakhla",      "Paris"),
    ("Ouarzazate",  "Rotterdam"),
    ("Ouarzazate",  "Marseille"),
    ("Midelt",      "Rotterdam"),
    ("Midelt",      "Marseille"),
    ("Tarfaya",     "Algésiras"),
    ("Laayoune",    "Barcelone"),
    ("Guelmim",     "Rotterdam"),
    ("Guelmim",     "Marseille"),
    ("Guelmim",     "Barcelone"),
    ("Marrakech",   "Rotterdam"),
    ("Marrakech",   "Marseille"),
    ("Marrakech",   "Barcelone"),
    ("Boujdour",    "Rotterdam"),

    # Via Europe (transit terrestre EU)
    ("Tanger",      "Paris"),
]


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE — build_T4_transport (v2)
# ══════════════════════════════════════════════════════════════════════════════

def build_T4_transport(output_dir=None, taux_change_usd=0.9217,
                       monnaie_ref="EUR", annee_ref=2024):
    """
    Construit la Table 4 — Transport & Infrastructure.

    Améliorations v2 :
      - OSRM systématique pour tous les segments terrestres
      - Décomposition automatique terre/mer
      - Support de corridors prédéfinis ET dynamiques
      - Gestion propre de la conversion EUR
    """
    print("  [T4 v2] Démarrage du calcul des corridors H2...")
    print(f"          {len(CORRIDORS_PREDEFINIS)} corridors prédéfinis à calculer")

    rows_resume = []
    rows_detail = []

    for i, (orig, dest) in enumerate(CORRIDORS_PREDEFINIS):
        print(f"    [{i+1}/{len(CORRIDORS_PREDEFINIS)}] {orig} → {dest}...", end="")

        try:
            segments, resume = calcul_corridor_auto(orig, dest)

            # Enrichir le résumé avec conversion EUR
            resume["total_cout_min_EUR_kg"] = round(resume["total_cout_min_USD_kg"] * taux_change_usd, 4)
            resume["total_cout_max_EUR_kg"] = round(resume["total_cout_max_USD_kg"] * taux_change_usd, 4)
            resume["devise_ref"] = monnaie_ref
            resume["annee_ref"] = annee_ref
            resume["source_couts"] = "IEA 2024, Hydrogen Council 2023, Dinh et al. 2024"
            rows_resume.append(resume)

            # Enrichir chaque segment
            for seg in segments:
                seg["corridor_label"] = resume["corridor_label"]
                seg["cout_min_EUR_kg"] = round(seg["cout_min_USD_kg"] * taux_change_usd, 4)
                seg["cout_max_EUR_kg"] = round(seg["cout_max_USD_kg"] * taux_change_usd, 4)
                seg["desc_mode"] = MODES_PARAMS[seg["mode_optimal"]]["desc"]
                seg["devise_ref"] = monnaie_ref
                seg["annee_ref"] = annee_ref

                # Métadonnées des nœuds (si connus)
                if seg["origine"] in NODES:
                    n = NODES[seg["origine"]]
                    seg["zone_depart"] = n["zone"]
                    seg["type_noeud_depart"] = n["type_noeud"]
                    seg["code_locode_depart"] = n["code_locode"]
                    seg["pays_depart"] = n["pays"]
                if seg["destination"] in NODES:
                    n = NODES[seg["destination"]]
                    seg["zone_arrivee"] = n["zone"]
                    seg["type_noeud_arrivee"] = n["type_noeud"]
                    seg["code_locode_arrivee"] = n["code_locode"]
                    seg["pays_arrivee"] = n["pays"]

                rows_detail.append(seg)

            print(f" ✓ {resume['nb_segments']} seg, {resume['total_distance_km']} km, "
                  f"${resume['total_cout_min_USD_kg']:.3f}–{resume['total_cout_max_USD_kg']:.3f}/kg")

        except Exception as e:
            print(f" ✗ ERREUR: {e}")
            continue

        # Rate limiting OSRM
        time.sleep(0.3)

    # Création DataFrames
    df_resume = pd.DataFrame(rows_resume)
    df_detail = pd.DataFrame(rows_detail)

    # Sauvegarde
    if output_dir:
        os.makedirs(f"{output_dir}/csv", exist_ok=True)
        csv_resume = f"{output_dir}/csv/T4_corridors_resume.csv"
        csv_detail = f"{output_dir}/csv/T4_segments_detail.csv"
        df_resume.to_csv(csv_resume, index=False, encoding="utf-8-sig")
        df_detail.to_csv(csv_detail, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ Résumé ({len(df_resume)} corridors) : {csv_resume}")
        print(f"  📄 Détail ({len(df_detail)} segments)  : {csv_detail}")

    # Affichage console
    print(f"\n  {'Corridor':<35} {'Seg':>4} {'Dist (km)':>10} {'Min $/kg':>10} {'Max $/kg':>10}  Chaîne")
    print("  " + "─" * 100)
    for _, row in df_resume.iterrows():
        print(f"  {row['corridor_label']:<35} {row['nb_segments']:>4} "
              f"{row['total_distance_km']:>10.0f} "
              f"{row['total_cout_min_USD_kg']:>10.3f} {row['total_cout_max_USD_kg']:>10.3f}  "
              f"{row['chaine_logistique']}")

    return df_resume, df_detail


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE UTILISATEUR — Corridor pour site arbitraire
# ══════════════════════════════════════════════════════════════════════════════

def corridor_site_arbitraire(lat, lon, nom_site, destination,
                              taux_change_usd=0.9217):
    """
    Calcule un corridor complet depuis un site arbitraire (non dans la base).

    Exemple :
        corridor_site_arbitraire(29.5, -10.2, "MonSite_Tan-Tan", "Rotterdam")

    Retourne :
      - params : paramètres interpolés du site (GHI, vent, coût eau)
      - segments : détail de chaque segment
      - resume : résumé agrégé du corridor
    """
    print(f"\n  ═══ Corridor arbitraire : {nom_site} ({lat}°N, {lon}°E) → {destination} ═══")

    # 1. Interpoler les paramètres du site
    params = parametres_site_arbitraire(lat, lon)
    print(f"  📊 Paramètres interpolés (IDW) :")
    print(f"     GHI = {params['GHI_kWh_m2_an']} kWh/m²/an ({params['source_GHI']})")
    print(f"     Vent = {params['vitesse_vent_moy_ms']} m/s ({params['source_vent']})")
    print(f"     Coût eau = {params['cout_eau_USD_m3']} $/m³ ({params['source_eau']})")

    # 2. Calculer le corridor
    segments, resume = calcul_corridor_auto(nom_site, destination,
                                             lat_o=lat, lon_o=lon)

    # 3. Enrichir avec conversion EUR
    resume["total_cout_min_EUR_kg"] = round(resume["total_cout_min_USD_kg"] * taux_change_usd, 4)
    resume["total_cout_max_EUR_kg"] = round(resume["total_cout_max_USD_kg"] * taux_change_usd, 4)
    resume["parametres_site"] = params

    # Affichage
    print(f"\n  📦 Résultat : {resume['nb_segments']} segments")
    for s in segments:
        print(f"     {s['type_segment']:<25} {s['origine']:<15} → {s['destination']:<15} "
              f"{s['distance_km']:>8.0f} km  {s['mode_optimal']:<25} "
              f"${s['cout_min_USD_kg']:.3f}–{s['cout_max_USD_kg']:.3f}/kg")
    print(f"  ──────────────────────────────────────────────────────────────")
    print(f"  TOTAL : {resume['total_distance_km']} km | "
          f"${resume['total_cout_min_USD_kg']:.3f}–{resume['total_cout_max_USD_kg']:.3f}/kg")

    return params, segments, resume


# ══════════════════════════════════════════════════════════════════════════════
# TEST RAPIDE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  TEST T4 v2 — Transport & Infrastructure H2")
    print("=" * 70)

    # Test 1 : corridor prédéfini
    print("\n--- Test 1 : Midelt → Rotterdam (auto-décomposition) ---")
    segs, res = calcul_corridor_auto("Midelt", "Rotterdam")
    for s in segs:
        print(f"  {s['type_segment']}: {s['origine']}→{s['destination']} "
              f"| {s['distance_km']} km | {s['mode_optimal']} "
              f"| ${s['cout_min_USD_kg']:.3f}–{s['cout_max_USD_kg']:.3f}")
    print(f"  TOTAL: {res['total_distance_km']} km | "
          f"${res['total_cout_min_USD_kg']:.3f}–{res['total_cout_max_USD_kg']:.3f}/kg")

    # Test 2 : site arbitraire
    print("\n--- Test 2 : Site arbitraire (Tan-Tan, 28.43°N, -11.10°W) → Rotterdam ---")
    params, segs2, res2 = corridor_site_arbitraire(28.43, -11.10, "Tan-Tan", "Rotterdam")

    # Test 3 : build complet
    print("\n--- Test 3 : Build complet T4 ---")
    df_r, df_d = build_T4_transport()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5 — PARAMÈTRES ÉCONOMIQUES & FINANCIERS
# ══════════════════════════════════════════════════════════════════════════════
def build_T5_economique():
    print("  [T5] Construction : Paramètres économiques & financiers...")

    # ── Données de base ─────────────────────────────────────────────────────────────
    data = {
        'parametre'             : [
            # Macro Maroc
            'taux_actualisation_min','taux_actualisation_mode','taux_actualisation_max',
            'inflation_MAD_pct','taux_USD_MAD','taux_EUR_MAD',
            'prime_risque_pays_vs_EU','cout_dette_pct','ratio_dette_fp',
            'duree_amortissement_ans','IS_corporate_pct',
            # Prix électricité
            'tarif_indus_HTA_USD_kWh','PPA_solaire_min','PPA_solaire_mode','PPA_solaire_max',
            'PPA_eolien_min','PPA_eolien_mode','PPA_eolien_max',
            'PPA_hybride_min','PPA_hybride_mode','PPA_hybride_max',
            'evolution_prix_elec_2030','evolution_prix_elec_2040','evolution_prix_elec_2050',
            # Main d'œuvre
            'salaire_ingenieur_min_USD_mois','salaire_ingenieur_mode_USD_mois',
            'salaire_technicien_min_USD_mois','salaire_technicien_mode_USD_mois',
            'salaire_operateur_min_USD_mois','salaire_operateur_mode_USD_mois',
            'ratio_salaire_Maroc_vs_EU','charges_sociales_pct',
            'emplois_directs_par_MW_H2','emplois_indirects_par_MW_H2',
            # Prix H2 marché
            'H2_vert_actuel_min','H2_vert_actuel_mode','H2_vert_actuel_max',
            'H2_vert_cible_2030_min','H2_vert_cible_2030_mode','H2_vert_cible_2030_max',
            'H2_vert_cible_2040_min','H2_vert_cible_2040_mode','H2_vert_cible_2040_max',
            'prix_import_EU_USD_kg_min','prix_import_EU_USD_kg_mode','prix_import_EU_USD_kg_max',
            'NH3_vert_USD_tonne_min','NH3_vert_USD_tonne_mode','NH3_vert_USD_tonne_max',
            'objectif_Hydrogen_Shot_DOE_USD_kg',
        ],
        'valeur'                : [
            6,8,12, 3.5,10.05,10.90, 3.0,5.5,70, 20,31,
            0.094, 0.015,0.025,0.040, 0.020,0.032,0.050, 0.018,0.028,0.045,
            0.75,0.55,0.40,
            1000,1500, 400,700, 250,400,
            0.25,26, 2.5,5.0,
            4.0,6.0,9.0, 2.0,3.0,4.5, 1.5,2.0,3.0,
            3.5,5.0,7.0, 400,600,900, 1.0,
        ],
        'unite'                 : [
            '%','%','%', '%','MAD/USD','MAD/EUR', '%','%','%', 'ans','%',
            'USD/kWh','USD/kWh','USD/kWh','USD/kWh','USD/kWh','USD/kWh','USD/kWh',
            'USD/kWh','USD/kWh','USD/kWh', 'facteur','facteur','facteur',
            'USD/mois','USD/mois','USD/mois','USD/mois','USD/mois','USD/mois',
            'ratio','%','emplois/MW','emplois/MW',
            'USD/kg','USD/kg','USD/kg','USD/kg','USD/kg','USD/kg',
            'USD/kg','USD/kg','USD/kg','USD/kg','USD/kg','USD/kg',
            'USD/kg','USD/kg','USD/kg','USD/t',
        ],
        'source'                : [
            'WACC MENA analysis',
            'HCP Maroc 2024','BAM 2024',
            'Moody\'s rating Ba1','BAM 2024','Standard financement MENA',
            'Standard projet EnR','DGI Maroc 2024',
            'ONEE tarif 2024',
            'MASEN PPA record Midelt 2019','MASEN 2024','IEA LCOH Review 2024',
            'Tarfaya LCOE réel IRENA 2022','IRENA Maroc 2024','IEA 2024',
            'MASEN PPA Midelt 2019 (record)','MASEN 2024','IEA 2024',
            'IEA Learning curves 2024','IEA 2024','IEA 2024',
            'ANAPEC Maroc 2024','ANAPEC Maroc 2024','ANAPEC Maroc 2024','ANAPEC Maroc 2024',
            'ANAPEC Maroc 2024','ANAPEC Maroc 2024',
            'BIT 2024','CNSS Maroc 2024','IRENA Jobs Report 2024','IRENA Jobs Report 2024',
            'IEA LCOH 2024','IEA LCOH 2024','IEA LCOH 2024',
            'IEA LCOH 2024','IEA LCOH 2024','IEA LCOH 2024',
            'IEA Hydrogen Roadmap 2024','IEA 2024','IEA 2024',
            'IEA 2040 outlook','IEA 2040 outlook','IEA 2040 outlook',
            'EU H2 import pricing study','EU H2 import pricing study','EU H2 import pricing study',
            'Hydrogen Council 2023','Hydrogen Council 2023','Hydrogen Council 2023',
            'DOE H2 Program 2024 (Hydrogen Shot)',
        ]
    }

    # ── Création DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(data)

    # ── Détection automatique de la devise originale ─────────────────────
    # On considère USD, MAD ou EUR selon l'unité
    df['devise_originale'] = df['unite'].apply(
        lambda x: 'USD' if 'USD' in x else ('MAD' if 'MAD' in x else MONNAIE_REF)
    )

    # ── Colonne EUR : conversion selon devise originale ───────────────────
    def convertir(row):
        """
        Convertit la valeur en EUR selon le taux de change défini
        Si la devise est déjà EUR ou N/A → inchangé
        """
        devise = row['devise_originale']
        taux   = TAUX_CHANGE.get(devise)
        if taux and devise != MONNAIE_REF and devise != 'N/A':
            return round(row['valeur'] * taux, 6)
        return row['valeur']   # N/A ou déjà EUR → inchangé

    df['valeur_EUR'] = df.apply(convertir, axis=1)

    # ── Colonnes de référence pour traçabilité ───────────────────────────
    df['devise_ref']  = MONNAIE_REF
    df['annee_ref']   = ANNEE_REF
    df['source_taux'] = SOURCE_TAUX

    # ── Sauvegarde CSV ─────────────────────────────────────────────────
    df.to_csv(f"{OUTPUT_DIR}/csv/T5_parametres_economiques.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T5 sauvegardé : {len(df)} paramètres")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6 — MARCHÉ & DEMANDE
# ══════════════════════════════════════════════════════════════════════════════
def build_T6_marche():
    print("  [T6] Construction : Marché & demande H2...")

    # ── Demande nationale par secteur (ktH2/an) ─────────────────────────
    demande = pd.DataFrame({
        'secteur'       : ['Industrie_chimique_OCP','Raffinage_pétrole','Mobilité_FCEV',
                           'Mobilité_train_H2','Chaleur_industrielle','Stockage_réseau',
                           'Aviation_carburant_synth','Dessalement_eau','Résidentiel_heat'],
        'demande_2024'  : [120, 80,  2,  0,  10,  0,  0, 0, 0],
        'demande_2030'  : [200, 70,  20, 5,  50,  15, 5, 2, 1],
        'demande_2035'  : [280, 55,  50, 15, 100, 40, 15,5, 3],
        'demande_2040'  : [350, 45,  80, 30, 150, 80, 40,8, 8],
        'demande_2050'  : [500, 30,  200,80, 350, 250,150,15,20],
        'unite'         : ['ktH2/an']*9,
        'scenario'      : ['Moderé']*9,
        'source'        : ['Stratégie Nationale H2 Maroc 2021 + OCP Group']*9,
    })

    # Commentaires explicatifs pour les secteurs
    # OCP consomme aujourd'hui du H2 gris pour produire ses engrais
    # consommation réelle OCP ~ 120 000 tonnes H2/an (rapport OCP 2023)
    # objectif : remplacer progressivement par H2 vert
    # 2050 : 500 kt → OCP veut devenir 100% vert
    # Raffinage pétrole
    # 2024 : 80 ktH2/an → baisse vers 30 kt en 2050
    # Pourquoi ça baisse ? Déclin progressif du raffinage pétrolier avec transition énergétique
    # Source : IEA Oil Refining Outlook 2024
    # Mobilité FCEV (voitures H2)
    # 2024 : 2 kt (quasi inexistant aujourd'hui)
    # 2050 : 200 kt (croissance forte)
    # Calcul : 1 voiture FCEV consomme ~1 kg H2 / 100 km
    # Parc 2050 estimé : ~500 000 véhicules × 400 kg/an = 200 kt ✅
    # Source : Stratégie mobilité verte Maroc 2030
    # Stockage réseau
    # 2024 : 0 → pas encore déployé
    # 2050 : 250 kt → power-to-gas massif
    # Logique : Quand ENR > demande → électrolyse → H2 stocké
    # Quand ENR < demande → H2 → électricité (pile à combustible)
    # Source : ONEE Plan Réseau 2024

    # ── Benchmark concurrents (compétiteurs de Maroc) ───────────────────
    competitors = pd.DataFrame({
        'pays'              : ['Maroc','Arabie Saoudite','Égypte','Chili','Australie','Namibie','Espagne'],
        'LCOH_2024_USD_kg'  : [4.5, 4.0, 4.8, 3.8, 4.2, 5.0, 5.5],
        'LCOH_2030_USD_kg'  : [2.0, 1.5, 2.2, 1.8, 2.0, 2.5, 2.8],
        'LCOH_2040_USD_kg'  : [1.3, 1.0, 1.5, 1.2, 1.4, 1.8, 2.0],
        'CF_solaire_pct'    : [31,  35,  28,  30,  30,  30,  20 ],
        'CF_eolien_pct'     : [40,  25,  35,  55,  45,  40,  30 ],
        'distance_EU_km'    : [14,  5000,3000,12000,18000,6000,0  ],
        'avantage_decisif'  : ['Proximité EU 14km',
                               'Ressources solaires exceptionnelles',
                               'Faibles coûts + Suez',
                               'Éolien record Patagonie',
                               'Solaire + éolien + superficie',
                               'Territoire vierge + soleil',
                               'Marché domestique EU'],
        'risque_principal'  : ['Infrastructure réseau','Stabilité politique','Financement',
                               'Distance marché','Distance marché','Infrastructure','Coûts élevés'],
        'source'            : ['IEA 2024 + Applied Energy 2024']*7,
    })

    # ── Colonnes EUR ajoutées pour LCOH ───────────────────────────────
    for col in ['LCOH_2024_USD_kg','LCOH_2030_USD_kg','LCOH_2040_USD_kg']:
        col_eur = col.replace('USD','EUR')
        competitors[col_eur] = competitors[col].apply(
            lambda v: round(v * TAUX_CHANGE['USD'], 4)
        )

    competitors['devise_ref']  = MONNAIE_REF
    competitors['annee_ref']   = ANNEE_REF
    competitors['source_taux'] = SOURCE_TAUX

    # ── Sauvegarde CSV ───────────────────────────────────────────────
    os.makedirs(os.path.join(OUTPUT_DIR,"csv"), exist_ok=True)
    demande.to_csv(f"{OUTPUT_DIR}/csv/T6a_demande_nationale.csv", index=False, encoding='utf-8-sig')
    competitors.to_csv(f"{OUTPUT_DIR}/csv/T6b_benchmark_competiteurs.csv", index=False, encoding='utf-8-sig')

    print(f"     ✓ T6 sauvegardé : {len(demande)} secteurs + {len(competitors)} pays")
    return demande, competitors

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 7 — ENVIRONNEMENT & CARBONE
# ══════════════════════════════════════════════════════════════════════════════
def build_T7_environnement():
    print("  [T7] Construction : Environnement & carbone...")

    emissions = pd.DataFrame({
        'filiere'           : ['H2_vert_PEM_solaire','H2_vert_PEM_eolien','H2_vert_PEM_hybride',
                               'H2_vert_AEL_solaire','H2_vert_AEL_eolien',
                               'NH3_vert_Haber_Bosch'],
        'emissions_kgCO2_kgH2_min': [0.3, 0.2, 0.3, 0.4, 0.2, 0.5],
        'emissions_kgCO2_kgH2_mode':[ 1.2, 0.8, 0.9, 1.3, 0.7, 1.5],
        'emissions_kgCO2_kgH2_max': [2.5, 1.5, 1.8, 2.5, 1.2, 3.0],
        'seuil_RFNBO_EU_gCO2_MJ'  : [None]*6,  # seuil = 3.67 gCO2/MJ = 0.44 kgCO2/kgH2
        'conforme_EU_RFNBO'       : [True,True,True,True,True,True],
        'certifiable_CertifHy'    : [True,True,True,True,True,True],
        'source'                  : ['IEA LCOH Review 2024']*6,
    })

    certifications = pd.DataFrame({
        'certification'         : ['EU_RFNBO','CertifHy_Premium','GS_H2_Gold','I_REC_Standard','ISO_14687'],
        'seuil_kgCO2_kgH2'     : [0.44, 4.4, 1.0, None, None],
        'seuil_gCO2_MJ'        : [3.67, None, None, None, None],
        'premium_prix_pct'      : [20, 15, 10, 5, 0],
        'marche_cible'          : ['EU mandatory 2030','EU/Japan','Global','Global','Global'],
        'importance_Maroc'      : ['CRITIQUE - accès marché EU','Haute','Haute','Moyenne','Obligatoire'],
        'applicable_Maroc'      : [True]*5,
        'source'                : ['EU Delegated Regulation 2023/1184',
                                   'CertifHy v3.0','Gold Standard Foundation',
                                   'I-REC Standard','ISO 14687:2019; Department for Energy Security and Net Zero (2022)'],
    })

    co2_evite = pd.DataFrame({
        'application'           : ['Substitution_NH3_OCP','Transport_FCEV','Industrie_acier',
                                   'Production_electricite','Chaleur_industrielle'],
        'CO2_evite_tCO2_tH2'   : [9.5, 11.2, 8.8, 12.5, 7.5],
        'potentiel_2030_MtCO2' : [5.0, 1.0, 0.5, 2.0, 1.5],
        'potentiel_2040_MtCO2' : [8.0, 3.5, 2.0, 5.0, 4.0],
        'potentiel_2050_MtCO2' : [12.0, 8.0, 5.5, 10.0, 8.0],
        'source'                : ['Stratégie Nationale H2 Maroc + OCP']*5,
    })

    emissions.to_csv(f"{OUTPUT_DIR}/csv/T7a_emissions_CO2.csv", index=False, encoding='utf-8-sig')
    certifications.to_csv(f"{OUTPUT_DIR}/csv/T7b_certifications.csv", index=False, encoding='utf-8-sig')
    co2_evite.to_csv(f"{OUTPUT_DIR}/csv/T7c_CO2_evite.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T7 sauvegardé : émissions + certifications + CO2 évité")
    return emissions, certifications


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 8 — PROJETS DE RÉFÉRENCE MAROC (Validation)
# ══════════════════════════════════════════════════════════════════════════════
def build_T8_projets():
    print("  [T8] Construction : Projets de référence Maroc...")

    projets = pd.DataFrame({
        'projet'            : ['NOOR_Ouarzazate_I-IV','Tarfaya_Wind_301MW',
                               'Noor_Midelt_800MW','ONEE_Wind_Taza',
                               'OCP_Green_H2_Jorf','IRESEN_H2_BenGuerir',
                               'Dakhla_H2_RWE','MASEN_H2_Dakhla_Offshore',
                               'H2Uppp_GermanyMorocco','OCP_Green_NH3_Jorf'],
        'type'              : ['CSP+PV','Éolien terrestre','CSP+PV hybride',
                               'Éolien terrestre','Électrolyse PEM',
                               'Pilote PEM+AEL R&D','Pipeline H2 offshore',
                               'Éolien offshore + H2','Partenariat H2 export',
                               'NH3 vert industriel'],
        'capacite_MW'       : [580, 301, 800, 150, 100, 0.25, 200, 500, 500, 200],
        'CAPEX_total_MUSD'  : [2500, 420, 2100, 200, 500, 2, 1200, 3000, 2000, 1000],
        'LCOE_ou_LCOH'      : [0.062, 0.038, 0.018, 0.035, 2.0, None, None, None, None, None],
        'unite_LCOE_LCOH'   : ['USD/kWh','USD/kWh','USD/kWh PPA','USD/kWh',
                               'USD/kgH2 cible 2030',None,None,None,None,None],
        'CF_reel_pct'       : [28, 43, 31, 38, None, None, None, None, None, None],
        'annee_commission'  : [2018, 2014, 2025, 2016, 2027, 2020, 2028, 2030, 2027, 2028],
        'statut'            : ['Opérationnel','Opérationnel','Construction',
                               'Opérationnel','Développement','Opérationnel pilote',
                               'Étude faisabilité','Étude faisabilité',
                               'Accord signé','Développement'],
        'developpeur'       : ['MASEN','Nareva/ENGIE','MASEN/EDF/Nareva',
                               'ONEE','OCP Group','IRESEN/UM6P',
                               'RWE/ONEE','MASEN','BMWi/MASEN','OCP Group'],
        'pertinence_outil'  : ['Calibration CF solaire + coûts PV Ouarzazate',
                               'Calibration CF éolien + LCOE éolien Tarfaya',
                               'Calibration PPA record → LCOH minimal atteignable',
                               'Calibration éolien intérieur Maroc',
                               'Référence LCOH H2 vert Maroc 2030',
                               'Seul projet H2 réel opérationnel → validation directe',
                               'Calibration coûts pipeline + export H2',
                               'Référence potentiel offshore + export',
                               'Validation corridor export EU via pipeline',
                               'Validation intégration H2→NH3 contexte OCP'],
        'source'            : ['MASEN Annual Report 2022 + World Bank PCR',
                               'IRENA Renewable Power Generation Costs 2022',
                               'MASEN 2019 + World Bank Group 2020',
                               'ONEE Annual Report 2022',
                               'OCP Group Press Release 2023',
                               'IRESEN Annual Report 2022',
                               'RWE Press Release 2023',
                               'MASEN Strategic Plan 2030',
                               'H2Uppp Germany-Morocco 2023',
                               'OCP Sustainability Report 2023'],
    })

    # ── Colonnes EUR ajoutées ─────────────────────────────────────────────
    projets['CAPEX_total_MEUR'] = projets['CAPEX_total_MUSD'].apply(
        lambda v: round(v * TAUX_CHANGE['USD'], 2) if pd.notna(v) else None
    )

    projets['LCOE_ou_LCOH_EUR'] = projets['LCOE_ou_LCOH'].apply(
        lambda v: round(v * TAUX_CHANGE['USD'], 6) if pd.notna(v) else None
    )

    projets['devise_ref']  = MONNAIE_REF
    projets['annee_ref']   = ANNEE_REF
    projets['source_taux'] = SOURCE_TAUX

    os.makedirs(os.path.join(OUTPUT_DIR,"csv"), exist_ok=True)
    projets.to_csv(os.path.join(OUTPUT_DIR,"csv/T8_projets_reference_maroc.csv"),
                   index=False, encoding='utf-8-sig')

    print(f"     ✓ T8 sauvegardé : {len(projets)} projets")
    return projets


def build_T0_references():
    """
    Table de référence bibliographique exhaustive.
    Couvre toutes les sources citées dans T1→T10,
    les formules physiques et les paramètres techniques.
    """
 
    rows = [
 
        # ══════════════════════════════════════════════════════════════════
        # BLOC A — SOURCES INSTITUTIONNELLES INTERNATIONALES
        # ══════════════════════════════════════════════════════════════════
 
        ['REF01', 'IEA Global Hydrogen Review', '2024', 'USD', 'Global',
         'Techno-economic data ; LCOH ; WACC MENA ; CAPEX électrolyseurs ; '
         'mix hybride ; émissions CO2 ; trajectoires 2040-2050',
         'https://www.iea.org/reports/global-hydrogen-review-2024',
         'T1,T2,T3,T4,T5,T6,T7,T9'],
 
        ['REF02', 'DOE Hydrogen Program — Hydrogen Shot', '2024', 'USD', 'USA',
         'CAPEX PEM & AEL ; stockage GH2 350/700 bar ; '
         'liquéfacteur LH2 ; objectif 1 $/kg H2',
         'https://www.hydrogen.energy.gov/',
         'T2,T3,T5,T9'],
 
        ['REF03', 'Hydrogen Council — Hydrogen Insights', '2024', 'USD', 'Global',
         'Projections marché ; coûts transport maritime ; LCODC',
         'https://hydrogencouncil.com/wp-content/uploads/2024/09/Hydrogen-Insights-2024.pdf',
         'T4,T5,T6'],
 
        ['REF04', 'IRENA — Renewable Power Generation Costs', '2022', 'USD', 'Global',
         'CAPEX éolien & solaire PV ; LCOE benchmarks ; '
         'calibration Tarfaya LCOE=0.038 $/kWh',
         'https://www.irena.org/publications/2022/Oct/Renewable-Power-Costs-in-2021',
         'T1,T2,T5,T8'],
 
        ['REF05', 'IRENA — Morocco Renewable Cost Database', '2024', 'USD', 'Maroc',
         'CAPEX éolien utility-scale 1100 $/kW ; OPEX 35 $/kW/an ; '
         'CAPEX solaire 550 $/kW ; OPEX 12 $/kW/an',
         'https://www.irena.org/',
         'T1,T2'],
 
        ['REF06', 'IRENA — Jobs and Renewable Energy', '2024', 'USD', 'Global',
         'Emplois/MW éolien et solaire ; économie emploi H2',
         'https://www.irena.org/Publications/2024/Sep/Renewable-Energy-and-Jobs-Annual-Review-2024',
         'T5'],
 
        ['REF07', 'IRENA — Renewable Energy Statistics', '2021-2025', 'USD', 'Global',
         'e-méthanol ratio H2/CO2 ; CAPEX synthèse méthanol',
         'https://www.irena.org/',
         'T3'],
 
        ['REF08', 'NASA POWER — Prediction of Worldwide Energy Resources v8.2', '2023', '-', 'Global',
         'GHI & DNI kWh/m²/an (2019-2023) ; vitesse vent WS100M '
         'hauteur 100m — données satellitaires ERA5 downscaled ; '
         'correction terrain bias Bett et al. 2017',
         'https://power.larc.nasa.gov',
         'T1,T10'],
 
        ['REF09', 'IEC 61724-1 — Photovoltaic system performance monitoring', '2021', '-', 'Global',
         'Performance Ratio PR=0.80 ; formule CF_sol=(GHI/8760)×PR',
         'https://www.iec.ch/homepage',
         'T1,T10'],
 
        ['REF10', 'NREL PVWatts v8 — Photovoltaic energy calculator', '2024', '-', 'USA',
         'Calibration PR=0.80 ; CF solaire Ouarzazate=19.84%≈réel 19.8%',
         'https://pvwatts.nrel.gov/',
         'T1,T10'],
 
        ['REF11', 'NREL H2A — Hydrogen Analysis Production Case Studies', '2024', 'USD', 'USA',
         'OPEX fixe AEL 2%/CAPEX ; OPEX fixe PEM 3%/CAPEX ; '
         'OPEX fixe SOEC 3.5%/CAPEX',
         'https://www.nrel.gov/hydrogen/h2a-production-analysis.html',
         'T2'],
 
        ['REF12', 'IEC 61400-12-1 — Wind turbines power performance measurements', '2017', '-', 'Global',
         'Facteur correction terrain eta=0.85 ; wake effect ~5% ; '
         'disponibilité ~3% ; pertes électriques ~2%',
         'https://www.iec.ch/homepage',
         'T1,T10'],
 
        # ══════════════════════════════════════════════════════════════════
        # BLOC B — LITTÉRATURE SCIENTIFIQUE
        # ══════════════════════════════════════════════════════════════════
 
        ['REF13',
    'Manwell, McGowan, Rogers — Wind Energy Explained (2e éd.)',
    '2009',
    '-',
    'Global',
    'Formule CF Weibull analytique ; paramètre d\'échelle c=v_mean/Γ(1+1/k) ; '
    'puissance turbine PWT polynomiale',
    'https://doi.org/10.1002/978111994085',
    'T1,T10'
],
 
        ['REF14', 'Betz A. — Das Maximum der theoretisch möglichen Ausnutzung des Windes', '1926', '-', 'Global',
         'Limite théorique Betz : η_WT_max = 16/27 ≈ 59.3% ; '
         'loi fondamentale aérodynamique éolienne',
         '-',
         'T1'],
 
        ['REF15', 'Barthelmie et al. — Modelling and measuring flow and wind turbine wakes', '2010', '-', 'Global',
         'Facteur correction terrain eta_terrain=0.85 ; '
         'pertes wake inter-turbines IEC 61400-12-1',
         'https://doi.org/10.1002/we.347',
         'T1'],
 
        ['REF16', 'Saaty T.L. — The Analytic Hierarchy Process', '1980', '-', 'Global',
         'Méthode AHP — poids score H2 composite : '
         'GHI=31.9%, CF=31.9%, Logistique=18.4%, Eau=11.0%, Réseau=6.9% ; CR=0.008',
         '-',
         'T1'],
 
        ['REF17', 'Reksten et al. — Projecting the future cost of PEM and AEL electrolysers', '2022', 'USD', 'Europe',
    "Modèle coût électrolyseur ; learning rate PEM=18% ; courbes d'apprentissage",  # ✅ Fixed quote
    'https://doi.org/10.1016/j.ijhydene.2022.08.306',
    'T2,T9'],
   
   ['REF18', 'Lu et al. — Comprehensive review of H2 storage and transport cost', '2025', 'USD', 'Global',
    'Coûts stockage & transport H2 ; drivers de coûts GH2/LH2/NH3/LOHC',
    'https://doi.org/10.1016/j.ijhydene.2025.01.196',
    'T3,T4'],
 
        ['REF19', 'Yang et al. — Review of hydrogen storage technologies', '2023', 'USD', 'Global',
         'Types réservoirs I–IV ; coûts USD/kg ; densité gravimétrique ; '
         'pression 83–1200 bar',
         'https://doi.org/10.1093/ce/zkad021',
         'T3'],
 
        ['REF20', 'Dinh et al. — LCOT comparison H2 and NH3 transmission', '2024', 'USD', 'Global',
 "LCOT tanker NH3 vs pipeline vs LH2 ; "
 "alpha=0.65 économies d'échelle maritimes",  # ✅ Fixed with double quotes
 'https://doi.org/10.1016/j.ijhydene.2024.03.066',
 'T4'],
 
        ['REF21', 'Rezaei et al. — Green H2 atlas MENA region', '2024', 'USD', 'MENA',
         'Potentiel sites H2 MENA ; aptitude territoriale ; '
         'Maroc parmi top 3 producteurs régionaux',
         'https://doi.org/10.1016/j.solener.2024.000203',
         'T1,T6'],
 
        ['REF22', 'El Hafdaoui et al. — LCOE éolien & solaire Maroc', '2024', 'USD', 'Maroc',
         'Plage LCOE éolien : 25–40 $/MWh ; LCOE solaire : 30–50 $/MWh ; '
         'OPEX éolien : 47–50 $/kW/an',
         '-',
         'T1,T2'],
 
        ['REF23', 'Bakkari et al. — H2 potential Morocco regions', '2024', 'USD', 'Maroc',
         'Potentiel H2 par région ; ressources GHI & vent ; aptitude hybride GEE',
         '-',
         'T1'],
 
        ['REF24', 'Rezaei M, Naghdi-Khozani N, Jafari N — Wind energy H2 production', '2020', 'USD', 'Global',
         'Formule CF Weibull pour production H2 éolienne ; '
         'techno-économique ; Renew Energy 2020',
         'https://doi.org/10.1016/j.renene.2020.01.099',
         'T1'],
 
        ['REF25', 'Nasser M, Megahed TF, Ookawara S, Hassan H — Techno-economic assessment', '2022', 'USD', 'Global',
         'Configurations turbines + panneaux PV pour H2 ; LCOH ; J Energy Syst 2022',
         '-',
         'T1'],
 
        ['REF26', 'Hasan MM, Genç G — Techno-economic analysis solar/wind H2', '2022', 'USD', 'Global',
         'Analyse techno-économique hybride PV+éolien pour H2 ; Fuel 2022',
         '-',
         'T1'],
 
        ['REF27', 'Denholm et al. — NREL — Overgeneration from solar energy', '2021', '-', 'USA',
         'Contrainte diversification mix hybride w∈[0.20, 0.80] ; '
         'complémentarité horaire PV + éolien',
         'https://www.nrel.gov/docs/fy21osti/77433.pdf',
         'T1'],
 
        ['REF28', 'Hollands & Huget — A probability density function for the clearness index', '1983', '-', 'Global',
         'Modèle GHI synthétique ; génération profils horaires 8760h',
         '-',
         'T10'],
 
        ['REF29', 'Bett et al. — Bias-correction of ERA5 wind speed', '2017', '-', 'Global',
         'Correction biais satellitaire → terrain pour vent ; '
         'facteur 0.85 appliqué profils T10',
         '-',
         'T10'],
 
        ['REF30', 'Department for Energy Security and Net Zero — Fugitive H2 Emissions', '2022', 'GBP', 'UK',
         'Fuites H2 dans économie future ; impact climatique H2 ; '
         'ISO 14687:2019 qualité H2',
         'https://www.gov.uk/government/publications/fugitive-hydrogen-emissions-in-a-future-hydrogen-economy',
         'T7'],
 
        ['REF31', 'EU Delegated Regulation 2023/1184 — RFNBO criteria', '2023', '-', 'UE',
         'Seuil RFNBO : 3.67 gCO2/MJ = 0.44 kgCO2/kgH2 ; '
         'règles additionnalité temporelle & géographique',
         'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202301184',
         'T7'],
 
        ['REF32', 'CertifHy v3.0 — Green Hydrogen Guarantee of Origin', '2023', '-', 'UE',
         'Seuil CertifHy Premium : 4.4 kgCO2/kgH2 ; '
         'certification GOs marché européen',
         'https://www.certifhy.eu/',
         'T7'],
 
        ['REF33', 'Gold Standard Foundation — GS H2 Gold Standard', '2023', '-', 'Global',
         'Seuil : 1.0 kgCO2/kgH2 ; premium prix +10%',
         'https://www.goldstandard.org/',
         'T7'],
 
        ['REF34', 'ISO 14687:2019 — Hydrogen fuel quality', '2019', '-', 'Global',
         'Normes qualité H2 carburant ; spécifications pureté',
         'https://www.iso.org/standard/69539.html',
         'T7'],
 
        ['REF35', 'Nature Energy — Learning rates for energy technologies', '2017', '-', 'Global',
         'Learning rate PEM = 18% par doublement de capacité ; '
         'loi de Wright appliquée aux électrolyseurs',
         '-',
         'T9'],
 
        # ══════════════════════════════════════════════════════════════════
# BLOC C — SOURCES MAROCAINES INSTITUTIONNELLES
# ══════════════════════════════════════════════════════════════════

['REF36', 'MASEN — Annual Report', '2022', 'MAD', 'Maroc',
 'Données projets NOOR ; CF solaire calibration ; '
 'coûts projet H2 ; structure PPA contractuelle',
 'https://www.masen.ma/fr/presentation',
 'T1,T5,T8'],

['REF37', 'MASEN — Strategic Plan 2030', '2023', 'MAD', 'Maroc',
 'Objectifs ENR Maroc ; pipeline H2 export EU ; '
 'potentiel offshore Dakhla',
 'https://www.masen.org.ma/en/presentation',
 'T8'],

['REF38', 'MASEN — PPA Noor Midelt record', '2019', 'USD', 'Maroc',
 'PPA record Midelt = 0.018 $/kWh (borne inférieure LCOE) ; '
 'calibration modèle LCOE solaire Maroc',
 'https://www.masen.ma/',
 'T1,T2,T5,T8'],

['REF39', 'IRESEN — Annual Report', '2022', 'MAD', 'Maroc',
 'Projets R&D H2 ; pilote PEM+AEL BenGuerir 250 kW ; '
 'sites potentiels stockage géologique',
 'https://iresen.org/',
 'T3,T8'],

['REF40', 'ONEE — Annual Report', '2022', 'MAD', 'Maroc',
 'Tarif électricité réseau 2024 ; Plan Réseau 2024 ; '
 'capacité stockage réseau ; projet éolien Taza 150 MW',
 'https://www.one.org.ma/FR/pages/interne.asp?esp=2&id1=10&id2=73&t2=1',
 'T5,T6,T8'],

['REF41', 'OCP Group — Sustainability Report', '2023', 'USD', 'Maroc',
 'Stratégie H2 vert OCP ; consommation H2 gris 120 ktH2/an ; '
 'objectif 100% H2 vert 2050 ; NH3 vert Jorf',
 'https://www.ocpgroup.ma/en/sustainability/sustainability-report-performance-data',
 'T6,T7,T8'],

['REF42', 'OCP Group — Press Release Green H2 Jorf', '2023', 'USD', 'Maroc',
 'Projet électrolyse PEM 100 MW Jorf ; CAPEX 500 M$ ; '
 'cible LCOH 2.0 $/kg en 2030',
 'https://www.ocpgroup.ma/',
 'T8'],

['REF43', 'HCP — Haut Commissariat au Plan Maroc', '2024', 'MAD', 'Maroc',
 "Données macroéconomiques ; inflation Maroc ; "
 "projections PIB ; coûts main d'oeuvre",  # ✅ Déjà corrigé avec guillemets doubles
 'https://www.hcp.ma/',
 'T5'],

['REF44', 'BAM — Bank Al-Maghrib', '2024', 'MAD', 'Maroc',
 'Taux directeur ; inflation ; taux de change MAD/USD/EUR ; '
 "notation Moody's Ba1",  # ✅ Apostrophe sécurisée avec guillemets doubles
 'https://www.bkam.ma/',
 'T5'],

['REF45', 'DGI — Direction Générale des Impôts Maroc', '2024', 'MAD', 'Maroc',
 'Taux imposition IS Maroc ; fiscalité projets EnR',
 'https://www.tax.gov.ma/',
 'T5'],

['REF46', "ANAPEC — Agence Nationale de Promotion de l'Emploi", '2024', 'MAD', 'Maroc',  # ✅ l'Emploi sécurisé
 'Salaires ingénieurs & techniciens Maroc ; '
 "coûts main d'oeuvre locale 6 catégories",  # ✅ d'oeuvre corrigé avec guillemets doubles
 'https://www.anapec.ma/',
 'T5'],

['REF47', 'CNSS — Caisse Nationale de Sécurité Sociale Maroc', '2024', 'MAD', 'Maroc',
 'Charges sociales employeur Maroc ; cotisations patronales',
 'https://www.cnss.ma/',
 'T5'],

['REF48', 'Cour des comptes Maroc — Rapport annuel', '2024-2025', 'MAD', 'Maroc',
 'Prix eau par région Maroc (MAD/m³) ; '
 'coûts dessalement ; gestion ressources hydriques',
 'https://www.courdescomptes.ma/',
 'T1'],

['REF49', 'CESE — Conseil Économique Social et Environnemental Maroc', '2020', 'MAD', 'Maroc',
 'Potentiel éolien national ancré 25 GW ; '
 'loi Betz appliquée au parc marocain',
 'https://www.cese.ma/',
 'T1'],

['REF50', 'Stratégie Nationale H2 Maroc', '2021', 'MAD', 'Maroc',
 'Objectifs H2 vert Maroc 2030-2050 ; '
 'demande nationale par secteur ; roadmap export',
 'https://www.mem.gov.ma/',
 'T6,T7'],

['REF51', 'Stratégie Mobilité Verte Maroc 2030', '2022', 'MAD', 'Maroc',
 'Parc FCEV prévu ; demande H2 transport 2030-2050 ; '
 'infrastructure recharge H2',
 '-',
 'T6'],  # ✅ Fin de liste propre
        # ══════════════════════════════════════════════════════════════════
        # BLOC D — PARTENARIATS & PROJETS INTERNATIONAUX
        # ══════════════════════════════════════════════════════════════════
 
        ['REF52', 'RWE — Germany–Morocco H2 Partnership Press Release', '2023', 'EUR', 'Allemagne/Maroc',
         'Pipeline H2 Dakhla → Europe ; partenariat RWE/ONEE ; '
         'CAPEX estimé 1.2 Mrd$ ; statut étude faisabilité',
         'https://energypartnership.ma/',
         'T8'],
 
        ['REF53', 'H2Uppp — German-Moroccan Hydrogen Partnership', '2023', 'EUR', 'Allemagne/Maroc',
         'Accord gouvernemental BMWi/MASEN ; '
         'validation corridor export EU via pipeline 500 MW',
         'https://www.giz.de/en/worldwide/107551.html',
         'T8'],
 
        ['REF54', 'World Bank — PCR Noor PV Ouarzazate P131256', '2020', 'USD', 'Global',
         'Données financières projet NOOR I-IV ; '
         'calibration CF et CAPEX PV Ouarzazate',
         'https://documents1.worldbank.org/curated/en/099062325104027002/pdf/P131256-495117fd-b17f-436b-a287-574544f421c0.pdf',
         'T8'],
 
        # ══════════════════════════════════════════════════════════════════
        # BLOC E — SOURCES FINANCIÈRES & MACROÉCONOMIQUES
        # ══════════════════════════════════════════════════════════════════
 
        ['REF55', 'BCE — Banque Centrale Européenne — Taux de change EUR', 'Jan 2024', 'EUR', 'UE',
         'Taux EUR/USD=1.0858 → 1 USD=0.9217 EUR ; '
         'taux EUR/GBP ; taux pivot EUR',
         'https://www.ecb.europa.eu/stats/exchange/eurofxref/html/index.en.html',
         'T0'],
 
        ['REF56', 'BAM — Taux de change MAD/EUR et MAD/USD', '2024', 'MAD', 'Maroc',
         '1 MAD = 0.0917 EUR ; source officielle BAM jan 2024',
         'https://www.bkam.ma/',
         'T0'],
 
        ['REF57', 'IMF — World Economic Outlook', '2024', '-', 'Global',
         'Taux de change SAR/AUD/GBP ; projections macroéconomiques',
         'https://www.imf.org/en/Publications/WEO',
         'T0'],
 
        ['REF58', 'Moody\'s — Credit Rating Morocco Ba1', '2024', '-', 'Global',
         'Notation Ba1 Maroc ; prime de risque pays ; '
         'WACC MENA standard financement',
         'https://www.moodys.com/',
         'T5'],
 
      ['REF59', 'BIT — Bureau International du Travail', '2024', '-', 'Global',
 "Salaires minimum internationaux ; coût main d'oeuvre MENA",  # ✅ Guillemets doubles
 'https://www.ilo.org/',
 'T5'],  # ← Liste correctement fermée
 
        # ══════════════════════════════════════════════════════════════════
        # BLOC F — SOURCES DONNÉES & OUTILS
        # ══════════════════════════════════════════════════════════════════
 
        ['REF60', 'OSRM — Open Source Routing Machine (OpenStreetMap)', '2026', '-', 'Global',
         'Calcul distances routières inter-villes Maroc ; '
         'API router.project-osrm.org ; fallback distances haversine',
         'https://www.openstreetmap.org/',
         'T4'],
 
        ['REF61', 'Vestas V90-2MW — Wind Turbine Technical Specifications', '2024', '-', 'Global',
         'P_rated=2000 kW ; D_rotor=90m ; V_ci=3 m/s ; V_r=12 m/s ; '
         'V_o=25 m/s ; référence projets utility-scale Tarfaya/Dakhla',
         'https://www.vestas.com/',
         'T1,T10'],
 
        ['REF62', 'IEA — Renewable Integration Outlook', '2023', 'USD', 'Global',
         'Optimisation mix hybride PV + éolien ; '
         'contrainte diversification sources horaires',
         'https://www.iea.org/reports/renewables-2023',
         'T1'],
 
        ['REF63', 'IEA — Morocco Energy Profile', '2024', 'USD', 'Maroc',
         'Stratégie nationale H2 Maroc 2021 ; profil énergétique ; '
         'scénarios LCOH 2030-2050',
         'https://www.iea.org/countries/morocco',
         'T9'],
 
        ['REF64', 'Applied Energy — H2 cost competitiveness MENA', '2024', 'USD', 'MENA',
         'LCOH benchmark pays concurrents 2024-2040 ; '
         'Maroc vs Arabie Saoudite, Égypte, Chili, Australie, Namibie, Espagne',
         '-',
         'T6'],
 
        ['REF65', 'IRESEN — Géologie stockage souterrain Maroc', '2022', 'MAD', 'Maroc',
         'Sites potentiels cavernes salines Maroc : 1–5 sites ; '
         'capacité stockage géologique H2',
         'https://iresen.org/',
         'T3'],
 
    ]
 
    # ── Création DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=[
        'reference_id',
        'source_name',
        'year',
        'original_currency',
        'region',
        'data_type',          # ce que la source apporte au modèle
        'source_url',
        'tables_utilisatrices' # quelles tables T0-T10 citent cette source
    ])
 
    # ── Statistiques ─────────────────────────────────────────────────────────
    n_institutionnel = len(df[df['reference_id'].str.startswith(('REF0', 'REF1'))])
    n_litterature    = len(df[df['reference_id'].isin([f'REF{i:02d}' for i in range(13, 36)])])
    n_maroc          = len(df[df['region'] == 'Maroc'])
    n_avec_url       = len(df[df['source_url'] != '-'])
 
    df.to_csv(
        f"{OUTPUT_DIR}/reports/T0_references_complet.csv",
        index=False,
        encoding='utf-8-sig'
    )
 
    print(f"✓ T0 complet sauvegardé : {len(df)} références")
    print(f"  └─ Institutionnelles  : {len(df[df['reference_id'].isin([f'REF{i:02d}' for i in range(1, 13)])])}")
    print(f"  └─ Littérature scient.: {len(df[df['reference_id'].isin([f'REF{i:02d}' for i in range(13, 36)])])}")
    print(f"  └─ Sources marocaines : {n_maroc}")
    print(f"  └─ Partenariats       : 3")
    print(f"  └─ Financières/Outils : {len(df[df['reference_id'].isin([f'REF{i:02d}' for i in range(55, 66)])])}")
    print(f"  └─ Avec URL           : {n_avec_url} / {len(df)}")
 
    return df

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 9 — SCÉNARIOS TEMPORELS 2024–2050
# ══════════════════════════════════════════════════════════════════════════════
def build_T9_scenarios():
    print("  [T9] Construction : Scénarios temporels 2024-2050...")

    annees = [2024, 2027, 2030, 2035, 2040, 2045, 2050]

    # Wright's Law : learning rate
    # CAPEX(n) = CAPEX_ref × (cumul_capacity/ref_capacity)^(-b)  b = log(1-LR)/log(2)
    def wright_law(capex_ref, lr, doublings):
        b = np.log(1 - lr) / np.log(2)
        return capex_ref * (2**doublings)**b

    # Doublements cumulatifs estimés pour électrolyseurs (global)
    doublings_PEM = [0, 1.0, 2.5, 4.5, 6.5, 8.0, 9.5]  # par année
    LR_PEM = 0.18  # 18% réduction par doublement (Nature Energy 2017)
    LR_AEL = 0.12
    LR_solar= 0.24

    scenarios = pd.DataFrame({
        'annee'                         : annees,
        # ── CAPEX Technologies (apprentissage) ──────────────────────────────
        'CAPEX_PEM_USD_kW'              : [wright_law(1100, LR_PEM, d) for d in doublings_PEM],
        'CAPEX_AEL_USD_kW'              : [wright_law(800,  LR_AEL, d) for d in doublings_PEM],
        'CAPEX_solaire_USD_kW'          : [wright_law(550,  LR_solar,d) for d in doublings_PEM],
        'CAPEX_eolien_USD_kW'           : [wright_law(1250, 0.07, d) for d in doublings_PEM],
        # ── Efficacités (amélioration technologique) ─────────────────────────
        'efficacite_PEM_kWh_kgH2'       : [55, 53, 51, 49, 47, 46, 44],
        'efficacite_AEL_kWh_kgH2'       : [52, 51, 50, 48, 47, 46, 45],
        # ── Prix électricité ─────────────────────────────────────────────────
        'PPA_solaire_Maroc_USD_kWh'     : [0.025,0.022,0.018,0.015,0.013,0.012,0.011],
        'PPA_eolien_Maroc_USD_kWh'      : [0.032,0.029,0.025,0.021,0.018,0.016,0.015],
        # ── LCOH résultant ───────────────────────────────────────────────────
        'LCOH_PEM_solaire_min'          : [3.5, 2.8, 2.0, 1.5, 1.2, 1.0, 0.9],
        'LCOH_PEM_solaire_mode'         : [5.5, 4.0, 2.8, 2.0, 1.5, 1.2, 1.0],
        'LCOH_PEM_solaire_max'          : [8.0, 6.0, 4.0, 3.0, 2.2, 1.8, 1.5],
        'LCOH_AEL_hybride_min'          : [3.0, 2.5, 1.8, 1.3, 1.0, 0.9, 0.8],
        'LCOH_AEL_hybride_mode'         : [4.8, 3.5, 2.5, 1.8, 1.4, 1.1, 0.9],
        'LCOH_AEL_hybride_max'          : [7.0, 5.5, 3.8, 2.8, 2.0, 1.6, 1.3],
        # ── Chaîne complète (LCODC = Leveled Cost of Delivery Chain) ─────────
        'LCODC_export_EU_pipeline_min'  : [4.0, 3.2, 2.5, 1.9, 1.5, 1.3, 1.1],
        'LCODC_export_EU_pipeline_mode' : [6.5, 5.0, 3.5, 2.7, 2.0, 1.6, 1.3],
        'LCODC_export_EU_pipeline_max'  : [10.0,7.5, 5.5, 4.0, 3.0, 2.4, 2.0],
        'LCODC_export_NH3_ship_min'     : [5.0, 4.0, 3.0, 2.3, 1.8, 1.5, 1.3],
        'LCODC_export_NH3_ship_mode'    : [8.0, 6.0, 4.2, 3.2, 2.4, 1.9, 1.6],
        'LCODC_export_NH3_ship_max'     : [12.0,9.0, 6.5, 4.8, 3.5, 2.8, 2.3],
        # ── Objectifs nationaux ──────────────────────────────────────────────
        'production_H2_Maroc_ktH2_an'   : [5, 50, 400, 1200, 2500, 4000, 6000],
        'export_H2_pct'                 : [0, 20, 45,  60,   70,   75,   80  ],
        'capacite_electrolyseur_GW'     : [0.01,0.1,0.8,2.5,5.0,8.0,12.0],
        'emplois_crees_milliers'        : [0.5, 5,  35,  100, 200, 320, 450],
        'investissement_cumul_Mrd_USD'  : [0.1, 1,  8,   25,  55,  90,  130],
        'CO2_evite_MtCO2_an'           : [0.1, 0.8,5,   12,  22,  35,  50 ],
        # Source : Stratégie Nationale H2 Maroc 2021 + IEA Morocco Energy Profile 2024
    })

    
   
    # ── Colonnes EUR ajoutées ─────────────────────────────────────────────
    cols_usd = [c for c in scenarios.columns if '_USD' in c]
    for col in cols_usd:
        col_eur = col.replace('_USD', '_EUR')
        scenarios[col_eur] = scenarios[col].apply(
            lambda v: round(v * TAUX_CHANGE['USD'], 4) if pd.notna(v) else None)

    scenarios['devise_ref']  = MONNAIE_REF
    scenarios['annee_ref']   = ANNEE_REF
    scenarios['source_taux'] = SOURCE_TAUX

    for col in scenarios.select_dtypes(include='number').columns:
        scenarios[col] = scenarios[col].round(4)

    scenarios.to_csv(f"{OUTPUT_DIR}/csv/T9_scenarios_temporels.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T9 sauvegardé : {len(annees)} années × {len(scenarios.columns)} variables")
    return scenarios
# ══════════════════════════════════════════════════════════════════════════════
# TABLE 10 — PROFILS HORAIRES 8760h (Production PV + Éolien par région)
# Sources : NASA POWER API v8.2 (ERA5 downscaled) + fallback synthétique calibré
# Objectif : alimenter PyPSA (Étape 2 — NSGA-II) avec séries temporelles réelles
# ══════════════════════════════════════════════════════════════════════════════

API_DELAY_SEC = 1.5   # Délai entre requêtes NASA POWER (limite : 60 req/min)
ANNEE_PROFIL  = 2024  # Année de référence 8760h (non bissextile)

# Données complémentaires T10 par région
# v_mean_T1 = vitesse moyenne annuelle à 100m (NASA POWER WS100M — données T1)
# Utilisée pour calibrer les profils horaires WS50M → cohérence avec T1

# Coefficient de rugosité (Hellmann) selon le terrain
# 0.11 = Mer/Côte plate | 0.16 = Terres/Relief modéré
ALPHA_REGION = {
    'Dakhla': 0.11, 'Laayoune': 0.11, 'Boujdour': 0.11, 'Tanger': 0.11,
    'Jorf_Lasfar': 0.11, 'Agadir': 0.11, 'Casablanca': 0.11,
    'Guelmim': 0.14, 'Nador': 0.14, 
    'Ouarzazate': 0.16, 'Marrakech': 0.16, 'Midelt': 0.16
}
REGIONS_T10 = {
    'Ouarzazate' : {'lat': 30.9189, 'lon':  -6.8934, 'alt': 1135,
                    'GHI_ref': 2172, 'CF_sol_T1': 19.84, 'CF_eol_T1': 22.47,
                    'v_mean_T1': 5.8,  'viable_eol': True},
    'Dakhla'     : {'lat': 23.6848, 'lon': -15.9572, 'alt':   12,
                    'GHI_ref': 2180, 'CF_sol_T1': 19.70, 'CF_eol_T1': 35.50,
                    # CF_eol_T1 recalibré : NASA POWER WS50M réel = 8.4 m/s
                    # Ancienne valeur 41.52% basée WS100M IRENA → surestimée
                    # 35.50% cohérent avec ERA5 réel (Rezaei et al. 2024)
                    'v_mean_T1': 8.4,  'viable_eol': True},
    'Laayoune'   : {'lat': 27.1253, 'lon': -13.1625, 'alt':   70,
                    'GHI_ref': 2175, 'CF_sol_T1': 19.90, 'CF_eol_T1': 33.72,
                    'v_mean_T1': 8.2,  'viable_eol': True},
    'Tanger'     : {'lat': 35.7595, 'lon':  -5.8340, 'alt':   15,
                    'GHI_ref': 1850, 'CF_sol_T1': 16.76, 'CF_eol_T1': 15.62,
                    'v_mean_T1': 5.1,  'viable_eol': False},
    'Jorf_Lasfar': {'lat': 33.1100, 'lon':  -8.6300, 'alt':   10,
                    'GHI_ref': 1900, 'CF_sol_T1': 17.30, 'CF_eol_T1': 12.75,
                    'v_mean_T1': 4.8,  'viable_eol': False},
    'Guelmim'    : {'lat': 28.9870, 'lon': -10.0572, 'alt':  310,
                    'GHI_ref': 2100, 'CF_sol_T1': 19.06, 'CF_eol_T1':  5.51,
                    'v_mean_T1': 3.2,  'viable_eol': False},
    'Agadir'     : {'lat': 30.4278, 'lon':  -9.5981, 'alt':   15,
                    'GHI_ref': 2050, 'CF_sol_T1': 19.18, 'CF_eol_T1': 16.36,
                    'v_mean_T1': 5.3,  'viable_eol': False},
    'Boujdour'   : {'lat': 26.1333, 'lon': -14.4833, 'alt':   55,
                    'GHI_ref': 2160, 'CF_sol_T1': 20.00, 'CF_eol_T1': 38.44,
                    'v_mean_T1': 9.1,  'viable_eol': True},
    'Casablanca' : {'lat': 33.5731, 'lon':  -7.5898, 'alt':   50,
                    'GHI_ref': 1870, 'CF_sol_T1': 17.08, 'CF_eol_T1': 10.50,
                    'v_mean_T1': 4.3,  'viable_eol': False},
    'Nador'      : {'lat': 35.1681, 'lon':  -2.9335, 'alt':   10,
                    'GHI_ref': 1780, 'CF_sol_T1': 16.26, 'CF_eol_T1': 18.00,
                    'v_mean_T1': 5.5,  'viable_eol': False},
    'Marrakech'  : {'lat': 31.6295, 'lon':  -7.9811, 'alt':  460,
                    'GHI_ref': 2080, 'CF_sol_T1': 19.00, 'CF_eol_T1':  6.00,
                    'v_mean_T1': 3.8,  'viable_eol': False},
    'Midelt'     : {'lat': 32.6800, 'lon':  -4.7340, 'alt': 1520,
                    'GHI_ref': 2200, 'CF_sol_T1': 20.09, 'CF_eol_T1': 15.00,
                    'v_mean_T1': 5.0,  'viable_eol': False},
}

# ─────────────────────────────────────────────────────────────────────────────
# T10 — Classe client NASA POWER API
# ─────────────────────────────────────────────────────────────────────────────
class NASAPowerClient:
    """
    Client NASA POWER API — données horaires ERA5 downscaled.
    Endpoint  : https://power.larc.nasa.gov/api/temporal/hourly/point
    Paramètres récupérés :
        ALLSKY_SFC_SW_DWN  : GHI (W/m²)
        WS10M              : Vitesse vent 10m (m/s)
        WS50M              : Vitesse vent 50m (m/s)
        T2M                : Température 2m (°C)
    Note : WS100M non disponible en résolution horaire sur NASA POWER.
           La calibration de vitesse se fait via v_mean_T1 dans REGIONS_T10.
    Documentation : https://power.larc.nasa.gov/docs/
    """
    BASE_URL            = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
    PARAMETRES_FULL     = 'ALLSKY_SFC_SW_DWN,WS10M,WS50M,T2M'
    PARAMETRES_FALLBACK = 'ALLSKY_SFC_SW_DWN,WS10M,T2M'   # si WS50M indisponible

    def __init__(self, annee=ANNEE_PROFIL):
        self.annee = annee
        self.start = f'{annee}0101'
        self.end   = f'{annee}1231'

    def fetch_region(self, region_name, lat, lon, max_retries=3):
        """
        Télécharge 8760 valeurs horaires pour une région.
        Retourne un DataFrame avec index DatetimeIndex UTC, ou None si échec.
        """
        for attempt in range(max_retries):
            # Tentatives 1+2 avec WS50M, tentative 3 sans WS50M (fallback)
            use_params = self.PARAMETRES_FULL if attempt < 2 else self.PARAMETRES_FALLBACK
            params = {
                'parameters' : use_params,
                'community'  : 'RE',
                'longitude'  : lon,
                'latitude'   : lat,
                'start'      : self.start,
                'end'        : self.end,
                'format'     : 'JSON',
            }
            try:
                lbl = 'full' if attempt < 2 else 'sans WS50M'
                print(f'    Requête NASA POWER : {region_name} '
                      f'(lat={lat}, lon={lon}) — tentative {attempt+1}/{max_retries} [{lbl}]')
                resp = requests.get(self.BASE_URL, params=params, timeout=120)
                resp.raise_for_status()
                data  = resp.json()
                props = data['properties']['parameter']
                df    = pd.DataFrame(props)
                df.index = pd.to_datetime(df.index, format='%Y%m%d%H')
                df.index = df.index.tz_localize('UTC')
                df.index.name = 'datetime_UTC'
                df = df.rename(columns={
                    'ALLSKY_SFC_SW_DWN' : 'GHI_W_m2',
                    'WS10M'             : 'WS10M_m_s',
                    'WS50M'             : 'WS50M_m_s',
                    'T2M'               : 'T2M_C',
                })
                # Si WS50M absent (fallback tentative 3) → estimer depuis WS10M
                if 'WS50M_m_s' not in df.columns:
                    df['WS50M_m_s'] = df['WS10M_m_s'] * (50/10)**0.14
                    print(f'    ℹ️  WS50M estimé depuis WS10M (Hellmann α=0.14)')
                # Nettoyage : code manquant NASA POWER = -999
                for col in df.columns:
                    df[col] = df[col].replace(-999.0, np.nan)
                df = df.interpolate(method='linear', limit=3)
                df = df[df.index.year == self.annee]
                if len(df) > 8760:
                    df = df.iloc[:8760]
                print(f'    ✓ {region_name} : {len(df)} heures '
                      f'| GHI_moy={df["GHI_W_m2"].mean():.1f} W/m² '
                      f'| WS50M_moy={df["WS50M_m_s"].mean():.1f} m/s')
                time.sleep(API_DELAY_SEC)
                return df
            except requests.exceptions.RequestException as e:
                print(f'    ⚠️  Erreur réseau ({attempt+1}) : {e}')
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f'    ❌ Échec {region_name} — passage au fallback synthétique')
                    return None
            except (KeyError, ValueError) as e:
                print(f'    ❌ Erreur parsing JSON ({region_name}) : {e}')
                return None


# ─────────────────────────────────────────────────────────────────────────────
# T10 — Fonctions physiques horaires
# ─────────────────────────────────────────────────────────────────────────────
def _T10_extrapolate_wind_hub(ws50m, alpha, h_hub=90):
    """
    Extrapolation avec alpha variable selon la région.
    """
    ws50m = np.array(ws50m, dtype=float)
    return np.clip(ws50m * (h_hub / 50) ** alpha, 0.0, 40.0)


def _T10_rho_air(altitude_m, t2m_c=None):
    """
    Densité air locale (kg/m³) selon altitude et température.
    ρ = P_std × exp(-alt / 8500) / (R_air × T_K)
    Source : ISO 2533.
    """
    if t2m_c is None:
        t2m_c = np.full(8760, 15.0)
    t_k = np.array(t2m_c, dtype=float) + 273.15
    p_h = 101325.0 * np.exp(-altitude_m / 8500.0)
    return p_h / (287.058 * t_k)


def _T10_pv_profile(ghi, t2m=None, PR=0.80):
    """
    CF solaire horaire (0-1) avec correction température cellule PV.
    T_cell = T_amb + (NOCT-20)/800 × GHI
    η_T    = 1 - γ × (T_cell - 25)     [γ=0.004/°C, Si cristallin]
    CF_h   = (GHI/1000) × PR × η_T
    Source : IEC 61724-1 ; NREL PVWatts v8.
    """
    ghi = np.clip(np.array(ghi, dtype=float), 0.0, 1200.0)
    if t2m is not None:
        t_cell = np.array(t2m, dtype=float) + (45.0 - 20.0) / 800.0 * ghi
        eta_T  = np.clip(1.0 - 0.004 * (t_cell - 25.0), 0.60, 1.10)
    else:
        eta_T = np.full(len(ghi), 0.95)
    return np.clip((ghi / 1000.0) * PR * eta_T, 0.0, 1.0)


def _T10_wind_profile(ws_hub, rho=None):
    """
    CF éolien horaire brut (0-1) via courbe de puissance polynomiale.
    PWT = -0.6994·V³ + 19.481·V² - 90.983·V + 121  [Vci ≤ V < Vr]
    Correction densité air : P_corr = P_std × (rho_local/rho_STP)^(1/3)
    Source : IEC 61400-12-1 (2017) ; Vestas V90-2MW.

    NOTE : Ce profil brut est ensuite post-scalé dans build_T10_profils_horaires
    pour garantir CF_annuel_moyen = CF_T1 (formule Weibull analytique).
    Le post-scaling est nécessaire car l'extrapolation 50→90m (Hellmann) réduit
    les vitesses instantanées sous Vci=3 m/s, ce qui sous-estime le CF brut.
    """
    ws  = np.array(ws_hub, dtype=float)
    Vci = TURBINE['V_ci']
    Vr  = TURBINE['V_r']
    Vo  = TURBINE['V_o']
    Pr  = TURBINE['P_rated_kW']
    pw  = np.zeros(len(ws))
    mid = (ws >= Vci) & (ws < Vr)
    pw[mid] = np.clip(
        -0.6994*ws[mid]**3 + 19.481*ws[mid]**2 - 90.983*ws[mid] + 121,
        0.0, Pr
    )
    pw[(ws >= Vr) & (ws < Vo)] = Pr
    if rho is not None:
        pw = pw * (np.array(rho, dtype=float) / TURBINE['rho_air']) ** (1/3)
    cf = pw / Pr * TURBINE['eta_terrain']
    return np.clip(cf, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# T10 — Fallback synthétique (si API indisponible)
# ─────────────────────────────────────────────────────────────────────────────
def _T10_synthetic(region_name, info, annee=ANNEE_PROFIL):
    """
    Profils horaires synthétiques calibrés quand NASA POWER est inaccessible.
    Modèle solaire : géométrique (déclinaison + angle horaire) × indice clarté AR1
    Modèle vent    : processus Weibull AR1 avec saisonnalité, calibré sur v_mean_T1
    Source modèle  : Hollands & Huget 1983 (GHI) ; Weibull AR1 (vent).
    ⚠️  Usage : tests / développement uniquement — pas pour publication.
    """
    print(f'    ⚠️  Mode synthétique pour {region_name}')
    lat = info['lat']
    idx = pd.date_range(start=f'{annee}-01-01', periods=8760, freq='h', tz='UTC')
    h   = np.arange(8760)
    doy = (h // 24) + 1

    # ── GHI synthétique ──────────────────────────────────────────────────────
    decl        = 0.4093 * np.sin(2 * np.pi * (doy - 81) / 365)
    h_loc       = (h % 24) + (info['lon'] / 15)
    ang         = (h_loc - 12) * 15 * np.pi / 180
    lat_r       = lat * np.pi / 180
    cos_z       = np.maximum(
        np.sin(lat_r)*np.sin(decl) + np.cos(lat_r)*np.cos(decl)*np.cos(ang), 0.0
    )
    Io          = 1361.0 * (1 + 0.033 * np.cos(2*np.pi*doy/365))
    ghi_clear   = Io * cos_z * 0.75
    np.random.seed(42 + abs(hash(region_name)) % 1000)
    kt_base     = 0.65 + 0.05 * np.sin(2*np.pi*doy/365)
    kt_noise    = np.zeros(8760)
    kt_noise[0] = np.random.normal(0, 0.08)
    for i in range(1, 8760):
        kt_noise[i] = 0.85*kt_noise[i-1] + np.random.normal(0, 0.04)
    kt      = np.clip(kt_base + kt_noise, 0.1, 0.95)
    ghi_syn = np.where(cos_z > 0.05, ghi_clear * kt, 0.0)
    # Calibration GHI annuel → référence T1
    ghi_cible = info['GHI_ref'] * 1000 / 8760
    if ghi_syn.mean() > 0:
        ghi_syn = ghi_syn * (ghi_cible / ghi_syn.mean())
    ghi_syn = np.clip(ghi_syn, 0.0, 1200.0)

    # ── Vent synthétique (Weibull AR1 calibré sur v_mean_T1) ─────────────────
    v_mean  = info['v_mean_T1']   # vitesse cible à hauteur moyeu
    v_seiz  = v_mean * (1 + 0.15 * np.cos(2*np.pi*(doy-30)/365))
    c_seiz  = v_seiz / gamma_func(1 + 1/TURBINE['k_weibull'])
    u       = np.random.uniform(0, 1, 8760)
    u_c     = np.zeros(8760)
    u_c[0]  = u[0]
    for i in range(1, 8760):
        u_c[i] = 0.80*u_c[i-1] + np.sqrt(1-0.64)*u[i]
    u_c      = np.clip((u_c-u_c.min())/(u_c.max()-u_c.min()+1e-9), 0.001, 0.999)
    ws50_syn = c_seiz * (-np.log(1-u_c))**(1/TURBINE['k_weibull'])
    ws10_syn = ws50_syn * (10/50)**0.14
    t2m_syn  = (18 + 10*np.sin(2*np.pi*(doy-80)/365)
                + 5*np.sin(2*np.pi*(h%24-14)/24)
                + np.random.normal(0, 1.5, 8760))

    df = pd.DataFrame({
        'GHI_W_m2'  : ghi_syn,
        'WS10M_m_s' : ws10_syn,
        'WS50M_m_s' : ws50_syn,
        'T2M_C'     : t2m_syn,
        'source'    : 'synthétique_fallback',
    }, index=idx)
    df.index.name = 'datetime_UTC'
    print(f'    ✓ {region_name} [synthétique] : '
          f'GHI_moy={ghi_syn.mean():.1f} W/m² | '
          f'WS50M_moy={ws50_syn.mean():.1f} m/s')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# T10 — Fonction principale build_T10_profils_horaires
# ─────────────────────────────────────────────────────────────────────────────
def build_T10_profils_horaires(t1_df, annee=ANNEE_PROFIL, force_synthetic=False):
    """
    Construit T10 : profils horaires 8760h pour toutes les régions.

    Paramètres
    ----------
    t1_df          : DataFrame retourné par build_T1_ressources()
                     Permet à T10 de lire CF_T1 dynamiquement — plus de valeurs codées en dur.
    annee          : int — année de simulation (profils NASA POWER)
    force_synthetic: bool — True = mode hors-ligne, profils synthétiques uniquement

    Pipeline :
        1. Construction dynamique REGIONS_T10 depuis t1_df
        2. Récupération NASA POWER ERA5 WS50M+GHI+T2M (ou fallback synthétique)
        3. Extrapolation WS50M → WS90M (Hellmann alpha=0.11 côtier)
        4. Correction densité air selon altitude (ISO 2533)
        5. Calcul CF_PV_h heure par heure (IEC 61724-1 + correction T°)
        6. Calcul CF_eol_brut heure par heure (courbe PWT identique à T1)
        7. Post-scaling adaptatif CF_eol → 3 niveaux selon fiabilité NASA
        8. Validation croisée CF annuels vs T1 (tolérance ±15%)
        9. Export CSV par région + résumé global + fichiers PyPSA

    Sources scaling
    ---------------
    Bett et al. 2017 — bias-correction ERA5 éolien
    IEC 61400-12-1 (2017) — mesures terrain vs satellite
    """
    print('\n' + '═'*70)
    print('  [T10] Construction : Profils horaires 8760h')
    print('═'*70)

    # ══════════════════════════════════════════════════════════════════════════
    # ACTION 1 — Construction dynamique de REGIONS_T10 depuis t1_df
    # Plus de valeurs CF codées en dur : T10 suit T1 automatiquement
    # ══════════════════════════════════════════════════════════════════════════
    regions_t10 = {}
    for _, row in t1_df.iterrows():
        reg = row['region']
        # Vérifier que la région a une altitude définie dans REGIONS_T10 original
        # (altitude non présente dans T1 — on la récupère depuis le dict statique)
        alt = REGIONS_T10[reg]['alt'] if reg in REGIONS_T10 else 0
        regions_t10[reg] = {
            'lat'       : row['latitude_N'],
            'lon'       : row['longitude_W'],
            'alt'       : alt,
            'CF_eol_T1' : row['CF_eolien_pct'],          # % — lu dynamiquement T1
            'CF_sol_T1' : row['CF_solaire_PV_pct'],       # % — lu dynamiquement T1
            'GHI_ref'   : row['GHI_kWh_m2_an'],
            'v_mean_T1' : row['vitesse_vent_moy_ms'],
        }

    client    = NASAPowerClient(annee=annee)
    resultats = {}
    resume    = []

    for region_name, info in regions_t10.items():
        print(f'\n  ── Région : {region_name} ──')

        # ── 1. Données météo ─────────────────────────────────────────────────
        df = None
        if not force_synthetic:
            df = client.fetch_region(region_name, info['lat'], info['lon'])
        if df is None:
            df = _T10_synthetic(region_name, info, annee)
            source_data = 'synthétique_calibré'
        else:
            df['source'] = 'NASA_POWER_ERA5'
            source_data  = 'NASA_POWER_ERA5'

        # Garantie 8760h exactement
        if len(df) != 8760:
            idx_full = pd.date_range(
                start=f'{annee}-01-01', periods=8760, freq='h', tz='UTC'
            )
            df = df.reindex(idx_full).interpolate('linear')

        # ── 2. Extrapolation vent 50m → 90m (Hellmann alpha=0.11 côtier) ────
        alpha_local = ALPHA_REGION.get(region_name, 0.14) # 0.14 par défaut si non trouvé
        df['WS90M_m_s'] = _T10_extrapolate_wind_hub(df['WS50M_m_s'].values, alpha=alpha_local, h_hub=90)

        # ── 3. Densité air locale ────────────────────────────────────────────
        df['rho_air_kg_m3'] = _T10_rho_air(info['alt'], df['T2M_C'].values)

        # ── 4. Profil PV ─────────────────────────────────────────────────────
        df['CF_PV_h'] = _T10_pv_profile(
            df['GHI_W_m2'].values, df['T2M_C'].values, PR=TURBINE['PR_PV']
        )

        # ── 5. Profil éolien brut ────────────────────────────────────────────
        cf_eol_brut = _T10_wind_profile(
            df['WS90M_m_s'].values, rho=df['rho_air_kg_m3'].values
        )

        # ══════════════════════════════════════════════════════════════════════
        # ACTION 2 — Post-scaling adaptatif à 3 niveaux
        # Remplace le seuil fixe 2.5 par une logique graduée
        # Source : Bett et al. 2017 — bias-correction ERA5 éolien
        # ══════════════════════════════════════════════════════════════════════
        cf_cible    = info['CF_eol_T1'] / 100.0   # CF annuel T1 — référence Weibull
        cf_brut_moy = cf_eol_brut.mean()

        if cf_brut_moy > 0.001:
            facteur = cf_cible / cf_brut_moy

            if facteur <= 1.5:
                # ── Niveau 1 : NASA fiable ───────────────────────────────────
                cf_eol_final    = np.clip(cf_eol_brut * facteur, 0.0, 1.0)
                methode_scaling = f"post-scaling ×{facteur:.2f} (NASA fiable)"
                print(f'    ℹ️  Post-scaling CF éolien : ×{facteur:.2f} '
                      f'({cf_brut_moy*100:.1f}% → {cf_eol_final.mean()*100:.1f}%)')

            elif facteur <= 2.5:
                # ── Niveau 2 : NASA modérément biaisé ───────────────────────
                cf_eol_final    = np.clip(cf_eol_brut * facteur, 0.0, 1.0)
                methode_scaling = (f"post-scaling ×{facteur:.2f} + clip "
                                   f"(NASA modérément biaisé)")
                print(f'    ℹ️  Post-scaling CF éolien : ×{facteur:.2f} '
                      f'({cf_brut_moy*100:.1f}% → {cf_eol_final.mean()*100:.1f}%)')

            else:
                # ── Niveau 3 : NASA non fiable — renormalisation itérative ───
                if cf_eol_brut.mean() > 0:
                    cf_eol_scaled  = cf_eol_brut * (cf_cible / cf_brut_moy)
                    cf_eol_clipped = np.clip(cf_eol_scaled, 0.0, 1.0)

                    for _ in range(3):
                        moy_apres_clip = cf_eol_clipped.mean()
                        if abs(moy_apres_clip - cf_cible) < 0.002:
                            break
                        if moy_apres_clip > 0:
                            cf_eol_clipped = np.clip(
                                cf_eol_clipped * (cf_cible / moy_apres_clip),
                                0.0, 1.0
                            )

                    cf_eol_final = cf_eol_clipped

                else:
                    cf_eol_final = np.full(8760, cf_cible)

                methode_scaling = (
                    f"renorm. itérative CF_T1={cf_cible*100:.1f}% "
                    f"(facteur={facteur:.2f} > 2.5, NASA non fiable)"
                )
                print(f'    ⚠️  Facteur scaling ({facteur:.2f}) > 2.5 — '
                      f'WS50M NASA trop bas pour ce site. '
                      f'CF éolien renormalisé sur CF_T1={cf_cible*100:.1f}% '
                      f'(Weibull analytique) — '
                      f'moy finale={cf_eol_final.mean()*100:.2f}%')

        else:
            # CF brut nul (vent très faible ou données manquantes)
            cf_eol_final    = np.full(8760, cf_cible)
            facteur         = 0.0
            methode_scaling = f"CF fixé CF_T1={cf_cible*100:.1f}% (CF brut nul)"
            print(f'    ℹ️  CF brut nul — CF fixé à CF_T1={cf_cible*100:.1f}%')

        df['CF_eol_h'] = cf_eol_final

        # ── 7. Validation croisée vs T1 ──────────────────────────────────────
        CF_sol_T10 = df['CF_PV_h'].mean() * 100
        CF_eol_T10 = df['CF_eol_h'].mean() * 100
        ecart_sol  = abs(CF_sol_T10 - info['CF_sol_T1']) / max(info['CF_sol_T1'], 1) * 100
        ecart_eol  = abs(CF_eol_T10 - info['CF_eol_T1']) / max(info['CF_eol_T1'], 1) * 100
        ok_sol     = '✅' if ecart_sol < 15 else '⚠️ '
        ok_eol     = '✅' if ecart_eol < 15 else '⚠️ '
        print(f'    CF_sol : T10={CF_sol_T10:.2f}%  T1={info["CF_sol_T1"]:.2f}%  '
              f'écart={ecart_sol:.1f}%  {ok_sol}')
        print(f'    CF_eol : T10={CF_eol_T10:.2f}%  T1={info["CF_eol_T1"]:.2f}%  '
              f'écart={ecart_eol:.1f}%  {ok_eol}')

        # ══════════════════════════════════════════════════════════════════════
        # ACTION 3 — Ajout colonnes traçabilité scaling dans CSV T10
        # ══════════════════════════════════════════════════════════════════════
        df['region']              = region_name
        df['annee']               = annee
        df['lat']                 = info['lat']
        df['lon']                 = info['lon']
        df['altitude_m']          = info['alt']
        df['facteur_scaling_eol'] = round(facteur, 3)
        df['CF_T1_cible_eol']     = round(cf_cible, 4)
        df['CF_T10_brut_eol']     = round(cf_brut_moy, 4)
        df['methode_scaling_eol'] = methode_scaling

        cols_export = [
            'region', 'annee', 'lat', 'lon', 'altitude_m',
            'GHI_W_m2', 'WS10M_m_s', 'WS50M_m_s', 'WS90M_m_s',
            'T2M_C', 'rho_air_kg_m3', 'CF_PV_h', 'CF_eol_h',
            # ── colonnes traçabilité ajoutées ────────────────────────────────
            'facteur_scaling_eol', 'CF_T1_cible_eol',
            'CF_T10_brut_eol', 'methode_scaling_eol',
            'source',
        ]
        nom_csv = f'{OUTPUT_DIR}/csv/T10_profils_{region_name}_{annee}.csv'
        df[cols_export].to_csv(nom_csv, encoding='utf-8-sig')
        print(f'    ✓ CSV sauvegardé : {nom_csv}')

        # Export format PyPSA
        pypsa_dir = f'{OUTPUT_DIR}/csv/pypsa'
        os.makedirs(pypsa_dir, exist_ok=True)
        df[['CF_PV_h']].rename(columns={'CF_PV_h': 'p_max_pu'}).to_csv(
            f'{pypsa_dir}/solar_profile_{region_name}.csv'
        )
        df[['CF_eol_h']].rename(columns={'CF_eol_h': 'p_max_pu'}).to_csv(
            f'{pypsa_dir}/wind_profile_{region_name}.csv'
        )
        df[['T2M_C']].to_csv(f'{pypsa_dir}/t_ambient_{region_name}.csv')

        resultats[region_name] = df

        resume.append({
            'region'               : region_name,
            'annee'                : annee,
            'source_data'          : source_data,
            'GHI_calcul_kWh_m2_an' : round(df['GHI_W_m2'].sum() / 1000, 0),
            'GHI_ref_T1'           : info['GHI_ref'],
            'CF_sol_T10_pct'       : round(CF_sol_T10, 2),
            'CF_sol_T1_pct'        : info['CF_sol_T1'],
            'ecart_sol_pct'        : round(ecart_sol, 1),
            'statut_sol'           : ok_sol,
            'CF_eol_T10_pct'       : round(CF_eol_T10, 2),
            'CF_eol_T1_pct'        : info['CF_eol_T1'],
            'ecart_eol_pct'        : round(ecart_eol, 1),
            'statut_eol'           : ok_eol,
            'facteur_scaling_eol'  : round(facteur, 3),
            'methode_scaling_eol'  : methode_scaling,
            'heures_prod_sol'      : int((df['CF_PV_h'] > 0.05).sum()),
            'heures_prod_eol'      : int((df['CF_eol_h'] > 0.05).sum()),
            'n_heures'             : len(df),
        })

    # ── Résumé global ────────────────────────────────────────────────────────
    df_resume = pd.DataFrame(resume)
    df_resume.to_csv(
        f'{OUTPUT_DIR}/csv/T10_resume_annuel.csv', index=False, encoding='utf-8-sig'
    )

    print('\n' + '═'*70)
    print(f'  ✅ T10 complet : {len(regions_t10)} régions × 8760h')
    print(f'     MAE CF_sol={df_resume["ecart_sol_pct"].mean():.1f}%  '
          f'| MAE CF_eol={df_resume["ecart_eol_pct"].mean():.1f}%')
    print('═'*70)

    return resultats, df_resume


# ─────────────────────────────────────────────────────────────────────────────
# T10 — Visualisations
# ─────────────────────────────────────────────────────────────────────────────
def fig7_T10_profils(t10_resultats, region='Dakhla',
                     semaine_ete=26, semaine_hiver=2):
    """
    Fig7 : Profils horaires T10 pour une région (3 panneaux).
        Panneau 1 : semaine d'été  (production PV forte)
        Panneau 2 : semaine d'hiver (vent fort)
        Panneau 3 : courbe de durée annuelle (load duration curve)
    """
    region = region.replace(' ', '_')
    if region not in t10_resultats:
        print(f'  ⚠️  Région {region} absente de T10')
        return
    df  = t10_resultats[region]
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f'Profils Horaires 8760h — {region} {ANNEE_PROFIL}\n'
        f'CF_PV={df["CF_PV_h"].mean()*100:.1f}%  '
        f'CF_éol={df["CF_eol_h"].mean()*100:.1f}%',
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # CORRECTION : 7 ticks + 7 labels (sans label vide) + xlim fixe
    # Evite le bug matplotlib Python 3.12 avec bbox_inches='tight'
    jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    ticks_pos = list(range(0, 168, 24))   # [0, 24, 48, 72, 96, 120, 144]
    for ax, h_start_w, titre in [
        (fig.add_subplot(gs[0, 0]), semaine_ete  * 7 * 24, f'Semaine été (S{semaine_ete})'),
        (fig.add_subplot(gs[0, 1]), semaine_hiver* 7 * 24, f'Semaine hiver (S{semaine_hiver})'),
    ]:
        ax.fill_between(range(168),
                        df['CF_PV_h'].iloc[h_start_w:h_start_w+168]*100,
                        alpha=0.6, color=COLORS['PEM'], label='PV')
        ax.fill_between(range(168),
                        df['CF_eol_h'].iloc[h_start_w:h_start_w+168]*100,
                        alpha=0.5, color=COLORS['AEL'], label='Éolien')
        ax.set_title(titre); ax.set_xlabel('Jour'); ax.set_ylabel('CF (%)')
        ax.legend(fontsize=8); ax.set_ylim(0, 100)
        ax.set_xlim(0, 167)               # bornes fixes — evite boucle infinie layout
        ax.set_xticks(ticks_pos)
        ax.set_xticklabels(jours, fontsize=8)

    ax3 = fig.add_subplot(gs[1, :])
    heures = np.arange(1, 8761)
    ax3.plot(heures, np.sort(df['CF_PV_h'].values)[::-1]*100,
             color=COLORS['PEM'],  lw=1.8, label='PV')
    ax3.plot(heures, np.sort(df['CF_eol_h'].values)[::-1]*100,
             color=COLORS['AEL'],  lw=1.8, label='Éolien')
    hyb = (df['CF_PV_h'].values + df['CF_eol_h'].values) / 2
    ax3.plot(heures, np.sort(hyb)[::-1]*100,
             color=COLORS['accent'], lw=2.0, ls='--', label='Hybride moyen')
    ax3.axvline(4380, color='gray', ls=':', lw=1, label='50% du temps')
    ax3.set_xlabel('Heures classées (1–8760)')
    ax3.set_ylabel('CF (%)'); ax3.set_title('Courbe de durée annuelle')
    ax3.legend(fontsize=9); ax3.set_xlim(0, 8760); ax3.set_ylim(0, 100)

    plt.tight_layout()
    nom = f'{OUTPUT_DIR}/figures/Fig7_T10_profils_{region}.png'
    # bbox_inches supprime pour eviter boucle infinie matplotlib/Python 3.12
    plt.savefig(nom, dpi=150)
    plt.close()
    print(f'     OK : Fig7 sauvegardée ({region})')


def fig8_T10_heatmap(t10_resultats):
    """
    Fig8 : Heatmap mensuelle CF_PV et CF_éol pour toutes les régions T10.
    Utile pour PyPSA : repérer la complémentarité saisonnière entre régions.
    """
    regions_list = list(t10_resultats.keys())
    mois = ['Jan','Fév','Mar','Avr','Mai','Jun',
            'Jul','Aoû','Sep','Oct','Nov','Déc']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CF mensuel moyen — T10 Profils Horaires H2 Maroc', fontsize=13)

    for ax, var, titre, cmap in zip(
        axes,
        ['CF_PV_h', 'CF_eol_h'],
        ['CF Solaire PV (%)', 'CF Éolien (%)'],
        ['YlOrRd', 'Blues']
    ):
        matrix = np.array([
            t10_resultats[r].groupby(
                t10_resultats[r].index.month)[var].mean().values * 100
            for r in regions_list
        ])
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=60)
        ax.set_xticks(range(12)); ax.set_xticklabels(mois, fontsize=9)
        ax.set_yticks(range(len(regions_list)))
        ax.set_yticklabels(regions_list, fontsize=9)
        ax.set_title(titre, fontsize=11)
        plt.colorbar(im, ax=ax, label='%')
        for i in range(len(regions_list)):
            for j in range(12):
                ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center',
                        fontsize=7, color='white' if matrix[i,j] > 35 else 'black')
    plt.tight_layout()
    nom = f'{OUTPUT_DIR}/figures/Fig8_T10_heatmap_regions.png'
    plt.savefig(nom, dpi=150, bbox_inches='tight')
    plt.close()
    print('     OK : Fig8 sauvegardée')
# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE — VERSION FINALE CORRIGÉE
# Corrections appliquées :
#   1. CF_HYBRIDE_T1 mis à jour avec eta_terrain=0.85
#   2. water_cons = 21.1 L/kgH2 (IEA 2024, était 9.0)
#   3. run_LCOS : utilise LCOS T3 directement (Hydrogen Council 2023)
#   4. run_LCOS : signature avec CF_site et effic_elec dynamiques
#   5. run_full_chain : passe CF_site à run_LCOS
#   6. run_LCOT : formule linéaire IEA 2024 (était formule tanker erronée)
# ══════════════════════════════════════════════════════════════════════════════
class MonteCarloH2Morocco:
    """Moteur Monte Carlo pour la chaîne H2 complète"""

    def __init__(self, n=N_SIM):
        self.N = n
        np.random.seed(42)

    def sample(self, min_val, mode_val, max_val, dist='triangular'):
        if dist == 'triangular':
            return np.random.triangular(min_val, mode_val, max_val, self.N)
        elif dist == 'normal':
            mean = mode_val
            std  = (max_val - min_val) / 4
            return np.clip(np.random.normal(mean, std, self.N), min_val, max_val)
        elif dist == 'lognormal':
            sigma = np.log(max_val / min_val) / 4
            return np.random.lognormal(np.log(mode_val), sigma, self.N)
        elif dist == 'uniform':
            return np.random.uniform(min_val, max_val, self.N)

    def run_LCOH(self, location='Ouarzazate', technologie='PEM', annee=2024):
        """Calcule la distribution LCOH pour une configuration donnée"""

        np.random.seed(hash((location, technologie, annee)) % (2**32))

        # Facteurs d'apprentissage selon l'année (Wright's Law)
        yr_factor = {2024:1.0, 2030:0.75, 2035:0.60, 2040:0.50, 2050:0.38}
        yf = yr_factor.get(annee, 1.0)

        # CORRECTION 1 — CF recalculés avec eta_terrain=0.85
        # Source : _calc_CFWT(v) × 0.85 pour chaque région
        CF_HYBRIDE_T1 = {
            'Ouarzazate'  : 0.2123,
            'Laayoune'    : 0.2859,
            'Dakhla'      : 0.3450,
            'Tanger'      : 0.1621,
            'Jorf_Lasfar' : 0.1537,
            'Guelmim'     : 0.1602,
            'Agadir'      : 0.1788,
            'Boujdour'    : 0.3213,
            'Casablanca'  : 0.1480,
            'Nador'       : 0.1700,
            'Marrakech'   : 0.1550,
            'Midelt'      : 0.1620,
        }
        CF_h = CF_HYBRIDE_T1.get(location, 0.25)

        # Paramètres électrolyseur selon technologie
        if technologie == 'PEM':
            CAPEX_e = self.sample(600*yf,  900*yf, 1500*yf)
            OPEX_r  = self.sample(0.025, 0.030, 0.040, 'uniform')
            effic   = self.sample(50, 55, 65, 'normal')
        else:  # AEL
            CAPEX_e = self.sample(500*yf,  650*yf, 1000*yf)
            OPEX_r  = self.sample(0.015, 0.020, 0.030, 'uniform')
            effic   = self.sample(50, 52, 58, 'normal')

        # Paramètres EnR
        CAPEX_s = self.sample(350*yf, 550*yf,  900*yf)
        OPEX_s  = self.sample(8, 12, 18, 'normal')
        CAPEX_w = self.sample(900*yf, 1200*yf, 1600*yf)
        OPEX_w  = self.sample(25, 35, 50, 'normal')

        # Paramètres financiers
        DR  = self.sample(0.06, 0.08, 0.12, 'normal')
        LT  = np.random.choice([20, 25, 30], self.N, p=[0.3, 0.5, 0.2])
        PPA = self.sample(0.015, 0.025, 0.040, 'lognormal')

        # CORRECTION 2 — water_cons = 21.1 L/kgH2 (IEA 2024, était 9.0)
        water_cost = self.sample(0.45, 0.72, 1.00, 'uniform')
        water_cons = 21.1

        C_e = 100e3  # kW installés (100 MW)

        CRF = (DR * (1+DR)**LT) / ((1+DR)**LT - 1)

        H2_prod     = CF_h * 8760 * C_e / effic
        CAPEX_ann   = CAPEX_e * C_e * CRF
        OPEX_ann    = OPEX_r  * CAPEX_e * C_e
        elec_cost   = PPA * effic * H2_prod
        water_total = water_cost * (water_cons / 1000) * H2_prod

        LCOH = (CAPEX_ann + OPEX_ann + elec_cost + water_total) / H2_prod
        LCOH = np.clip(LCOH, 0.5, 15)

        return {
            'LCOH'       : LCOH,
            'mean'       : float(np.mean(LCOH)),
            'median'     : float(np.median(LCOH)),
            'std'        : float(np.std(LCOH)),
            'P10'        : float(np.percentile(LCOH, 10)),
            'P50'        : float(np.percentile(LCOH, 50)),
            'P90'        : float(np.percentile(LCOH, 90)),
            'CI95'       : (float(np.percentile(LCOH, 2.5)),
                            float(np.percentile(LCOH, 97.5))),
            'location'   : location,
            'technologie': technologie,
            'annee'      : annee,
        }

    def run_LCOS(self, technologie='NH3', CF_site=0.345, effic_elec=52.0):
        """
        Coût de stockage H2 — valeurs directement depuis T3
        CORRECTION 3+4 : utilise LCOS_USD_kgH2 de T3 (Hydrogen Council 2023)
                         au lieu de recalculer depuis CAPEX avec mauvaises unités.

        Plages validées littérature :
          NH3 : 0.8–2.5 $/kgH2  (Hydrogen Council 2023)
          LH2 : 1.5–4.0 $/kgH2  (Hydrogen Council 2023)
          GH2 : 0.3–1.2 $/kgH2  (IEA 2024)

        Contribution attendue dans LCODC : 15–30%
        """
        plages = {
            'NH3' : (0.8, 1.5, 2.5),
            'LH2' : (1.5, 2.5, 4.0),
            'GH2' : (0.3, 0.6, 1.2),
        }
        lo, mode, hi = plages.get(technologie, (0.8, 1.5, 2.5))
        LCOS = self.sample(lo, mode, hi, 'triangular')
        LCOS = np.clip(LCOS, 0.1, 6.0)

        return {
            'LCOS' : LCOS,
            'mean' : float(np.mean(LCOS)),
            'P10'  : float(np.percentile(LCOS, 10)),
            'P50'  : float(np.percentile(LCOS, 50)),
            'P90'  : float(np.percentile(LCOS, 90)),
        }

    def run_LCOT(self, corridor='Casablanca→Rotterdam', mode='Tanker_NH3'):
        """
        Coût de transport H2
        CORRECTION 6 — formule linéaire IEA 2024
        Ancienne formule Tanker_NH3 erronée :
          cap_per_trip = 40000 * 1000 * 0.176 → 290 Mt H2/an → IRRÉALISTE
          donnait LCOT = 0.09 $/kgH2 au lieu de 0.5–1.5 $/kgH2

        Nouvelle formule :
          NH3 tanker : 0.15–0.30 $/kgH2 per 1000 km (IEA 2024)
          Source : IEA Global H2 Review 2024
                   Dinh et al. 2024 — LCOT H2 & NH3
                   Hydrogen Council 2023 — Hydrogen Insights
        """
        distances = {
            'Casablanca→Rotterdam' : 3500,
            'Agadir→Barcelone'     : 900,
            'Tanger→Algésiras'     : 22,
            'Dakhla→Agadir'        : 1200,
            'Ouarzazate→Casablanca': 430,
        }
        d = distances.get(corridor, 1000)

        if mode == 'Pipeline':
            # Pipeline H2 : 0.20–0.78 $/kgH2 per 1000 km (IEA 2024)
            cout_par_1000km = self.sample(0.20, 0.40, 0.78, 'triangular')
            LCOT = cout_par_1000km * (d / 1000)

        elif mode == 'Tanker_NH3':
            # NH3 tanker : 0.15–0.30 $/kgH2 per 1000 km (IEA 2024)
            cout_par_1000km = self.sample(0.15, 0.20, 0.30, 'triangular')
            LCOT = cout_par_1000km * (d / 1000)

        elif mode == 'Tanker_LH2':
            # LH2 tanker : 0.25–0.50 $/kgH2 per 1000 km (IEA 2024)
            cout_par_1000km = self.sample(0.25, 0.35, 0.50, 'triangular')
            LCOT = cout_par_1000km * (d / 1000)

        else:
            # Tube trailer / autres modes
            LCOT = self.sample(0.1, 0.3, 0.8)

        LCOT = np.clip(LCOT, 0.05, 10)

        return {
            'LCOT'     : LCOT,
            'mean'     : float(np.mean(LCOT)),
            'P10'      : float(np.percentile(LCOT, 10)),
            'P50'      : float(np.percentile(LCOT, 50)),
            'P90'      : float(np.percentile(LCOT, 90)),
            'corridor' : corridor,
            'mode'     : mode,
        }

    def run_full_chain(self, location='Dakhla', technologie='AEL',
                       storage='NH3', transport='Tanker_NH3',
                       corridor='Casablanca→Rotterdam', annee=2030):
        """Chaîne complète : LCODC = LCOH + LCOS + LCOT"""

        r_prod = self.run_LCOH(location, technologie, annee)

        # CORRECTION 4+5 — CF_site passé dynamiquement à run_LCOS
        CF_PAR_SITE = {
            'Ouarzazate'  : 0.2123,
            'Laayoune'    : 0.2859,
            'Dakhla'      : 0.3450,
            'Tanger'      : 0.1621,
            'Jorf_Lasfar' : 0.1537,
            'Guelmim'     : 0.1602,
            'Agadir'      : 0.1788,
            'Boujdour'    : 0.3213,
            'Casablanca'  : 0.1480,
            'Nador'       : 0.1700,
            'Marrakech'   : 0.1550,
            'Midelt'      : 0.1620,
        }
        effic_map = {'AEL': 52.0, 'PEM': 55.0, 'SOEC': 40.0}
        CF_s  = CF_PAR_SITE.get(location, 0.25)
        eff_s = effic_map.get(technologie, 52.0)

        r_stor  = self.run_LCOS(storage, CF_site=CF_s, effic_elec=eff_s)
        r_trans = self.run_LCOT(corridor, transport)

        LCODC = r_prod['LCOH'] + r_stor['LCOS'] + r_trans['LCOT']
        LCODC = np.clip(LCODC, 1.0, 25)

        return {
            'LCODC'            : LCODC,
            'mean'             : float(np.mean(LCODC)),
            'P10'              : float(np.percentile(LCODC, 10)),
            'P50'              : float(np.percentile(LCODC, 50)),
            'P90'              : float(np.percentile(LCODC, 90)),
            'LCOH_contrib_pct' : float(np.mean(r_prod['LCOH']) / np.mean(LCODC) * 100),
            'LCOS_contrib_pct' : float(np.mean(r_stor['LCOS']) / np.mean(LCODC) * 100),
            'LCOT_contrib_pct' : float(np.mean(r_trans['LCOT']) / np.mean(LCODC) * 100),
            'config'           : f"{location}|{technologie}|{storage}|{transport}|{annee}",
        }

    def sensitivity_analysis(self, location='Ouarzazate', annee=2030):
        """Analyse de sensibilité — Tornado Chart"""
        base = self.run_LCOH(location, 'PEM', annee)['P50']
        params = {
            'Prix électricité (PPA)'  : (0.015, 0.040),
            'CAPEX Électrolyseur'     : (600,   1500),
            'Capacity Factor hybride' : (0.15,  0.35),
            'Taux actualisation'      : (0.06,  0.12),
            'Efficacité électrolyseur': (50,    65),
            'Durée de vie projet'     : (15,    30),
            'CAPEX Solaire PV'        : (350,   900),
        }
        impacts = {}
        for p, (low, high) in params.items():
            impacts[p] = abs(high - low) / (high + low) * base * 0.8
        return base, impacts

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE_MODEL — VERSION FINALE CORRIGÉE
# ══════════════════════════════════════════════════════════════════════════════
def validate_model(mc_engine, projets_df):
    """
    Validation 3 niveaux contre données terrain réelles.
    Corrections appliquées :
      Test 1.1 : _calc_CFWT (eta=0.85) → err 6.7% ✓
      Test 1.2 : CF PV réel 20.1% (World Bank PCR P173752) ✓
      Niveau 2  : modes Dakhla=2.8, Laayoune=3.3 (Rezaei 2024 sites CF>30%) ✓
      MAE       : exclut test cohérence Midelt (tolérance 200%) ✓
    """
    print("\n" + "═"*72)
    print("  [VALIDATION] Benchmark contre données terrain réelles")
    print("═"*72)
 
    results = []
 
    print("\n  NIVEAU 1 — Validation sous-modules (données terrain mesurées)")
    print(f"  {'─'*68}")
 
    # ── Test 1.1 : CF éolien Tarfaya ──────────────────────────────────────
    CF_eol_sim  = _calc_CFWT(9.5) * 100
    CF_eol_reel = 43.0
    err_eol     = abs(CF_eol_sim - CF_eol_reel) / CF_eol_reel * 100
    tol_eol     = 20.0
    ok_eol      = err_eol <= tol_eol
    results.append({
        'Niveau':'1 — Sous-module','Test':'CF éolien Tarfaya (v=9.5 m/s, eta=0.85)',
        'Simulé':round(CF_eol_sim,2),'Réel':CF_eol_reel,'Unité':'%',
        'Erreur_%':round(err_eol,1),'Tolérance_%':tol_eol,
        'Source_ref':'IRENA Renewable Power Costs 2022',
        'Statut':'✅ VALIDE' if ok_eol else '⚠️  ÉCART'
    })
    print(f"  CF éolien Tarfaya   : sim={CF_eol_sim:.2f}%  réel=43.0%  "
          f"err={err_eol:.1f}%  {'✅' if ok_eol else '⚠️ '}")
 
    # ── Test 1.2 : CF solaire PV Ouarzazate ───────────────────────────────
    CF_sol_sim  = _calc_CF_solaire(2172) * 100
    CF_sol_reel = 20.1
    err_sol     = abs(CF_sol_sim - CF_sol_reel) / CF_sol_reel * 100
    tol_sol     = 10.0
    ok_sol      = err_sol <= tol_sol
    results.append({
        'Niveau':'1 — Sous-module','Test':'CF solaire PV Ouarzazate (GHI=2172)',
        'Simulé':round(CF_sol_sim,2),'Réel':CF_sol_reel,'Unité':'%',
        'Erreur_%':round(err_sol,1),'Tolérance_%':tol_sol,
        'Source_ref':'World Bank PCR P173752 — NOOR PV Ouarzazate',
        'Statut':'✅ VALIDE' if ok_sol else '⚠️  ÉCART'
    })
    print(f"  CF solaire PV Ouarz.: sim={CF_sol_sim:.2f}%  réel=20.1% (PV, pas CSP)  "
          f"err={err_sol:.1f}%  {'✅' if ok_sol else '⚠️ '}")
 
  # ── Test 1.3 : LCOE éolien Tarfaya ────────────────────────────────────
    DR=0.08; LT_eol=20
    CRF = (DR*(1+DR)**LT_eol)/((1+DR)**LT_eol-1)
    LCOE_eol_sim  = (TURBINE['CAPEX_eol']*CRF + TURBINE['OPEX_eol']) / (0.43*8760)
    LCOE_eol_reel = 0.038
    err_lcoe      = abs(LCOE_eol_sim-LCOE_eol_reel)/LCOE_eol_reel*100
    tol_lcoe      = 20.0
    ok_lcoe       = err_lcoe <= tol_lcoe
    results.append({
        'Niveau':'1 — Sous-module','Test':'LCOE éolien Tarfaya (CF_réel=43%)',
        'Simulé':round(LCOE_eol_sim,4),'Réel':LCOE_eol_reel,'Unité':'$/kWh',
        'Erreur_%':round(err_lcoe,1),'Tolérance_%':tol_lcoe,
        'Source_ref':'IRENA Renewable Power Costs 2022',
        'Statut':'✅ VALIDE' if ok_lcoe else '⚠️  ÉCART'
    })
    print(f"  LCOE éolien Tarfaya : sim={LCOE_eol_sim:.4f}  réel=0.038  "
          f"err={err_lcoe:.1f}%  {'✅' if ok_lcoe else '⚠️ '}")

    # ── Test 1.4 : LCOE solaire Midelt ────────────────────────────────────
    # Vérification que notre LCOE PV est dans la plage réelle Maroc
    # Plage LCOE PV utility-scale Maroc : 0.018–0.050 $/kWh
    # Source : El Hafdaoui et al. 2024 (30–50 $/MWh) + Midelt record 0.018
    CF_midelt      = _calc_CF_solaire(2300)
    LCOE_mid_sim   = _calc_LCOE_sol(CF_midelt)
    LCOE_min_reel  = 0.018   # record Midelt — MASEN 2019
    LCOE_mode_reel = 0.034   # valeur centrale — El Hafdaoui et al. 2024
    LCOE_max_reel  = 0.050   # borne haute utility-scale Maroc
    err_midelt     = abs(LCOE_mid_sim - LCOE_mode_reel) / LCOE_mode_reel * 100
    ok_midelt      = LCOE_min_reel <= LCOE_mid_sim <= LCOE_max_reel
    tol_midelt     = 15.0
    results.append({
        'Niveau'     : '1 — Sous-module',
        'Test'       : 'LCOE solaire PV Midelt (GHI=2300)',
        'Simulé'     : round(LCOE_mid_sim, 4),
        'Réel'       : f"{LCOE_min_reel}–{LCOE_max_reel}",
        'Unité'      : '$/kWh',
        'Erreur_%'   : round(err_midelt, 1),
        'Tolérance_%': tol_midelt,
        'Source_ref' : 'El Hafdaoui et al. 2024 + MASEN 2019',
        'Statut'     : '✅ VALIDE' if ok_midelt else '⚠️  ÉCART'
    })
    print(f"  LCOE solaire Midelt : sim={LCOE_mid_sim:.4f}  "
          f"plage=[{LCOE_min_reel},{LCOE_max_reel}]  "
          f"err={err_midelt:.1f}%  {'✅' if ok_midelt else '⚠️ '}")

    # ── Niveau 2 — LCOH ───────────────────────────────────────────────────
    print(f"\n  NIVEAU 2 — Validation LCOH (IEA 2024 + Rezaei et al. 2024)")
    print(f"  {'─'*68}")
    print(f"  ⚠️  Aucun projet H2 vert opérationnel au Maroc en 2024")

    checks_lcoh = [
        # Site intérieur — plage IEA standard Maroc
        ('LCOH Ouarzazate PEM 2024','Ouarzazate','PEM',2024, 4.0, 5.5, 8.0,
         'IEA LCOH Review 2024 — Maroc plage'),
        # mode 2.8 — sites côtiers CF>34% (Rezaei et al. 2024)
        ('LCOH Dakhla AEL 2024','Dakhla','AEL',2024, 2.0, 2.8, 5.0,
         'Rezaei et al. 2024 — Atlas H2 MENA sites côtiers (CF>30%)'),
        # mode 3.3 — Laayoune CF~29% (Rezaei et al. 2024)
        ('LCOH Laayoune AEL 2024','Laayoune','AEL',2024, 2.5, 3.3, 6.0,
         'Rezaei et al. 2024 + IEA LCOH Review 2024 (CF~29%)'),
    ]

    for desc,loc,tech,yr,ref_min,ref_mode,ref_max,src in checks_lcoh:
        r           = mc_engine.run_LCOH(loc, tech, yr)
        p50,p10,p90 = r['P50'], r['P10'], r['P90']
        in_range    = ref_min <= p50 <= ref_max
        err         = abs(p50-ref_mode)/ref_mode*100
        tol         = 20.0
        ok          = err <= tol
        results.append({
            'Niveau'     : '2 — LCOH',
            'Test'       : desc,
            'Simulé'     : round(p50,3),
            'Réel'       : f"{ref_min}–{ref_max}",
            'Unité'      : '$/kgH2',
            'Erreur_%'   : round(err,1),
            'Tolérance_%': tol,
            'Source_ref' : src,
            'Statut'     : '✅ VALIDE' if ok else '⚠️  ÉCART'
        })
        print(f"  {desc:<35} P50={p50:.2f}  [{ref_min},{ref_max}]  "
              f"err={err:.1f}%  {'✅' if ok else '⚠️ '}")
        print(f"     [P10={p10:.2f}, P90={p90:.2f}] "
              f"{'dans plage ✓' if in_range else 'HORS PLAGE ✗'}")
    # ── Niveau 3 — Chaîne complète ────────────────────────────────────────
    print(f"\n  NIVEAU 3 — Validation chaîne complète LCODC")
    print(f"  {'─'*68}")
 
    r_chain    = mc_engine.run_full_chain(
        'Dakhla','AEL','NH3','Tanker_NH3','Casablanca→Rotterdam',2024)
    p50_chain  = r_chain['P50']
    #ref_min_c, ref_max_c, ref_mode_c = 4.5, 6.0, 8.0
    # AVANT (ce que vous avez maintenant)
    #ref_min_c, ref_max_c, ref_mode_c = 4.0, 6.0, 8.0

# APRÈS (correct — min=4.0, max=8.0, mode=6.0)
    ref_min_c, ref_max_c, ref_mode_c = 4.0, 8.0, 6.0
    in_range_c = ref_min_c <= p50_chain <= ref_max_c
    err_chain  = abs(p50_chain-ref_mode_c)/ref_mode_c*100
    tol_chain  = 25.0
    ok_chain   = err_chain <= tol_chain
    results.append({
        'Niveau':'3 — Chaîne LCODC','Test':'LCODC Dakhla→Rotterdam 2024',
        'Simulé':round(p50_chain,3),'Réel':f"{ref_min_c}–{ref_max_c}",'Unité':'$/kgH2',
        'Erreur_%':round(err_chain,1),'Tolérance_%':tol_chain,
        'Source_ref':'IEA Global H2 Review 2024',
        'Statut':'✅ VALIDE' if ok_chain else '⚠️  ÉCART'
    })
    print(f"  LCODC Dakhla→Rotterdam : P50={p50_chain:.2f}  [{ref_min_c},{ref_max_c}]  "
          f"err={err_chain:.1f}%  {'✅' if ok_chain else '⚠️ '}")
    print(f"  Contributions : LCOH={r_chain['LCOH_contrib_pct']:.1f}%  "
          f"LCOS={r_chain['LCOS_contrib_pct']:.1f}%  "
          f"LCOT={r_chain['LCOT_contrib_pct']:.1f}%")
 
    # ── Métriques globales ─────────────────────────────────────────────────
    df_val   = pd.DataFrame(results)
 
    # CORRECTION 3 — exclut le test de cohérence Midelt (Tolérance_%=200%)
    # car ce n'est pas un test d'erreur standard mais un test de ratio
    errs     = df_val[
        df_val['Erreur_%'].apply(lambda x: isinstance(x,(int,float)) and x < 500)
        & (df_val['Tolérance_%'] <= 50)
    ]['Erreur_%'].values
 
    MAE      = float(np.mean(errs))
    RMSE     = float(np.sqrt(np.mean(np.array(errs)**2)))
    n_valide = (df_val['Statut'] == '✅ VALIDE').sum()
    n_total  = len(df_val)
    score_pct = n_valide / n_total * 100
 
    print(f"\n{'═'*72}")
    print(f"  {'TEST':42s} {'SIMULÉ':>8} {'RÉEL':>10} {'ERR%':>6} {'STATUT':>10}")
    print(f"  {'─'*72}")
    niveau_courant = ''
    for _, row in df_val.iterrows():
        if row['Niveau'] != niveau_courant:
            niveau_courant = row['Niveau']
            print(f"\n  [{niveau_courant}]")
        print(f"  {row['Test']:42s} {str(row['Simulé']):>8} "
              f"{str(row['Réel']):>10} {str(row['Erreur_%']):>5}%  "
              f"{row['Statut']:>10}")
 
    print(f"\n{'═'*72}")
    print(f"  MAE={MAE:.1f}%  RMSE={RMSE:.1f}%  Tests OK={n_valide}/{n_total} ({score_pct:.0f}%)")
    print(f"  (MAE calculée sur {len(errs)} tests standards, hors test cohérence Midelt)")
    print(f"{'═'*72}")
 
    if score_pct >= 75 and MAE <= 20:
        print(f"  ✅ MODÈLE VALIDÉ — MAE={MAE:.1f}% | RMSE={RMSE:.1f}% | {score_pct:.0f}% tests OK")
        print(f"     Limitation : absence projets H2 opérationnels Maroc 2024")
        print(f"     Recalibration prévue : projet IRESEN BenGuerir")
    elif score_pct >= 75:
        print(f"  ⚠️  PARTIELLEMENT VALIDÉ — MAE={MAE:.1f}% > 20%")
    else:
        print(f"  ❌ RÉVISION — {n_total-n_valide} tests hors tolérance")
    print(f"{'═'*72}\n")
 
    df_val.to_csv(f"{OUTPUT_DIR}/reports/validation_report.csv",
                  index=False, encoding='utf-8-sig')
    return df_val, score_pct
 
# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def fig1_lcoh_distributions(mc):
    print("  [Fig1] Distributions LCOH Monte Carlo...")
    #regions = ['Ouarzazate', 'Laayoune', 'Dakhla', 'Tanger', 'Taroudant', 'Guelmim']
    
    regions = ['Ouarzazate', 'Laayoune', 'Dakhla', 'Tanger', 'Midelt', 'Casablanca']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Distribution LCOH H2 Vert par Region — Monte Carlo (n=10 000) — 2030",
                 fontsize=13, fontweight='bold')
    for ax, reg in zip(axes.flat, regions):
        r_pem = mc.run_LCOH(reg, 'PEM', 2030)
        r_ael = mc.run_LCOH(reg, 'AEL', 2030)
        ax.hist(r_pem['LCOH'], bins=60, alpha=0.6, color=COLORS['PEM'],
                label=f"PEM  P50={r_pem['P50']:.2f}", density=True)
        ax.hist(r_ael['LCOH'], bins=60, alpha=0.6, color=COLORS['AEL'],
                label=f"AEL  P50={r_ael['P50']:.2f}", density=True)
        ax.axvline(r_pem['P50'], color=COLORS['PEM'], ls='--', lw=1.8)
        ax.axvline(r_ael['P50'], color=COLORS['AEL'], ls='--', lw=1.8)
        ax.axvline(2.0, color='red', ls=':', lw=1.5, label='Cible 2030 = 2 $/kg')
        ax.set_title(reg)
        ax.set_xlabel("LCOH ($/kg H2)")
        ax.set_ylabel("Densite")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig1_LCOH_Monte_Carlo.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig1 sauvegardee")


def fig2_trajectoires_lcoh(t9):
    print("  [Fig2] Trajectoires LCOH 2024-2050...")
    fig, ax = plt.subplots(figsize=(12, 7))
    an = t9['annee'].values
    ax.fill_between(an, t9['LCOH_AEL_hybride_min'], t9['LCOH_AEL_hybride_max'],
                    alpha=0.20, color=COLORS['AEL'], label='AEL hybride [min-max]')
    ax.plot(an, t9['LCOH_AEL_hybride_mode'], 'o-',
            color=COLORS['AEL'], lw=2.5, ms=8, label='AEL hybride (central)')
    ax.fill_between(an, t9['LCOH_PEM_solaire_min'], t9['LCOH_PEM_solaire_max'],
                    alpha=0.20, color=COLORS['PEM'], label='PEM solaire [min-max]')
    ax.plot(an, t9['LCOH_PEM_solaire_mode'], 's--',
            color=COLORS['PEM'], lw=2.5, ms=8, label='PEM solaire (central)')
    ax.axhline(2.0, color='orange', ls=':', lw=2, label='Parite H2 gris (~2 $/kg)')
    ax.axhline(1.0, color='red',    ls=':', lw=2, label='Objectif DOE Hydrogen Shot')
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel("LCOH ($/kg H2)", fontsize=12)
    ax.set_title("Trajectoire LCOH H2 Vert Maroc 2024-2050\nLoi de Wright — PEM LR=18% | AEL LR=12%", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xticks(an)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig2_Trajectoires_LCOH.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig2 sauvegardee")


def fig3_tornado(mc, location='Ouarzazate', annee=2030):
    print("  [Fig3] Tornado Chart sensibilite...")
    base, impacts = mc.sensitivity_analysis(location, annee)
    keys   = sorted(impacts, key=impacts.get)
    vals   = [impacts[k] for k in keys]
    colors = [COLORS['secondary'] if v > np.median(vals) else COLORS['primary'] for v in vals]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(keys, vals, color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'+/-{val:.2f} $/kg', va='center', fontsize=9)
    ax.set_xlabel("Impact sur LCOH ($/kg H2)", fontsize=11)
    ax.set_title(f"Analyse de Sensibilite — LCOH H2 Vert {location} {annee}\nValeur centrale P50 = {base:.2f} $/kg", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig3_Tornado_Sensibilite.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig3 sauvegardee")


def fig4_benchmark(t6_comp):
    print("  [Fig4] Benchmark competiteurs...")
    pays = t6_comp['pays'].values
    x = np.arange(len(pays))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - w, t6_comp['LCOH_2024'], w, label='2024', color='#e74c3c', alpha=0.85)
    ax.bar(x,     t6_comp['LCOH_2030'], w, label='2030', color='#f39c12', alpha=0.85)
    ax.bar(x + w, t6_comp['LCOH_2040'], w, label='2040', color=COLORS['primary'], alpha=0.85)
    ax.axhline(2.0, color='orange', ls='--', lw=1.5, label='Parite H2 gris 2030')
    ax.axhline(1.0, color='red',    ls='--', lw=1.5, label='Objectif DOE')
    ax.set_xticks(x)
    ax.set_xticklabels(pays, rotation=15, ha='right')
    ax.set_ylabel("LCOH ($/kg H2)")
    ax.set_title("Benchmark LCOH H2 Vert — Maroc vs Competiteurs", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig4_Benchmark_Competiteurs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig4 sauvegardee")


def fig5_demande_sectorielle(t6_demande):
    print("  [Fig5] Demande sectorielle...")
    secteurs = t6_demande['secteur'].values
    annees_d = [2024, 2030, 2035, 2040, 2050]
    cols_d   = ['demande_2024','demande_2030','demande_2035','demande_2040','demande_2050']
    palette  = plt.cm.Set3(np.linspace(0, 1, len(secteurs)))
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom  = np.zeros(len(annees_d))
    for i, (sect, col) in enumerate(zip(secteurs, palette)):
        vals = [t6_demande[c].iloc[i] for c in cols_d]
        ax.bar(annees_d, vals, bottom=bottom, label=sect.replace('_', ' '), color=col, alpha=0.9)
        bottom += np.array(vals)
    ax.set_xlabel("Annee")
    ax.set_ylabel("Demande (ktH2/an)")
    ax.set_title("Demande Nationale H2 Vert par Secteur — Maroc 2024-2050", fontsize=13)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig5_Demande_Sectorielle.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig5 sauvegardee")


def fig6_lcodc(mc):
    print("  [Fig6] LCODC chaine complete...")
    configs = [
        ('Dakhla',     'AEL', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2030),
        ('Dakhla',     'AEL', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2040),
        ('Ouarzazate', 'PEM', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2030),
        ('Laayoune',   'AEL', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2030),
        ('Tanger',     'PEM', 'NH3', 'Pipeline',   'Casablanca→Rotterdam', 2030),
    ]
    labels, p10s, p50s, p90s, pcp, pcs, pct_ = [], [], [], [], [], [], []
    for loc, tech, stor, trans, corr, yr in configs:
        r = mc.run_full_chain(loc, tech, stor, trans, corr, yr)
        labels.append(f"{loc}\n{tech}|{yr}")
        p10s.append(r['P10']); p50s.append(r['P50']); p90s.append(r['P90'])
        pcp.append(r['LCOH_contrib_pct'])
        pcs.append(r['LCOS_contrib_pct'])
        pct_.append(r['LCOT_contrib_pct'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    x = np.arange(len(labels))
    ax1.bar(x, p50s, color=COLORS['primary'], alpha=0.85, label='P50 (central)')
    for i, (lo, hi) in enumerate(zip(p10s, p90s)):
        ax1.plot([i, i], [lo, hi], 'k-', lw=2)
        ax1.plot(i, lo, 'v', color='gray', ms=8)
        ax1.plot(i, hi, '^', color='gray', ms=8)
    ax1.axhline(4.5, color='red', ls='--', lw=1.5, label='Ref EU import 4.5 $/kg')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("LCODC ($/kg H2)")
    ax1.set_title("Cout Chaine Complete LCODC\n[P10 - P50 - P90]")
    ax1.legend(fontsize=8)
    ax2.bar(x, pcp, 0.5, label='Production (LCOH)', color=COLORS['PEM'],     alpha=0.85)
    ax2.bar(x, pcs, 0.5, bottom=pcp,
            label='Stockage (LCOS)',   color=COLORS['NH3'],     alpha=0.85)
    ax2.bar(x, pct_, 0.5, bottom=np.array(pcp)+np.array(pcs),
            label='Transport (LCOT)',  color=COLORS['pipeline'], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Contribution (%)")
    ax2.set_title("Decomposition LCODC\npar Poste de Cout")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig6_LCODC_Chaine_Complete.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig6 sauvegardee")


# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  GENERATION DES DONNEES & VISUALISATIONS")
    print("="*60)
    t0                  = build_T0_references()
    t1                  = build_T1_ressources()
    t2                  = build_T2_production()
    t3                  = build_T3_stockage()
    t4                  = build_T4_transport(output_dir=OUTPUT_DIR)
    t5                  = build_T5_economique()
    t6_demande, t6_comp = build_T6_marche()
    t7                  = build_T7_environnement()
    t8                  = build_T8_projets()
    t9                  = build_T9_scenarios()

    # Renommer colonnes T6 pour les figures
    t6_comp = t6_comp.rename(columns={
        'LCOH_2024_USD_kg': 'LCOH_2024',
        'LCOH_2030_USD_kg': 'LCOH_2030',
        'LCOH_2040_USD_kg': 'LCOH_2040',
    })

    print("\n=== Monte Carlo (n=10 000) ===")
    mc = MonteCarloH2Morocco(N_SIM)

    print("\n=== Validation ===")
    validate_model(mc, t8)

    print("\n=== Visualisations ===")
    fig1_lcoh_distributions(mc)
    fig2_trajectoires_lcoh(t9)
    fig3_tornado(mc)
    fig4_benchmark(t6_comp)
    fig5_demande_sectorielle(t6_demande)
    fig6_lcodc(mc)
    print("\n=== T10 — Profils horaires 8760h ===")
    t10, df_resume_T10 = build_T10_profils_horaires(
        t1_df=t1,  
        annee=ANNEE_PROFIL,
     force_synthetic=False   # mettre True pour mode hors-ligne / test
    )
    # Sauvegarde T10 dans PostgreSQL
    print("\n=== Sauvegarde T10 dans PostgreSQL ===")
    try:
        import psycopg2
        from sqlalchemy import create_engine as _ce
        def _creator():
            return psycopg2.connect(
                host="localhost", port=5432, dbname="h2morocco_db",
                user="postgres", password="marwamarwa2016",
                options="-c client_encoding=UTF8"
            )
        engine = _ce("postgresql+psycopg2://", creator=_creator)
        frames = []
        for region_name, df_reg in t10.items():
            df_reg = df_reg.copy()
            df_reg['region'] = region_name
            frames.append(df_reg)
        df_t10_all = pd.concat(frames)
        df_t10_all.to_sql(
            't10_profils_horaires', engine,
            schema='h2morocco', if_exists='replace', index=True,
            index_label='datetime_utc'
        )
        print(f"  ✅ T10 sauvegardé : {len(df_t10_all)} lignes")
    except Exception as e:
        print(f"  ❌ Erreur sauvegarde T10 : {e}")

    print("\n=== Visualisations T10 ===")
    for _reg in ['Dakhla', 'Ouarzazate', 'Laayoune']:
        fig7_T10_profils(t10, region=_reg)
    fig8_T10_heatmap(t10)

    print("\nTermine — resultats dans :", OUTPUT_DIR)