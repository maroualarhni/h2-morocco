# -*- coding: utf-8 -*-


# h2app/engine.py — Moteur de calcul (toutes les formules physiques et économiques)
"""
Sources intégrées: basededonnees.py, Etape1macbeth.py,
modele_stockage.py, Etape4transport.py, etape2.py
"""
import numpy as np, math
from scipy.special import gamma as gamma_func
from scipy.optimize import linprog
from scipy.stats import triang

# ═══════════════════════ CONSTANTES GLOBALES ═══════════════════════════════════
TAUX_EUR = 0.9217
LHV_H2 = 33.33; HHV_H2 = 39.41; R_IDEAL = 8.314; M_H2 = 2.016e-3
T_STD = 293.15; GAMMA_H2 = 1.41
ANNEES = [2024, 2030, 2035, 2040, 2050]

# Seuils CO₂ (Étape 1)
CO2_SEUIL_UE = 3.38; CO2_SEUIL_BEST = 1.0; CO2_REF_GRIS = 10.0
CI_ENR_MIN = 20.0; CI_GRID_MAX = 500.0; CI_IND_OVERH = 1.15

CO2_UPSTREAM = {
    'GH2_350bar':1.2,'GH2_700bar':1.4,'LH2':1.8,'NH3':2.1,
    'LOHC':1.6,'Caverne_saline':1.0,'e_methanol':2.3,'Caverne':1.0,'eMethanol':2.3,
}
ENERGY_SOURCE_MAP = {
    'GH2_350bar':{'elec':'enr','compr':'grid','synth':None},
    'GH2_700bar':{'elec':'enr','compr':'grid','synth':None},
    'LH2':       {'elec':'enr','compr':'grid','synth':None},
    'NH3':       {'elec':'enr','compr':'grid','synth':'enr'},
    'LOHC':      {'elec':'enr','compr':'grid','synth':'grid'},
    'Caverne':   {'elec':'enr','compr':'grid','synth':None},
    'e_methanol':{'elec':'enr','compr':'grid','synth':'enr'},
    'eMethanol': {'elec':'enr','compr':'grid','synth':'enr'},
}
# OPEX spécifiques (Étape 1)
NH3_CATAL_COST = 0.012; NH3_CATAL_LIFE = 3.0; NH3_YIELD = 5.6
LH2_BOILOFF = 0.20; LH2_DAYS = 14; LH2_PRICE = 6.0
LOHC_DEGRAD = 2.0; LOHC_OIL_COST = 3.5; LOHC_OIL_KG = 16.0

# Turbine éolienne
TURBINE = {'V_ci':3,'V_r':12,'V_o':25,'k':2,'PR':0.80,'DR':0.08,
    'Cs':550,'Os':12,'Ls':25,'Ce':1100,'Oe':35,'Le':20,'CF_min':0.15}

# ═══════════════════════ PHYSIQUE — PRODUCTION (base_de_donnees_corrige_F.py) ═══
def CRF(r,n):
    if r<1e-9: return 1.0/n
    return (r*(1+r)**n)/((1+r)**n-1)

def haversine(a1,o1,a2,o2):
    R=6371; a1,o1,a2,o2=map(math.radians,[a1,o1,a2,o2])
    d=a2-a1;e=o2-o1
    return round(2*R*math.asin(math.sqrt(math.sin(d/2)**2+math.cos(a1)*math.cos(a2)*math.sin(e/2)**2)),1)

def cf_solaire(ghi):
    return float(np.clip((ghi/8760)*TURBINE['PR'],0,0.35))

def cf_eolien(v):
    if v<1: return 0.0
    k=TURBINE['k']; c=v/gamma_func(1+1/k)
    a=np.exp(-(TURBINE['V_ci']/c)**k); b=np.exp(-(TURBINE['V_r']/c)**k)
    d=np.exp(-(TURBINE['V_o']/c)**k)
    den=(TURBINE['V_r']/c)**k-(TURBINE['V_ci']/c)**k
    return float(np.clip(((a-b)/den-d)*0.85,0,1)) if den else 0.0

#  ajout des courbes d'apprentissage pour PV et éolien
# ═══════════════════════ CAPEX LEARNING RATES — UNIFIÉ ════════════════════════
CAPEX_PARAMS = {
    'sol':  {'c24': 550,  'l1': 0.07, 'l2': 0.04, 'fl': 180},
    'eol':  {'c24': 1100, 'l1': 0.04, 'l2': 0.025,'fl': 700},
    'PEM':  {'c24': 900,  'l1': 0.08, 'l2': 0.04, 'fl': 180},
    'AEL':  {'c24': 650,  'l1': 0.06, 'l2': 0.03, 'fl': 200},
    'SOEC': {'c24': 3500, 'l1': 0.10, 'l2': 0.06, 'fl': 400},
    'bat':  {'c24': 150,  'l1': 0.05, 'l2': 0.03, 'fl': 60 },
}

def capex_lr_unified(key, annee):
    """CAPEX avec learning rate — remplace capex_lr() et capex_enr_lr()."""
    p = CAPEX_PARAMS[key]
    if annee <= 2024: return p['c24']
    v = p['c24'] * (1 - p['l1']) ** (min(annee, 2030) - 2024)
    if annee > 2030:
        v *= (1 - p['l2']) ** (annee - 2030)
    return max(v, p['fl'])

# Alias rétrocompatibles (pour ne pas casser app.py si il les appelle déjà)
def capex_enr_lr(type_enr, annee): return capex_lr_unified(type_enr, annee)
def capex_lr(c, l1, l2, f, y): return max(
    c * (1-l1)**(min(y,2030)-2024) if y>2024 else c, f) if y<=2030 else max(
    c * (1-l1)**6 * (1-l2)**(y-2030), f)

def lcoe_sol(cf, annee=2024):
    if cf <= 0: return None
    cx = capex_lr_unified('sol', annee)
    return round((cx * CRF(TURBINE['DR'], TURBINE['Ls']) + TURBINE['Os']) / (cf * 8760), 6)

def lcoe_eol(cf, annee=2024):
    if cf < TURBINE['CF_min']: return None
    cx = capex_lr_unified('eol', annee)
    return round((cx * CRF(TURBINE['DR'], TURBINE['Le']) + TURBINE['Oe']) / (cf * 8760), 6)

def calc_hybride(ce, cs, le, ls, w_min=0.15, w_max=0.70):
    """Optimisation mix PV/éolien — inchangée."""
    if cs + ce == 0: return 0, 0, ls or 0
    if le is None: return cs, 0.0, ls
    best_w, best_val = w_min, float('inf')
    for w in np.linspace(w_min, w_max, 71):
        ch = w * ce + (1 - w) * cs
        lh = w * le + (1 - w) * ls
        if ch <= 0: continue
        val = lh / ch
        if val < best_val:
            best_val = val
            best_w = w
    ch = best_w * ce + (1 - best_w) * cs
    lh = best_w * le + (1 - best_w) * ls
    return ch, best_w, lh

def calc_lcoh(lh, ch, cx, op, eta, dr, lt, tech=None, annee=None):
    """
    LCOH corrigé — si tech+annee fournis, recalcule cx dynamiquement.
    Rétrocompatible : fonctionne aussi avec cx fixe comme avant.
    """
    if lh is None or ch <= 0: return None
    if tech is not None and annee is not None:
        cx = capex_lr_unified(tech, annee)   # ← CAPEX dynamique
    h = ch * 8760
    return round(lh * eta + (cx * CRF(dr, lt)) / h * eta + (cx * op) / h * eta, 4)

# ═══════════════════════ CO₂ (Étape 1 — MACBETH) ═════════════════════════════
def get_ci_sources(cf_pct):
    """
    cf_pct : facteur de charge hybride en % (ex: 31.0)
    Retourne (ci_enr, ci_grid, ci_ind) en gCO2/kWh
    """
    ci_enr  = 20.0    # ENR : solaire/éolien
    ci_grid = 351.3   # Réseau Maroc (ONEE 2023)
    ci_ind  = 550.0   # Industriel (diesel/gaz)
    return ci_enr, ci_grid, ci_ind

def calc_co2_tech(tech, ci_enr, ci_grid, ci_ind, e_compr=2.0, e_synth=0):
    CI_ELECTROLYSEUR = 55.0   # kWh/kgH2 (consommation électrolyseur PEM)
    co2_elec  = CI_ELECTROLYSEUR * ci_enr / 1000
    co2_compr = e_compr * ci_enr / 1000
    co2_synth = e_synth * ci_enr / 1000
    return round(co2_elec + co2_compr + co2_synth, 3)

def score_co2_zone(co2):
    if co2 >= CO2_REF_GRIS: return 0
    if co2 > CO2_SEUIL_UE: return round((CO2_REF_GRIS-co2)/(CO2_REF_GRIS-CO2_SEUIL_UE)*40,2)
    if co2 > CO2_SEUIL_BEST: return round(40+(CO2_SEUIL_UE-co2)/(CO2_SEUIL_UE-CO2_SEUIL_BEST)*45,2)
    return round(85+min((CO2_SEUIL_BEST-co2)/CO2_SEUIL_BEST,1)*15,2)

def co2_zone_label(co2):
    if co2<=CO2_SEUIL_BEST: return 'Premium vert','🟢'
    if co2<=CO2_SEUIL_UE: return 'Certifiable RFNBO UE','🟡'
    if co2<CO2_REF_GRIS: return 'Hors certification','🟠'
    return 'Gris','🔴'

def score_eau_zone(water_L):
    if water_L>=50: return 0
    if water_L>25: return round((50-water_L)/25*50,2)
    if water_L>12: return round(50+(25-water_L)/13*35,2)
    return round(85+min((12-water_L)/12,1)*15,2)

# OPEX tech-spécifique (Étape 1)
def calc_opex_specifique(tech, prod_kgH2_an):
    if tech=='NH3': return prod_kgH2_an*NH3_YIELD*NH3_CATAL_COST/NH3_CATAL_LIFE, 'Catalyseur Haber-Bosch'
    if tech=='LH2': return prod_kgH2_an*(LH2_BOILOFF/100)*LH2_DAYS*LH2_PRICE, f'Boil-off {LH2_BOILOFF}%/j×{LH2_DAYS}j'
    if tech=='LOHC': return prod_kgH2_an*LOHC_OIL_KG*LOHC_OIL_COST*LOHC_DEGRAD/100, 'Dégradation huile DBT'
    return 0, '—'

# ═══════════════════════ AHP + MACBETH (Étape 1) ═════════════════════════════
CRITERIA = ['CAPEX','OPEX','Densite','LCOS','TRL','CO2_Real','Eau_Real']
DELTA_CAT = {0:0,1:0.05,2:0.10,3:0.15,4:0.20,5:0.25}
SENS_CRIT = {'CAPEX':'min','OPEX':'min','LCOS':'min','Densite':'max','TRL':'max','CO2_Real':'max','Eau_Real':'max'}

# Dans engine.py — remplacer la fonction ahp_weights()
def ahp_weights(matrix):
    """Version alignée avec AHPEngine de ETAPE1MACBETH.py"""
    m = np.array(matrix, dtype=float)
    n = m.shape[0]
    col_sums = m.sum(axis=0)
    norm = m / col_sums
    w = norm.mean(axis=1)
    w /= w.sum()
    Aw = m @ w
    lambda_max = np.mean(Aw / w)
    CI = (lambda_max - n) / (n - 1)
    RI = {3:.58, 4:.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}.get(n, 1.45)
    CR = CI / RI if RI else 0.0
    # Si CR >= 0.10 → poids égaux (même comportement qu'AHPEngine.is_valid)
    if CR >= 0.10:
        w = np.ones(n) / n
    return w, CR

AHP_SCENARIOS = {
    'EXPORT': [
        [1,1,3,1/2,2,1/5,1/4],[1,1,3,1/2,2,1/5,1/4],[1/3,1/3,1,1/3,1,1/7,1/6],
        [2,2,3,1,3,1/3,1/2],[1/2,1/2,1,1/3,1,1/5,1/4],[5,5,7,3,5,1,2],[4,4,6,2,4,1/2,1]],
    'Pôle_Industriel': [
        [1,2,4,1/2,3,3,3],[1/2,1,3,1/3,2,2,2],[1/4,1/3,1,1/5,1,1,1],
        [2,3,5,1,4,4,4],[1/3,1/2,1,1/4,1,1,1],[1/3,1/2,1,1/4,1,1,1],[1/3,1/2,1,1/4,1,1,1]],
    'SITE_ISOLE': [
        [1,1,1/2,1,1/4,1/2,1/5],[1,1,1/2,1,1/4,1/2,1/5],[2,2,1,2,1/2,1,1/3],
        [1,1,1/2,1,1/4,1/2,1/5],[4,4,2,4,1,3,1/2],[2,2,1,2,1/3,1,1/4],[5,5,3,5,2,4,1]],
}

def build_macbeth_matrix(vals, sense='min'):
    n=len(vals); mat=np.zeros((n,n),dtype=int); v=np.array(vals,float); rng=np.ptp(v)
    if rng<1e-12: return mat
    for i in range(n):
        for j in range(n):
            if i==j: continue
            diff=(v[j]-v[i])/rng if sense=='min' else (v[i]-v[j])/rng
            a=abs(diff)
            cat=5 if a>.75 else (4 if a>.5 else (3 if a>.3 else (2 if a>.15 else (1 if a>.05 else 0))))
            mat[i,j]=cat if diff>0 else -cat
    return mat

def fix_transitivity(mat):
    m=mat.copy(); n=m.shape[0]; v=0
    for i in range(n):
        for j in range(n):
            if m[i,j]<=0: continue
            for k in range(n):
                if k==i or m[j,k]<=0: continue
                if m[i,k]<=0:
                    c=max(1,min(int(m[i,j]),int(m[j,k]))-1)
                    m[i,k]=c; m[k,i]=-c; v+=1
    return m,v

def solve_macbeth_lp(n_techs, cat_matrix):
    if n_techs<2: return [0.5]*n_techs
    sums=cat_matrix.sum(axis=1); ib=int(np.argmax(sums)); iw=int(np.argmin(sums))
    if ib==iw: return [0.5]*n_techs
    A_ub,b_ub=[],[]
    for i in range(n_techs):
        for j in range(n_techs):
            k=int(cat_matrix[i,j])
            if k<=0: continue
            row=[0.0]*n_techs; row[i]=-1;row[j]=1; A_ub.append(row); b_ub.append(-DELTA_CAT[k])
    rb=[0.0]*n_techs; rb[ib]=1; rw=[0.0]*n_techs; rw[iw]=1
    try:
        res=linprog([0]*n_techs,A_ub=A_ub or None,b_ub=b_ub or None,
            A_eq=[rb,rw],b_eq=[1,0],bounds=[(0,1)]*n_techs,method='highs')
        if res.status==0: return list(res.x)
    except: pass
    # fallback
    lo,hi=sums.min(),sums.max()
    if hi==lo: return [0.5]*n_techs
    return [(s-lo)/(hi-lo) for s in sums]

# PAR (aligné avec ETAPE1MACBETH.py) :
def run_macbeth_full(techs, data_df, weights_dict):
    scores = {t: 0.0 for t in techs}
    for crit, w in weights_dict.items():
        if w < 1e-9 or crit not in data_df.columns: continue
        vals = data_df[crit].values
        mat = build_macbeth_matrix(vals, SENS_CRIT.get(crit, 'min'))
        mat, n_viol = fix_transitivity(mat)
        
        # LP avec ancrage best=1, worst=0 (identique à solve_macbeth_lp)
        n = len(techs)
        sums = mat.sum(axis=1)
        idx_best = int(np.argmax(sums))
        idx_worst = int(np.argmin(sums))
        
        A_ub, b_ub = [], []
        for i in range(n):
            for j in range(n):
                k = int(mat[i, j])
                if k <= 0: continue
                row = [0.0] * n
                row[i] = -1.0; row[j] = 1.0
                A_ub.append(row); b_ub.append(-DELTA_CAT[k])
        
        row_best = [0.0]*n; row_best[idx_best] = 1.0
        row_worst = [0.0]*n; row_worst[idx_worst] = 1.0
        
        try:
            from scipy.optimize import linprog
            res = linprog([0.0]*n,
                A_ub=A_ub or None, b_ub=b_ub or None,
                A_eq=[row_best, row_worst], b_eq=[1.0, 0.0],
                bounds=[(0.0, 1.0)]*n, method='highs')
            if res.status == 0:
                s_dict = {techs[i]: float(res.x[i]) for i in range(n)}
            else:
                raise ValueError("LP infaisable")
        except:
            # Fallback rang
            lo, hi = sums.min(), sums.max()
            s_dict = {techs[i]: float((sums[i]-lo)/(hi-lo)) if hi > lo else 0.5
                      for i in range(n)}
        
        for t, v in s_dict.items():
            scores[t] += w * v * 100.0
    return scores

def analyse_robustesse(techs, data_df, weights_dict, n_pert=200, pct=0.20):
    """Robustesse Monte Carlo sur les poids AHP."""
    np.random.seed(42)
    crit=list(weights_dict.keys()); w_arr=np.array([weights_dict[c] for c in crit])
    all_ranks={t:[] for t in techs}
    for _ in range(n_pert):
        noise=np.random.uniform(1-pct,1+pct,len(w_arr))
        wp=w_arr*noise; wp/=wp.sum()
        sc=run_macbeth_full(techs,data_df,dict(zip(crit,wp)))
        ranked=sorted(sc.items(),key=lambda x:-x[1])
        for rg,(tech,_) in enumerate(ranked,1): all_ranks[tech].append(rg)
    results={}
    for t in techs:
        r=all_ranks[t]
        results[t]={'median':int(np.median(r)),'std':round(np.std(r),2),
            'pct_top1':round(100*r.count(1)/n_pert,1)}
    return results

# ═══════════════════════ STOCKAGE LCOS (modele_stockage.py) ═══════════════════
def compression_work(p_in,p_out,stages=2,eta_is=0.75,eta_mec=0.95):
    if p_out<=p_in: return 0
    ratio=p_out/p_in; exp=(GAMMA_H2-1)/(stages*GAMMA_H2)
    w=stages*(GAMMA_H2/(GAMMA_H2-1))*(R_IDEAL/M_H2)*T_STD*(ratio**exp-1)
    return w/3.6e6/(eta_is*eta_mec)

STORAGE_LEARNING = {
    "GH2_tank":{"c24":600,"r1":.030,"r2":.020,"fl":200},
    "GH2_tank700":{"c24":900,"r1":.028,"r2":.018,"fl":280},
    "LH2_liq":{"c24":200e6,"r1":.040,"r2":.025,"fl":80e6},
    "NH3_synth":{"c24":130e6,"r1":.025,"r2":.015,"fl":60e6},
    "NH3_crack":{"c24":90e6,"r1":.050,"r2":.030,"fl":20e6},
    "LOHC_sys":{"c24":100e6,"r1":.035,"r2":.025,"fl":45e6},
    "EMeth_syn":{"c24":75e6,"r1":.030,"r2":.020,"fl":35e6},
    "Caverne":{"c24":400,"r1":.010,"r2":.005,"fl":300},
}
def storage_capex_lr(key,yr):
    c=STORAGE_LEARNING[key]
    if yr<=2024: return c["c24"]
    if yr<=2030: v=c["c24"]*(1-c["r1"])**(yr-2024)
    else: v=c["c24"]*(1-c["r1"])**6*(1-c["r2"])**(yr-2030)
    return max(v,c["fl"])

JOURS_STOCKAGE_DEFAULT = {"Ouarzazate":20,"Laayoune":10,"Dakhla":7,"Tanger":15,
    "Jorf_Lasfar":12,"Guelmim":18,"Boujdour":12,"Agadir":10,"Casablanca":12,
    "Nador":14,"Marrakech":15,"Midelt":18,"_default":14}

PRIX_ELEC_REGION = {"Dakhla":0.022,"Ouarzazate":0.030,"Tanger":0.035,
    "Jorf_Lasfar":0.038,"Laayoune":0.025,"Guelmim":0.028,"Boujdour":0.024,
    "Agadir":0.032,"Casablanca":0.035,"Nador":0.033,"Marrakech":0.034,"Midelt":0.030,
    "_default":0.032}

TECH_DISPO_REGION = {
    "Dakhla":["GH2_350bar","GH2_700bar","LH2","NH3","LOHC","eMethanol"],
    "Ouarzazate":["GH2_350bar","GH2_700bar","NH3","eMethanol"],
    "Laayoune":["GH2_350bar","GH2_700bar","LH2","NH3","LOHC","eMethanol"],
    "Jorf_Lasfar": ["GH2_350bar", "GH2_700bar", "NH3", "LOHC", "Caverne_saline", "eMethanol"],
     "Tanger":     ["GH2_350bar", "GH2_700bar", "LH2", "NH3", "LOHC", "Caverne_saline", "eMethanol"],
    "Guelmim":["GH2_350bar","GH2_700bar","NH3","eMethanol"],
    "Boujdour":["GH2_350bar","GH2_700bar","LH2","NH3","eMethanol"],
    "Agadir":["GH2_350bar","GH2_700bar","NH3","LOHC","eMethanol"],
    "Casablanca":["GH2_350bar","GH2_700bar","NH3","LOHC","eMethanol"],
    "Nador":["GH2_350bar","GH2_700bar","NH3","LOHC","eMethanol"],
    "Marrakech":["GH2_350bar","GH2_700bar","NH3","eMethanol"],
    "Midelt":["GH2_350bar","GH2_700bar","NH3","eMethanol"],
    "_default":["GH2_350bar","GH2_700bar","LH2","NH3","LOHC","Caverne","eMethanol"],
}

BENCHMARKS_LCOS = {
    "GH2_350bar":{"min":0.30,"max":1.50,"src":"IEA 2024"},
    "GH2_700bar":{"min":0.50,"max":2.00,"src":"IEA 2024"},
    "LH2":{"min":1.50,"max":4.00,"src":"IRENA 2023"},
    "NH3":{"min":1.20,"max":3.50,"src":"IEA 2024"},
    "LOHC":{"min":1.50,"max":4.50,"src":"IRENA 2023"},
    "Caverne":{"min":0.10,"max":0.50,"src":"IEA 2024"},
    "eMethanol":{"min":1.00,"max":3.50,"src":"IRENA 2023"},
}

PRIX_MARCHE = {"europe_2024":6.0,"europe_2030":4.5,"europe_2035":3.5,"europe_2040":3.0}

SCENARIOS_STOCKAGE = {
    "optimiste":{"lt_adj":0.9,"wacc":0.06,"px_adj":0.85},
    "central":{"lt_adj":1.0,"wacc":0.08,"px_adj":1.0},
    "pessimiste":{"lt_adj":1.15,"wacc":0.11,"px_adj":1.25},
}
# ── Économies d'échelle réservoirs ──────────────────────────────────────────
TANK_SCALE = {
    "GH2_tank":    {'p_ref': 600,  'ref_kg': 1000, 'alpha': 0.70},
    "GH2_tank700": {'p_ref': 900,  'ref_kg': 1000, 'alpha': 0.70},
    "LH2_liq":     {'p_ref': 1200, 'ref_kg': 5000, 'alpha': 0.75},
}

def p_tank_scaled(tank_key, cap_kg, annee):
    sc = TANK_SCALE.get(tank_key, {'p_ref': 600, 'ref_kg': 1000, 'alpha': 0.70})
    p_base = storage_capex_lr(tank_key, annee)
    p_scaled = p_base * (max(cap_kg, 1) / sc['ref_kg']) ** (sc['alpha'] - 1)
    return max(p_scaled, 50)

# ── Économies d'échelle NH3 ──────────────────────────────────────────────────
NH3_CAPEX_REF = {
    'synth': {'usd_per_t_an': 600, 'ref_t_an': 100000, 'alpha': 0.65, 'min_m': 20e6},
    'crack': {'usd_per_t_an': 400, 'ref_t_an': 100000, 'alpha': 0.65, 'min_m': 8e6},
}

def nh3_capex_scaled(component, qty_h2_kg_an, annee):
    ref = NH3_CAPEX_REF[component]
    qty_nh3_t_an = qty_h2_kg_an * 5.6 / 1000
    lr_factor = (1 - 0.02) ** max(0, annee - 2024)
    capex = ref['usd_per_t_an'] * ref['ref_t_an'] * \
            (qty_nh3_t_an / ref['ref_t_an']) ** ref['alpha'] * lr_factor
    return max(capex, ref['min_m'])
def lcos_gh2(qty,cap_kg,p_tank,p_comp_per_kw,e_comp,px,wacc=.08,lt=20):
    # p_tank = $/kgH2 storage, cap_kg = qty*jours/365
    # p_comp_per_kw already includes scaling
    capex_tank = p_tank * cap_kg
    capex_comp = p_comp_per_kw  # already scaled for throughput
    capex = capex_tank + capex_comp
    crf=CRF(wacc,lt)
    return round((capex*crf+.02*capex+e_comp*qty*px)/qty,4)

def lcos_gh2_700(qty,cap_kg,p_tank,p_comp_per_kw,e_comp,px,wacc=.08,lt=20):
    capex_tank = p_tank * cap_kg
    capex_comp = p_comp_per_kw
    capex = capex_tank + capex_comp
    crf=CRF(wacc,lt)
    return round((capex*crf+.022*capex+e_comp*qty*px)/qty,4)

def lcos_lh2(qty,cap_kg,p_res,p_liq,e_liq,px,boil,j,wacc=.08,lt=20):
    perte=1-(1-boil/100)**j; out=max(qty*(1-perte),qty*.01)
    capex=p_res*cap_kg+p_liq; crf=CRF(wacc,lt)
    return round((capex*crf+.025*capex+e_liq*qty*px)/out,4)

def lcos_nh3(qty,cs,cstock,ccrack,es,ec,px,eta_s,eta_c,wacc=.08,lt=25):
    eta=eta_s/100*.99*eta_c/100; capex=cs+cstock+ccrack; crf=CRF(wacc,lt)
    return round((capex*crf+.03*capex+(es+ec+3.6)*qty*px)/(qty*eta),4)
# APRÈS
CHALEUR_LOHC_KWH_KG = 5.51   # kWh_th/kgH2 — déhydrogénation DBT (40 kJ/mol H2)

def lcos_lohc(qty, ch, cs, cd, eh, ed, ch_th, px, perte=.08, wacc=.08, lt=15):
    """
    ch_th : coût de la chaleur en USD/kWh_th (ex: 0.03-0.06 USD/kWh)
    CHALEUR_LOHC_KWH_KG : énergie thermique nécessaire à la déhydrogénation
    """
    eta = 1 - perte
    capex = ch + cs + cd
    crf = CRF(wacc, lt)
    cout_chaleur = ch_th * CHALEUR_LOHC_KWH_KG * qty   # USD/an (physiquement cohérent)
    return round(
        (capex * crf + .03 * capex + (eh + ed) * qty * px + cout_chaleur) / (qty * eta),
        4
    )

def lcos_cavern(qty,cap_kwh_kg,e_comp,px,eff=.98,wacc=.08,lt=50):
    capex=cap_kwh_kg*qty*LHV_H2; crf=CRF(wacc,lt)
    return round((capex*crf+.01*capex+e_comp*qty*px)/(qty*eff),4)

def lcos_emethanol(qty,cs,csto,cr,es,er,px,eta_s,eta_r,wacc=.08,lt=20):
    eta=eta_s/100*.998*eta_r/100; capex=cs+csto+(cr or 0); crf=CRF(wacc,lt)
    return round((capex*crf+.03*capex+(es+er+1.5)*qty*px)/(qty*eta),4)

def run_storage_optimizer(region, annee, scenario, qty, lcoh_prod, jours_override=None,caverne_dispo=False):
    """LCOS — synchronisé avec modele_stockage.py StorageOptimizer.run_all()"""
    print(f"DEBUG run_storage — qty={qty:,.0f} kg/an = {qty/1000:.0f} t/an")
    print(f"DEBUG                cap_kg_day={qty/365:,.0f} kg/j vs REF=50_000 kg/j")
    sc = SCENARIOS_STOCKAGE[scenario]
    px = PRIX_ELEC_REGION.get(region, 0.032) * sc['px_adj']
    wacc = sc['wacc']

    if jours_override is not None:
        j = jours_override
        source_jours = f"profil 8760h ({j}j)"
    else:
        j = JOURS_STOCKAGE_DEFAULT.get(region, 14)
        source_jours = f"défaut région ({j}j)"

    techs_dispo = TECH_DISPO_REGION.get(region, TECH_DISPO_REGION['_default'])
    cap_kg = qty * j / 365
    results = []
    if not caverne_dispo:
             techs_dispo = [t for t in techs_dispo
           if t not in ('Caverne', 'Caverne_saline')]

    for tech in techs_dispo:
        try:
            
            if tech == "GH2_350bar":
                e_comp = compression_work(1, 350, 3)
                p_tank = p_tank_scaled("GH2_tank", cap_kg, annee)
                p_comp = 1200 * e_comp * 0.95
                capex  = p_tank * cap_kg + p_comp
                crf    = CRF(wacc, int(20 * sc['lt_adj']))
                lcos   = round((capex * crf + 0.02 * capex + e_comp * qty * px) / qty, 4)
                eff, co2 = 0.97, 0.8

            elif tech == "GH2_700bar":
                e_comp   = compression_work(1, 700, 5)
                p_tank   = p_tank_scaled("GH2_tank700", cap_kg, annee)
                p_comp   = 1500 * e_comp * 0.95
                capex    = p_tank * cap_kg + p_comp
                crf      = CRF(wacc, int(20 * sc['lt_adj']))
                lcos     = round((capex * crf + 0.022 * capex + e_comp * qty * px) / qty, 4)
                eff, co2 = 0.96, 1.1

            elif tech == "LH2":
                e_liq          = 9.5
                REF_CAP_KG_DAY = 50_000
                REF_CAPEX_USD  = storage_capex_lr("LH2_liq", annee)
                ALPHA          = 0.65

                cap_kg_day     = max(qty / 365, 1)
                p_liq_scaled   = REF_CAPEX_USD * (cap_kg_day / REF_CAP_KG_DAY) ** ALPHA

                # Réservoir cryogénique : économies d'échelle (ref 1000 kg → 200 $/kg)
                # IEA 2024 : grand réservoir LH2 = 50-150 $/kgH2
                REF_TANK_KG    = 10_000        # kg référence réservoir
                REF_TANK_PRICE = 200           # $/kg à 10 t
                ALPHA_TANK     = 0.70
                p_tank_lh2     = REF_TANK_PRICE * (max(cap_kg, REF_TANK_KG) / REF_TANK_KG) ** (ALPHA_TANK - 1)
                # Plancher 50 $/kg (grand réservoir IEA) — plafond 400 $/kg (petit)
                p_tank_lh2     = max(50, min(p_tank_lh2, 400))

                perte          = 1 - (1 - 0.002) ** j
                out            = max(qty * (1 - perte), qty * 0.01)
                capex          = p_tank_lh2 * cap_kg + p_liq_scaled
                crf            = CRF(wacc, int(20 * sc['lt_adj']))
                lcos           = round(
                    (capex * crf + 0.025 * capex + e_liq * qty * px) / out, 4)
                eff            = round(1 - perte, 4)
                co2            = 4.5

            elif tech == "NH3":
                cs_nh3  = nh3_capex_scaled('synth', qty, annee)
                ccrack  = nh3_capex_scaled('crack', qty, annee)
                cstock  = 320 * qty / 1000
                eta_rt  = (85 / 100) * 0.99 * (85 / 100)
                capex   = cs_nh3 + cstock + ccrack
                crf     = CRF(wacc, int(25 * sc['lt_adj']))
                lcos    = round(
                    (capex * crf + 0.03 * capex + (10.5 + 10.07 + 3.6) * qty * px)
                    / (qty * eta_rt), 4)
                eff     = round(eta_rt, 4)
                co2     = 2.2

            elif tech == "LOHC":
                ch        = storage_capex_lr("LOHC_sys", annee) * 0.8
                cs_l      = 18e6
                cd        = storage_capex_lr("LOHC_sys", annee) * 1.1
                eta       = 1 - 0.08
                capex     = ch + cs_l + cd
                crf       = CRF(wacc, int(15 * sc['lt_adj']))
                heat_cost = 8.0 * qty * 0.03
                lcos      = round(
                    (capex * crf + 0.03 * capex + (4.5 + 3.0) * qty * px + heat_cost)
                    / (qty * eta), 4)
                eff, co2  = round(eta, 4), 1.8

            elif tech in ("Caverne","Caverne_saline"):
                cap_kwh_kg = 3.0
                e_comp     = 1.5
                eff_c      = 0.98
                capex      = cap_kwh_kg * qty * LHV_H2
                crf        = CRF(wacc, int(50 * sc['lt_adj']))
                lcos       = round(
                    (capex * crf + 0.01 * capex + e_comp * qty * px)
                    / (qty * eff_c), 4)
                eff, co2   = eff_c, 0.3

            elif tech == "eMethanol":
                cs         = storage_capex_lr("EMeth_syn", annee) * 0.9
                csto       = 12e6
                eta_rt     = (74 / 100) * 0.998 * (68 / 100)
                capex      = cs + csto
                crf        = CRF(wacc, int(20 * sc['lt_adj']))
                co2_per_kg = (100 / 1000) * 7.5
                lcos       = round(
                    (capex * crf + 0.03 * capex + (7.5 + 11 + 1.5) * qty * px)
                    / (qty * eta_rt) + co2_per_kg, 4)
                eff        = round(eta_rt, 4)
                co2        = 3.5

            else:
                continue

            bench = BENCHMARKS_LCOS.get(tech, {})

            bench = BENCHMARKS_LCOS.get(tech, {})
            if bench and bench['min'] <= lcos <= bench['max']:
                bval = '✅ Dans plage'
            elif bench and lcos > bench.get('max', 999):
                bval = f'⚠ Au-dessus ({bench.get("max","")} — {bench.get("src","")})'
            else:
                bval = '—'

            prix_k     = f"europe_{annee}" if annee in [2024, 2030, 2035, 2040] else "europe_2030"
            prix_m     = PRIX_MARCHE.get(prix_k, 4.5)
            lcoh_total = lcos + lcoh_prod

            results.append({
                'tech': tech, 'LCOS': lcos, 'LCOH_total': lcoh_total,
                'eff': eff, 'co2': co2, 'jours': j,
                'source_jours': source_jours, 'cap_kg': round(cap_kg),
                'region': region, 'annee': annee, 'validation': bval,
                'marge': round(prix_m - lcoh_total, 3),
                'rentable': (prix_m - lcoh_total) > 0,
                'prix_marche': prix_m,
            })

        except Exception as e:
            results.append({
                'tech': tech, 'LCOS': None, 'LCOH_total': None,
                'eff': 0, 'co2': 0, 'jours': j,
                'source_jours': source_jours, 'cap_kg': 0,
                'region': region, 'annee': annee,
                'validation': f'⛔ Erreur: {e}',
                'rentable': False, 'marge': 0, 'prix_marche': 0,
            })

    return sorted([r for r in results if r['LCOS'] is not None and r['LCOS'] > 0.01],
                  key=lambda x: x['LCOS'])
# Monte Carlo stockage
MC_DIST_STORAGE = {
    "GH2_350bar":{"capex_res":(450,600,850),"px_elec":(.015,.025,.040),"wacc":(.05,.08,.12)},
    "GH2_700bar":{"capex_res":(700,900,1200),"px_elec":(.015,.025,.040),"wacc":(.05,.08,.12)},
    "LH2":{"p_liq":(150e6,200e6,260e6),"boiloff":(.12,.20,.32),"px_elec":(.015,.025,.040)},
    "NH3":{"cs":(100e6,130e6,170e6),"eta_c":(80,85,90),"px_elec":(.015,.025,.040)},
}

def run_mc_storage(tech, qty, n=3000, seed=42):
    rng=np.random.default_rng(seed)
    if tech not in MC_DIST_STORAGE: return None
    dist=MC_DIST_STORAGE[tech]
    samples={k:triang.rvs((c-a)/(b-a),loc=a,scale=b-a,size=n,random_state=rng) for k,(a,c,b) in dist.items()}
    if tech in ("GH2_350bar","GH2_700bar"):
        p_bar=350 if tech=="GH2_350bar" else 700
        ec=compression_work(1,p_bar,3 if p_bar==350 else 5)
        cap=samples["capex_res"]*qty*15/365+6e6
        crf_v=(samples["wacc"]*(1+samples["wacc"])**20)/((1+samples["wacc"])**20-1)
        lcos_v=(cap*crf_v+.02*cap+ec*qty*samples["px_elec"])/qty
    elif tech=="LH2":
        perte=1-(1-samples["boiloff"]/100)**10; out=np.maximum(qty*(1-perte),qty*.01)
        cap=1200*qty*10/365+200e6
        lcos_v=(cap*.10185+.025*cap+9.5*qty*samples["px_elec"])/out
    else:
        cap=130e6+300*qty/1000+90e6; crf_v=CRF(.08,25)
        eta=.72*.99*(samples["eta_c"]/100)
        lcos_v=(cap*crf_v+.03*cap+29.1*qty*samples["px_elec"])/(qty*eta)
    lcos_v=np.clip(lcos_v,.1,25)
    return {'mean':round(float(np.mean(lcos_v)),3),'P10':round(float(np.percentile(lcos_v,10)),3),
        'P50':round(float(np.median(lcos_v)),3),'P90':round(float(np.percentile(lcos_v,90)),3),
        'CV':round(float(np.std(lcos_v)/np.mean(lcos_v)*100),1)}

# ═══════════════════════ PRODUCTION 8760h (Étape 2 simplifié) ═════════════════
CALIB_PROFILS = {
    'Dakhla':{'CF_sol':.197,'CF_eol':.415},'Ouarzazate':{'CF_sol':.198,'CF_eol':.225},
    'Laayoune':{'CF_sol':.199,'CF_eol':.337},'Tanger':{'CF_sol':.168,'CF_eol':.156},
    'Jorf_Lasfar':{'CF_sol':.173,'CF_eol':.128},'Agadir':{'CF_sol':.192,'CF_eol':.164},
    'Boujdour':{'CF_sol':.200,'CF_eol':.384},'Casablanca':{'CF_sol':.171,'CF_eol':.105},
    'Nador':{'CF_sol':.163,'CF_eol':.180},'Marrakech':{'CF_sol':.190,'CF_eol':.060},
    'Midelt':{'CF_sol':.201,'CF_eol':.150},'Guelmim':{'CF_sol':.177,'CF_eol':.165},
}

def simulate_8760h(region, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, tech='PEM',cal_override=None):
    """Simulation simplifiée 8760h (Étape 2) — Mode flexible.
    A6: retourne aussi le profil H₂ horaire pour calcul jours stockage."""
    eff_map={'PEM':55,'AEL':52,'SOEC':42}; minload_map={'PEM':.10,'AEL':.20,'SOEC':.15}
    eff=eff_map.get(tech,55); minload=minload_map.get(tech,.10)
    P_elec=ELEC_MW*1e3; BAT_kWh=BAT_MWH*1e3
    P_elec_min = P_elec * minload
    cal = CALIB_PROFILS.get(region, None)
    if cal is None:
        cal = {'CF_sol': 0.190, 'CF_eol': 0.200}
    np.random.seed(42); h=np.arange(8760); dj=h//24; hj=h%24
    decl=23.45*np.sin(2*np.pi*(dj-81)/365); hs=hj-12
    az=np.clip(np.cos(np.radians(decl))*np.cos(np.radians(15*hs)),0,1)
    CF_PV=np.clip(az*(1+.15*np.sin(2*np.pi*(dj-172)/365))*cal['CF_sol']*2.5,0,.95)
    CF_PV*=cal['CF_sol']/(CF_PV.mean()+1e-9); CF_PV=np.clip(CF_PV,0,.95)
    k=2; c_w=max(cal['CF_eol'],.01)/.886*12; u=np.random.uniform(0,1,8760)
    v_h=c_w*(-np.log(1-u+1e-9))**(1/k)*(1+.2*np.cos(2*np.pi*(dj-30)/365))
    def pwc(v):
        cf=np.zeros_like(v); m=(v>=3)&(v<12)
        cf[m]=np.clip((-0.6994*v[m]**3+19.481*v[m]**2-90.983*v[m]+121)/2000,0,1)
        cf[(v>=12)&(v<25)]=1; return cf
    raw_eol=pwc(v_h)*.85
    mean_eol=raw_eol.mean()
    if mean_eol > 1e-6 and cal['CF_eol'] > 0.01:
        CF_EOL=np.clip(raw_eol*cal['CF_eol']/mean_eol,0,1)
    else:
        CF_EOL=np.zeros(8760)
    P_PV=CF_PV*PV_MW*1e3; P_EOL=CF_EOL*EOL_MW*1e3; P_enr=P_PV+P_EOL
    has_bat=BAT_MWH>0; SOC=BAT_kWh*.5 if has_bat else 0
    SOC_min=BAT_kWh*.10 if has_bat else 0; SOC_max=BAT_kWh*.90 if has_bat else 0
    E_f=0; h_ok=0; E_curt=0
    h2_profil = np.zeros(8760)  # A6: profil horaire H₂ (kg/h)
    for i in range(8760):
        P_dispo = P_enr[i]
        if has_bat:
            if P_dispo > P_elec:
                charge = min((P_dispo - P_elec)*.92, SOC_max - SOC)
                SOC += charge
                E_curt += max(0, P_dispo - P_elec - charge/.92)
                P_effective = P_elec
            elif P_dispo < P_elec:
                deficit = P_elec - P_dispo
                bat_dispo = (SOC - SOC_min) * .92
                bat_fourni = min(deficit, bat_dispo)
                SOC -= bat_fourni / .92
                P_effective = P_dispo + bat_fourni
            else:
                P_effective = P_elec
        else:
            P_effective = min(P_dispo, P_elec)
            E_curt += max(0, P_dispo - P_elec)
        if P_effective >= P_elec_min:
            elec_used = min(P_effective, P_elec)
            E_f += elec_used
            h2_profil[i] = elec_used / eff
            h_ok += 1   # ← CORRECTION : toute heure de fonctionnement comptée

    H2=E_f/eff; fiab=h_ok/8760
    return {'H2_kg_an':float(H2),'fiabilite':float(fiab),'E_fournie_MWh':float(E_f/1000),
        'E_curtail_MWh':float(E_curt/1000),'h_full_load':h_ok,'avec_bat':has_bat,
        'h2_profil':h2_profil}  # A6: profil horaire ajouté


def simulate_8760h_constant(region, PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                             BUFFER_H2_KG, tech='PEM', demande_kg_h=None,cal_override=None):
    """A4: Simulation 8760h avec DEMANDE H₂ CONSTANTE 24/7."""
    eff_map = {'PEM': 55, 'AEL': 52, 'SOEC': 42}
    minload_map = {'PEM': .10, 'AEL': .20, 'SOEC': .15}
    eff = eff_map.get(tech, 55)
    minload = minload_map.get(tech, .10)
    P_elec = ELEC_MW * 1e3
    P_elec_min = P_elec * minload

    cal = CALIB_PROFILS.get(region, None)
    if cal is None:
        cal = {'CF_sol': 0.190, 'CF_eol': 0.200}

    if demande_kg_h is None:
        demande_kg_h = P_elec / eff * 0.85

    # Générer profils ENR
    np.random.seed(42); h = np.arange(8760); dj = h // 24; hj = h % 24
    decl = 23.45 * np.sin(2 * np.pi * (dj - 81) / 365); hs = hj - 12
    az = np.clip(np.cos(np.radians(decl)) * np.cos(np.radians(15 * hs)), 0, 1)
    CF_PV = np.clip(az * (1 + .15 * np.sin(2 * np.pi * (dj - 172) / 365)) * cal['CF_sol'] * 2.5, 0, .95)
    CF_PV *= cal['CF_sol'] / (CF_PV.mean() + 1e-9)
    CF_PV = np.clip(CF_PV, 0, .95)
    k = 2; c_w = max(cal['CF_eol'], .01) / .886 * 12
    u = np.random.uniform(0, 1, 8760)
    v_h = c_w * (-np.log(1 - u + 1e-9)) ** (1 / k) * (1 + .2 * np.cos(2 * np.pi * (dj - 30) / 365))

    def pwc(v):
        cf = np.zeros_like(v); m = (v >= 3) & (v < 12)
        cf[m] = np.clip((-0.6994 * v[m] ** 3 + 19.481 * v[m] ** 2 - 90.983 * v[m] + 121) / 2000, 0, 1)
        cf[(v >= 12) & (v < 25)] = 1
        return cf

    raw_eol = pwc(v_h) * .85; mean_eol = raw_eol.mean()
    if mean_eol > 1e-6 and cal['CF_eol'] > 0.01:
        CF_EOL = np.clip(raw_eol * cal['CF_eol'] / mean_eol, 0, 1)
    else:
        CF_EOL = np.zeros(8760)

    P_PV = CF_PV * PV_MW * 1e3; P_EOL = CF_EOL * EOL_MW * 1e3; P_enr = P_PV + P_EOL

    # Batterie
    BAT_kWh = BAT_MWH * 1e3; has_bat = BAT_MWH > 0
    SOC_bat = BAT_kWh * .5 if has_bat else 0
    SOC_min_bat = BAT_kWh * .10 if has_bat else 0
    SOC_max_bat = BAT_kWh * .90 if has_bat else 0

    # Buffer H₂
    SOC_h2 = BUFFER_H2_KG * 0.5
    SOC_h2_max = BUFFER_H2_KG; SOC_h2_min = 0
    H2_delivre = 0; heures_ok = 0; H2_perdu = 0
    h2_profil = np.zeros(8760); buffer_trace = np.zeros(8760)

    for i in range(8760):
        P_dispo = P_enr[i]
        if has_bat:
            if P_dispo > P_elec:
                charge = min((P_dispo - P_elec) * .92, SOC_max_bat - SOC_bat)
                SOC_bat += charge; P_effective = P_elec
            elif P_dispo < P_elec:
                deficit = P_elec - P_dispo
                bat_d = (SOC_bat - SOC_min_bat) * .92
                bat_f = min(deficit, bat_d)
                SOC_bat -= bat_f / .92; P_effective = P_dispo + bat_f
            else:
                P_effective = P_elec
        else:
            P_effective = min(P_dispo, P_elec)

        if P_effective >= P_elec_min:
            h2_prod_h = min(P_effective, P_elec) / eff
        else:
            h2_prod_h = 0
        h2_profil[i] = h2_prod_h

        delta = h2_prod_h - demande_kg_h
        if delta >= 0:
            stockable = min(delta, SOC_h2_max - SOC_h2)
            SOC_h2 += stockable
            H2_perdu += (delta - stockable)
            H2_delivre += demande_kg_h; heures_ok += 1
        else:
            besoin = -delta
            if SOC_h2 >= besoin:
                SOC_h2 -= besoin
                H2_delivre += demande_kg_h; heures_ok += 1
            else:
                H2_delivre += h2_prod_h + SOC_h2; SOC_h2 = 0
        buffer_trace[i] = SOC_h2

    taux_service = heures_ok / 8760
    return {
        'H2_delivre_kg': float(H2_delivre),
        'H2_kg_an': float(H2_delivre),
        'taux_service': float(taux_service),
        'fiabilite': float(taux_service),
        'heures_ok': heures_ok,
        'h_full_load': heures_ok,
        'demande_kg_h': demande_kg_h,
        'demande_annuelle_kg': demande_kg_h * 8760,
        'taux_couverture': float(H2_delivre / (demande_kg_h * 8760)) if demande_kg_h > 0 else 0,
        'H2_perdu_kg': float(H2_perdu),
        'buffer_min_kg': float(buffer_trace.min()),
        'buffer_max_kg': float(buffer_trace.max()),
        'buffer_cycles': float((buffer_trace.max() - buffer_trace.min()) / max(BUFFER_H2_KG, 1)),
        'E_fournie_MWh': float(np.sum(h2_profil) * eff_map.get(tech, 52) / 1000),
        'E_curtail_MWh': 0,
        'avec_bat': has_bat,
        'h2_profil': h2_profil,
        'buffer_trace': buffer_trace,
        'mode': 'constant_24_7',
    }
 
# A6: Calcul jours stockage depuis profil (méthode réservoir équivalent)
def jours_stockage_from_profil(profil_8760h):
    demande_moyenne = np.mean(profil_8760h)
    if demande_moyenne <= 0: return 14.0
    
    # Vérification : si fiabilité < 80%, le profil est suspect
    heures_prod = np.sum(profil_8760h > 0)
    fiabilite_profil = heures_prod / 8760
    if fiabilite_profil < 0.70:
        # Fallback sur valeur régionale — le dimensionnement ENR est insuffisant
        return 14.0
    
    ecart = profil_8760h - demande_moyenne
    deficit_cumule = np.cumsum(ecart)
    amplitude = deficit_cumule.max() - deficit_cumule.min()
    prod_journaliere = demande_moyenne * 24
    jours = amplitude / prod_journaliere if prod_journaliere > 0 else 14
    return round(max(min(jours, 30.0), 1.0), 1)


# A7: Cache distances OSRM
def get_osrm_distance(lat1, lon1, lat2, lon2):
    """Appelle OSRM pour obtenir la distance routière réelle."""
    try:
        import requests
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        r = requests.get(url, timeout=8)
        data = r.json()
        if data.get("code") == "Ok":
            return round(data["routes"][0]["distance"] / 1000, 1), "OSRM"
    except: pass
    return round(haversine(lat1,lon1,lat2,lon2)*1.3, 1), "Haversine×1.3"

def build_distance_cache(sites_dict):
    """Construit le cache des distances OSRM pour toutes les paires de sites.
    Exécuté une seule fois, résultats sauvegardés."""
    import itertools, time, json
    cache = {}
    sites = list(sites_dict.keys())
    for orig, dest in itertools.product(sites, sites):
        if orig == dest: cache[f"{orig}→{dest}"] = {"km": 0, "src": "identique"}; continue
        o, d = sites_dict[orig], sites_dict[dest]
        km, src = get_osrm_distance(o['lat'], o['lon'], d['lat'], d['lon'])
        cache[f"{orig}→{dest}"] = {"km": km, "src": src}
        time.sleep(0.3)
    return cache

# APRÈS :
def calc_lcoh_detailed(PV_MW, EOL_MW, ELEC_MW, BAT_MWH, H2_kg_an,
                       tech='PEM', annee=2024):
    DR = .08
    PT = {'PEM':{'O':.03,'L':20}, 'AEL':{'O':.02,'L':25},
          'SOEC':{'O':.04,'L':10}}
    p = PT.get(tech, PT['PEM'])
    if H2_kg_an <= 0: return None
    # CAPEX dynamiques avec learning rate
    cx_pv   = capex_lr_unified('sol',  annee)
    cx_eol  = capex_lr_unified('eol',  annee)
    cx_elec = capex_lr_unified(tech,   annee)
    cx_bat  = capex_lr_unified('bat',  annee)
    ann_pv   = PV_MW   * 1e3 * cx_pv   * CRF(DR, 25)
    ann_eol  = EOL_MW  * 1e3 * cx_eol  * CRF(DR, 20)
    ann_elec = ELEC_MW * 1e3 * cx_elec * CRF(DR, p['L'])
    ann_bat  = BAT_MWH * 1e3 * cx_bat  * CRF(DR, 15)
    opx_pv   = PV_MW   * 1e3 * 12
    opx_eol  = EOL_MW  * 1e3 * 35
    opx_elec = ELEC_MW * 1e3 * cx_elec * p['O']
    opx_bat  = BAT_MWH * 1e3 * cx_bat  * .01
    c_eau=21.1/1000*.72*H2_kg_an
    total=(ann_pv+ann_eol+ann_elec+ann_bat+opx_pv+opx_eol+opx_elec+opx_bat+c_eau)/H2_kg_an
    return {'total':round(total,4),'PV':round((ann_pv+opx_pv)/H2_kg_an,4),
        'Eolien':round((ann_eol+opx_eol)/H2_kg_an,4),'Electrolyseur':round((ann_elec+opx_elec)/H2_kg_an,4),
        'Batterie':round((ann_bat+opx_bat)/H2_kg_an,4),'Eau':round(c_eau/H2_kg_an,4)}

# ═══════════════════════ TRANSPORT (Étape 4) ═══════════════════════════════════
TRANSPORT_MODES = {
    'Tube_trailer':{'fixe':.50,'km':.008,'co2':150,'desc':'Remorque tube (<200km)'},
    'Pipeline_reconverti':{'fixe':.15,'km':.0004,'co2':20,'desc':'Pipeline gaz reconverti (200-600km)'},
    'Pipeline_H2_neuf':{'fixe':.20,'km':.0006,'co2':25,'desc':'Pipeline H₂ neuf (>600km)'},
    'Pipeline_sous_marin':{'fixe':.30,'km':.005,'co2':15,'desc':'Sous-marin détroit (<50km)'},
    'Tanker_NH3':{'fixe':.10,'km_alpha':.11,'alpha':.65,'craq':.75,'co2':80,'cap':1.95,
        'desc':'Tanker NH₃ maritime (export)'},
}

LCOT_FACTEURS = {
    "Pipeline_H2_neuf":{2024:1,2030:.90,2035:.82,2040:.75,2050:.65},
    "Pipeline_reconverti":{2024:1,2030:.92,2035:.85,2040:.78,2050:.68},
    "Pipeline_sous_marin":{2024:1,2030:.90,2035:.82,2040:.75,2050:.65},
    "Tube_trailer":{2024:1,2030:.93,2035:.87,2040:.81,2050:.72},
    "Tanker_NH3":{2024:1,2030:.87,2035:.76,2040:.67,2050:.55},
}

def lcot_year_factor(mode,yr):
    f=LCOT_FACTEURS.get(mode,{}); bornes=sorted(f.keys())
    for i in range(len(bornes)-1):
        if bornes[i]<=yr<=bornes[i+1]:
            a=(yr-bornes[i])/(bornes[i+1]-bornes[i])
            return f[bornes[i]]+a*(f[bornes[i+1]]-f[bornes[i]])
    return f.get(max(bornes),1) if bornes else 1

def calc_lcot(mode, dist, annee=2024):
    m=TRANSPORT_MODES[mode]; f=lcot_year_factor(mode,annee)
    if 'km_alpha' in m:  # Tanker
        cost=m['fixe']+m['km_alpha']*(dist/1000)**m['alpha']+m['craq']
        return round(min(cost,m['cap'])*f,4)
    return round((m['fixe']+m['km']*dist)*f,4)

def mode_optimal_segment(d, terrestre, annee=2024):
    if terrestre:
        if d<200: mode='Tube_trailer'
        elif d<600: mode='Pipeline_reconverti'
        else: mode='Pipeline_H2_neuf'
    else:
        mode='Pipeline_sous_marin' if d<50 else 'Tanker_NH3'
    return mode, calc_lcot(mode,d,annee)

PORTS_MAROC = ['Tanger','Casablanca','Agadir','Nador','Jorf_Lasfar','Dakhla','Laayoune']

DEMANDE_NOEUDS = {
    "Rotterdam":{2024:0,2030:200,2035:350,2040:600,2050:1200},
    "Barcelone":{2024:0,2030:100,2035:180,2040:300,2050:600},
    "Marseille":{2024:0,2030:80,2035:140,2040:240,2050:480},
}
OFFRE_NOEUDS = {
    "Dakhla":{2024:30,2030:400,2035:800,2040:1500,2050:3000},
    "Laayoune":{2024:20,2030:300,2035:600,2040:1000,2050:2000},
    "Ouarzazate":{2024:5,2030:150,2035:250,2040:400,2050:700},
}
# ═══════════════════════ IDW — SITES ARBITRAIRES ══════════════════════════════
# Intégré depuis BASEDEDONNEES.py — Shepard 1968
# Permet d'estimer GHI, vent, coût eau pour tout point (lat, lon) du Maroc
# en interpolant depuis les 12 régions de référence.
# ══════════════════════════════════════════════════════════════════════════════
 
# ── Données de référence des 12 régions (source : T1 BASEDEDONNEES.py) ────────
REGIONS_REF_IDW = {
    "Laayoune"   : {"lat": 27.1253, "lon": -13.1625, "GHI": 2160, "vent": 7.8, "cout_eau": 0.75},
    "Dakhla"     : {"lat": 23.6848, "lon": -15.9572, "GHI": 2155, "vent": 9.0, "cout_eau": 0.70},
    "Boujdour"   : {"lat": 26.1000, "lon": -14.5000, "GHI": 2175, "vent": 8.5, "cout_eau": 0.80},
    "Guelmim"    : {"lat": 28.9870, "lon": -10.0572, "GHI": 1940, "vent": 5.5, "cout_eau": 0.95},
    "Jorf_Lasfar": {"lat": 33.1100, "lon":  -8.6300, "GHI": 1900, "vent": 5.0, "cout_eau": 0.50},
    "Ouarzazate" : {"lat": 30.9189, "lon":  -6.8934, "GHI": 2180, "vent": 5.5, "cout_eau": 1.00},
    "Agadir"     : {"lat": 30.4278, "lon":  -9.5981, "GHI": 2095, "vent": 5.5, "cout_eau": 0.48},
    "Tanger"     : {"lat": 35.7595, "lon":  -5.8340, "GHI": 1840, "vent": 9.5, "cout_eau": 0.52},
    "Casablanca" : {"lat": 33.5731, "lon":  -7.5898, "GHI": 1875, "vent": 4.5, "cout_eau": 0.50},
    "Nador"      : {"lat": 35.1681, "lon":  -2.9335, "GHI": 1785, "vent": 5.8, "cout_eau": 0.55},
    "Marrakech"  : {"lat": 31.6295, "lon":  -7.9811, "GHI": 2085, "vent": 4.0, "cout_eau": 0.65},
    "Midelt"     : {"lat": 32.6800, "lon":  -4.7340, "GHI": 2200, "vent": 5.5, "cout_eau": 0.90},
}
 
 
def interpolation_idw(lat, lon, parametre, power=2):
    """..."""
    weights = []
    values  = []
    for name, ref in REGIONS_REF_IDW.items():
        d = haversine(lat, lon, ref["lat"], ref["lon"])
        if d < 1.0:
            return ref[parametre], name
        w = 1.0 / (d ** power)
        weights.append(w)
        values.append(ref[parametre])
    total_w = sum(weights)
    result  = sum(w * v for w, v in zip(weights, values)) / total_w
    return round(result, 2), "IDW interpolé"


# ── Ratios ENR calibrés NSGA-II — doit être avant calcul_chaine_complete ──
RATIO_ENR_NSGA2 = {
    "Dakhla": 4.1, "Laayoune": 3.8, "Tanger": 3.5,
    "Boujdour": 4.0, "Ouarzazate": 3.2, "Midelt": 3.0,
    "Guelmim": 3.3, "Jorf_Lasfar": 3.5, "Agadir": 3.4,
    "Casablanca": 3.8, "Nador": 3.6, "Marrakech": 3.2,
}


def calc_lcoh_from_sim(PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                        H2_kg_an, tech='PEM', annee=2024):
    """LCOH depuis simulation réelle — aligné avec calc_lcoh_detailed."""
    if H2_kg_an <= 0: return None
    DR = 0.08
    PT = {'PEM':{'O':.03,'L':20},'AEL':{'O':.02,'L':25},'SOEC':{'O':.04,'L':10}}
    p  = PT.get(tech, PT['PEM'])
    cx_pv   = capex_lr_unified('sol', annee)
    cx_eol  = capex_lr_unified('eol', annee)
    cx_elec = capex_lr_unified(tech,  annee)
    cx_bat  = capex_lr_unified('bat', annee)
    total = (
        PV_MW   * 1e3 * cx_pv   * CRF(DR, 25) +
        EOL_MW  * 1e3 * cx_eol  * CRF(DR, 20) +
        ELEC_MW * 1e3 * cx_elec * CRF(DR, p['L']) +
        BAT_MWH * 1e3 * cx_bat  * CRF(DR, 15) +
        PV_MW   * 1e3 * 12 +
        EOL_MW  * 1e3 * 35 +
        ELEC_MW * 1e3 * cx_elec * p['O'] +
        9.0 * H2_kg_an / 1000 * 21.1
    )
    return round(total / H2_kg_an, 4)


def calcul_chaine_complete(site, annee, qh2, scenario_stock,
                            use_bat, mode_constant=False):
    """Calcul complet synchronisé avec Étapes 1-2-3-4."""
    from engine import SITES as SITES_REF  # fallback si besoin
    S = SITES_REF.get(site, {})
    qkg = qh2 * 1000
    cs = cf_solaire(S.get('ghi', 2000))
    ce = cf_eolien(S.get('ws', 7.0))
    ls = lcoe_sol(cs, annee)
    le = lcoe_eol(ce, annee)
    ch, we, lh = calc_hybride(ce, cs, le, ls)
    lcoh_r = {}
    for t, p in {'PEM':{'op':.03,'eta':55,'lt':20},
                  'AEL':{'op':.02,'eta':52,'lt':25},
                  'SOEC':{'op':.04,'eta':42,'lt':10}}.items():
        cx = capex_lr_unified(t, annee)
        lcoh_r[t] = calc_lcoh(lh, ch, cx, p['op'], p['eta'], 0.08, p['lt'])
    be = min(lcoh_r, key=lambda k: lcoh_r[k] or 999)
    eta_be = {'PEM':55,'AEL':52,'SOEC':42}[be]
    ch_real = max(ch, 0.01)
    ELEC_MW = max(5, qkg * (eta_be/1000) / (ch_real * 8760) * 0.85)
    ratio   = RATIO_ENR_NSGA2.get(site, 3.5)
    P_enr   = ELEC_MW * ratio
    we_eff  = we if we > 0.01 else 0
    PV_MW   = max(5, P_enr * (1 - we_eff))
    EOL_MW  = max(5, P_enr * we_eff) if we_eff > 0 else 0.0
    BAT_MWH = ELEC_MW * 4.0 if use_bat else 0
    sim = simulate_8760h(site, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, be)
    bl  = calc_lcoh_from_sim(PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                              sim['H2_kg_an'], be, annee)
    jours = jours_stockage_from_profil(sim['h2_profil'])
    stor  = run_storage_optimizer(site, annee, scenario_stock,
                                   qkg, bl, jours_override=jours)
    return {
        'be': be, 'bl': bl, 'ch': ch, 'we': we,
        'PV_MW': PV_MW, 'EOL_MW': EOL_MW,
        'ELEC_MW': ELEC_MW, 'BAT_MWH': BAT_MWH,
        'sim': sim, 'jours': jours, 'stor': stor,
    }


def parametres_site_arbitraire(lat, lon):
    """..."""
    ghi,  src_ghi  = interpolation_idw(lat, lon, "GHI")
    vent, src_vent = interpolation_idw(lat, lon, "vent")
    eau,  src_eau  = interpolation_idw(lat, lon, "cout_eau")
    distances = {
        name: haversine(lat, lon, ref["lat"], ref["lon"])
        for name, ref in REGIONS_REF_IDW.items()
    }
    region_proche = min(distances, key=distances.get)
    dist_proche   = round(distances[region_proche], 1)
    cs = cf_solaire(ghi)
    ce = cf_eolien(vent)
    return {
        "ghi": ghi, "ws": vent, "eau": eau,
        "dni": round(ghi * 1.05, 0),
        "CF_solaire": round(cs * 100, 2),
        "CF_eolien":  round(ce * 100, 2),
        "src_ghi": src_ghi, "src_vent": src_vent, "src_eau": src_eau,
        "methode": "IDW (Inverse Distance Weighting, p=2) — Shepard 1968",
        "region_proxy": region_proche,
        "dist_proxy_km": dist_proche,
        "lat": lat, "lon": lon,
    }


def site_arbitraire_vers_dict_app(lat, lon, nom_site,
                                   port_km=None, surface_km2=None,
                                   reseau="Bonne", caverne=False):
    """..."""
    params = parametres_site_arbitraire(lat, lon)
    if port_km is None:
        PORTS_REF = {
            "Laayoune": 20, "Dakhla": 8, "Boujdour": 12, "Guelmim": 55,
            "Jorf_Lasfar": 2, "Ouarzazate": 350, "Agadir": 5, "Tanger": 15,
            "Casablanca": 40, "Nador": 8, "Marrakech": 230, "Midelt": 380,
        }
        port_km = PORTS_REF.get(params["region_proxy"], 100)
    return {
        "lat": lat, "lon": lon,
        "ghi": params["ghi"], "dni": params["dni"],
        "ws":  params["ws"],  "eau": params["eau"],
        "port": port_km, "surface": surface_km2 or 500,
        "reseau": reseau,
        "dispo_eau": "Faible" if params["eau"] > 0.80 else "Bonne",
        "caverne": caverne,
        "src": f"IDW interpolé depuis {params['region_proxy']} "
               f"({params['dist_proxy_km']} km) — Shepard 1968",
        "_idw": params,
    }