
# -*- coding: utf-8 -*-


# app.py — H2 MOROCCO: Plateforme d'Aide a la Decision
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st, pandas as pd, numpy as np, math, json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import *


st.set_page_config(page_title="H2 Morocco", page_icon="H2", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════ HELPERS DOM STABLE ═══════════════════════════════════
import hashlib, time

if "session_dom_id" not in st.session_state:
    st.session_state.session_dom_id = hashlib.md5(
        f"{time.time()}{id(st)}".encode()
    ).hexdigest()[:8]

def dom_id(*parts):
    """Génère un ID stable pour les clés Streamlit et éléments HTML"""
    return f"{st.session_state.session_dom_id}-{'-'.join(str(p).replace(' ', '_') for p in parts)}"
# ═════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════ BASE DE DONNEES ══════════════════════════════════════
# Force SITES correct (écrase celui de engine si importé)
SITES = {
    'Laayoune':   {'lat':27.13,'lon':-13.16,'ghi':2160,'dni':2210,'ws':7.8,'eau':.75,'port':20,'surface':8500,'reseau':'Excellente','dispo_eau':'Faible','caverne':False,'src':'CDER/Solargis + NASA POWER v8.2'},
    'Dakhla':     {'lat':23.68,'lon':-15.96,'ghi':2155,'dni':2170,'ws':9.0,'eau':.70,'port':8,'surface':15000,'reseau':'Excellente','dispo_eau':'Faible','caverne':False,'src':'NASA POWER / CDER 7.5-8.5@40m'},
    'Boujdour':   {'lat':26.10,'lon':-14.50,'ghi':2175,'dni':2200,'ws':8.5,'eau':.80,'port':12,'surface':3500,'reseau':'Bonne','dispo_eau':'Tres faible','caverne':False,'src':'Solargis + GSA 2.0'},
    'Guelmim':    {'lat':28.99,'lon':-10.06,'ghi':1940,'dni':2100,'ws':5.5,'eau':.95,'port':55,'surface':2800,'reseau':'Excellente','dispo_eau':'Tres faible','caverne':False,'src':'GSA 2.0 + Atlas CDER'},
    'Jorf_Lasfar':{'lat':33.11,'lon':-8.63,'ghi':1900,'dni':1840,'ws':5.0,'eau':.50,'port':2,'surface':0.3,'reseau':'Excellente','dispo_eau':'Bonne','caverne':True,'src':'GSA 2.0 / CDER cote'},
    'Ouarzazate': {'lat':30.92,'lon':-6.89,'ghi':2180,'dni':2463,'ws':5.5,'eau':1.0,'port':350,'surface':4000,'reseau':'Bonne','dispo_eau':'Tres faible','caverne':False,'src':'MASEN NOOR mesure'},
    'Agadir':     {'lat':30.43,'lon':-9.60,'ghi':2095,'dni':2050,'ws':5.5,'eau':.48,'port':5,'surface':250,'reseau':'Excellente','dispo_eau':'Bonne','caverne':False,'src':'GSA 2.0 / CDER alize'},
    'Tanger':     {'lat':35.76,'lon':-5.83,'ghi':1840,'dni':1790,'ws':9.5,'eau':.52,'port':15,'surface':120,'reseau':'Excellente','dispo_eau':'Bonne','caverne':False,'src':'CDER 8-11@40m meilleur Maroc'},
    'Casablanca': {'lat':33.57,'lon':-7.59,'ghi':1875,'dni':1815,'ws':4.5,'eau':.50,'port':40,'surface':10,'reseau':'Excellente','dispo_eau':'Bonne','caverne':False,'src':'GSA 2.0'},
    'Nador':      {'lat':35.17,'lon':-2.93,'ghi':1785,'dni':1720,'ws':5.8,'eau':.55,'port':8,'surface':350,'reseau':'Bonne','dispo_eau':'Bonne','caverne':False,'src':'GSA 2.0 / CDER tramontane'},
    'Marrakech':  {'lat':31.63,'lon':-7.98,'ghi':2085,'dni':2350,'ws':4.0,'eau':.65,'port':230,'surface':180,'reseau':'Excellente','dispo_eau':'Moyenne','caverne':False,'src':'CDER/Solargis'},
    'Midelt':     {'lat':32.68,'lon':-4.73,'ghi':2200,'dni':2400,'ws':5.5,'eau':.90,'port':380,'surface':2500,'reseau':'Bonne','dispo_eau':'Faible','caverne':False,'src':'Noor Midelt MASEN'},
}
DESTINATIONS = {
    'Rotterdam':{'lat':51.92,'lon':4.48,'pays':'Pays-Bas','type':'Port H2'},
    'Barcelone':{'lat':41.39,'lon':2.17,'pays':'Espagne','type':'Port'},
    'Marseille':{'lat':43.30,'lon':5.37,'pays':'France','type':'Port'},
    'Algesiras':{'lat':36.14,'lon':-5.45,'pays':'Espagne','type':'Detroit'},
    'Paris':{'lat':48.86,'lon':2.35,'pays':'France','type':'Marche'},
    'Dakar':{'lat':14.72,'lon':-17.47,'pays':'Senegal','type':'Port Afrique'},
    'Canaries':{'lat':28.12,'lon':-15.44,'pays':'Espagne','type':'Hub insulaire'},
    'Casablanca':{'lat':33.57,'lon':-7.59,'pays':'Maroc','type':'Hub central'},
    'Tanger':{'lat':35.76,'lon':-5.83,'pays':'Maroc','type':'Port/Hub'},
    'Agadir':{'lat':30.43,'lon':-9.60,'pays':'Maroc','type':'Port/Hub'},
    'Jorf_Lasfar':{'lat':33.11,'lon':-8.63,'pays':'Maroc','type':'Port OCP'},
    'Nador':{'lat':35.17,'lon':-2.93,'pays':'Maroc','type':'Port'},
}
STORAGE_INFO = {
    'GH2_350bar':{'nom':'H2 350 bar','den':23.5,'trl':9,'col':'#546E7A'},
    'GH2_700bar':{'nom':'H2 700 bar','den':40.2,'trl':9,'col':'#78909C'},
    'LH2':{'nom':'H2 Liquide','den':70.8,'trl':7,'col':'#0097A7'},
    'NH3':{'nom':'Ammoniac','den':121,'trl':9,'col':'#D84315'},
    'LOHC':{'nom':'LOHC','den':57,'trl':7,'col':'#5D4037'},
    'Caverne':{'nom':'Caverne Saline','den':120,'trl':8,'col':'#689F38'},
    'eMethanol':{'nom':'e-Methanol','den':140,'trl':7,'col':'#7B1FA2'},
}
ELECS = {
    'PEM':{'c24':900,'l1':.08,'l2':.04,'f':180,'op':.03,'eta':55,'lt':20,'trl':8,'col':'#1565C0'},
    'AEL':{'c24':650,'l1':.06,'l2':.03,'f':200,'op':.02,'eta':52,'lt':25,'trl':9,'col':'#2E7D32'},
    'SOEC':{'c24':3500,'l1':.10,'l2':.06,'f':400,'op':.04,'eta':42,'lt':10,'trl':6,'col':'#6A1B9A'},
}
# ═══════════════════════ CACHE OSRM ═══════════════════════════════════════════
@st.cache_data(ttl=86400)
def load_or_build_distance_cache():
    cache_file = os.path.join(os.path.dirname(__file__), "distance_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    cache = {}
    all_nodes = {**{k:{'lat':v['lat'],'lon':v['lon']} for k,v in SITES.items()},
                 **{k:{'lat':v['lat'],'lon':v['lon']} for k,v in DESTINATIONS.items()}}
    for orig in SITES:
        for dest_k in list(SITES.keys()) + list(DESTINATIONS.keys()):
            if orig == dest_k: continue
            o, d = all_nodes[orig], all_nodes[dest_k]
            km, src = get_osrm_distance(o['lat'], o['lon'], d['lat'], d['lon'])
            cache[f"{orig}->{dest_k}"] = {"km": km, "src": src}
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    except: pass
    return cache

def get_cached_distance(orig, dest_k, cache):
    for fmt in [f"{orig}->{dest_k}", f"{orig}\u2192{dest_k}"]:
        if fmt in cache:
            return cache[fmt]["km"], cache[fmt]["src"]
    o = SITES.get(orig) or DESTINATIONS.get(orig)
    d = SITES.get(dest_k) or DESTINATIONS.get(dest_k)
    if o and d:
        return round(haversine(o['lat'],o['lon'],d['lat'],d['lon'])*1.3, 1), "Haversine x1.3"
    return 1000, "Estimation"

# ═══════════════════════ ROUTAGE ═══════════════════════════════════════════════
def find_routes(orig, dest_k, annee, dist_cache):
    o = SITES[orig]; d = DESTINATIONS[dest_k]
    dest_maroc = dest_k in SITES
    routes = []
    direct_dist = haversine(o['lat'], o['lon'], d['lat'], d['lon'])

    if dest_maroc:
        dist, src = get_cached_distance(orig, dest_k, dist_cache)
        mode, cost = mode_optimal_segment(dist, True, annee)
        routes.append({'label': f"{orig} > {dest_k} (direct)", 'segs': [
            {'de': orig, 'a': dest_k, 'dist': dist, 'mode': mode, 'cost': cost,
             'type': 'Domestique', 'src_dist': src}], 'cost': cost, 'dist': dist})
    else:
        # ─── cherche port le plus proche PARMI TOUS les ports
        # y compris orig lui-même s'il est un port
        ports_valides = [p for p in PORTS_MAROC if p in SITES]
        
        port_proche = min(
            ports_valides,
            key=lambda p: haversine(o['lat'], o['lon'],
                                    SITES[p]['lat'], SITES[p]['lon'])
        )

        # ─── Route via port le plus proche (même si c'est orig)
        if port_proche == orig:
            # orig EST un port — pas de segment terrestre
            dist_mer = haversine(o['lat'], o['lon'], d['lat'], d['lon'])
            m2, c2 = mode_optimal_segment(dist_mer, False, annee)
            routes.append({
                'label': f"{orig} > {dest_k} (direct maritime)",
                'segs': [{'de': orig, 'a': dest_k, 'dist': dist_mer,
                          'mode': m2, 'cost': c2, 'type': 'Export',
                          'src_dist': 'Haversine'}],
                'cost': c2, 'dist': dist_mer
            })
        else:
            d_port = haversine(o['lat'], o['lon'],
                               SITES[port_proche]['lat'], SITES[port_proche]['lon'])
            dist_mer = haversine(SITES[port_proche]['lat'], SITES[port_proche]['lon'],
                                 d['lat'], d['lon'])
            m1, c1 = mode_optimal_segment(d_port, True, annee)
            m2, c2 = mode_optimal_segment(dist_mer, False, annee)
            routes.append({
                'label': f"{orig} > {port_proche} > {dest_k}",
                'segs': [
                    {'de': orig, 'a': port_proche, 'dist': d_port,
                     'mode': m1, 'cost': c1, 'type': 'Domestique',
                     'src_dist': 'Haversine'},
                    {'de': port_proche, 'a': dest_k, 'dist': dist_mer,
                     'mode': m2, 'cost': c2, 'type': 'Export',
                     'src_dist': 'Haversine'}
                ],
                'cost': c1 + c2, 'dist': d_port + dist_mer
            })

        # ─── Routes alternatives via autres ports
        for port in ports_valides:
            if port == orig: continue   # déjà traité ci-dessus
            p = SITES[port]
            dist_port_dest = haversine(p['lat'], p['lon'], d['lat'], d['lon'])
            dist_orig_port_hav = haversine(o['lat'], o['lon'], p['lat'], p['lon'])
            if dist_orig_port_hav + dist_port_dest > direct_dist * 1.5: continue
            dt, src_t = get_cached_distance(orig, port, dist_cache)
            m1, c1 = mode_optimal_segment(dt, True, annee)
            m2, c2 = mode_optimal_segment(dist_port_dest, False, annee)
            routes.append({
                'label': f"{orig} > {port} > {dest_k}",
                'segs': [
                    {'de': orig, 'a': port, 'dist': dt, 'mode': m1,
                     'cost': c1, 'type': 'Domestique', 'src_dist': src_t},
                    {'de': port, 'a': dest_k, 'dist': dist_port_dest,
                     'mode': m2, 'cost': c2, 'type': 'Export',
                     'src_dist': 'Haversine'}
                ],
                'cost': c1 + c2, 'dist': dt + dist_port_dest
            })

    return sorted(routes, key=lambda r: r['cost'])

# ═══════════════════════ CSS ══════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700&display=swap');
.h2-hero{background:#006233;color:#fff;border-radius:10px;padding:22px 26px;margin-bottom:16px;font-family:'Source Sans 3',sans-serif}
.h2-hero h2{font-size:1.45rem;font-weight:700;margin:0 0 5px 0;color:#fff;letter-spacing:-0.01em}
.h2-hero p{font-size:0.82rem;margin:0;opacity:0.78;line-height:1.5;color:#fff}
.h2-kpi-row{display:flex;gap:10px;margin:10px 0 14px 0;flex-wrap:wrap}
.h2-kpi{flex:1 1 140px;min-width:120px;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px 10px;text-align:center;font-family:'Source Sans 3',sans-serif}
.h2-kpi-val{font-size:1.3rem;font-weight:700;line-height:1.25;margin-bottom:3px}
.h2-kpi-lbl{font-size:0.67rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;color:#64748b;line-height:1.35;overflow-wrap:break-word;word-break:break-word}
.h2-kpi.green{border-left:3px solid #006233}.h2-kpi.green .h2-kpi-val{color:#006233}
.h2-kpi.amber{border-left:3px solid #d97706}.h2-kpi.amber .h2-kpi-val{color:#d97706}
.h2-kpi.blue{border-left:3px solid #1565c0}.h2-kpi.blue .h2-kpi-val{color:#1565c0}
.h2-kpi.red{border-left:3px solid #c1272d}.h2-kpi.red .h2-kpi-val{color:#c1272d}
.h2-sec{font-family:'Source Sans 3',sans-serif;font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#006233;border-bottom:2px solid #006233;padding-bottom:6px;margin:20px 0 10px 0}
.h2-ok{font-family:'Source Sans 3',sans-serif;background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;border-radius:6px;padding:12px 16px;margin:8px 0;font-size:0.84rem;color:#14532d;line-height:1.55}
.h2-warn{font-family:'Source Sans 3',sans-serif;background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #d97706;border-radius:6px;padding:12px 16px;margin:8px 0;font-size:0.84rem;color:#78350f;line-height:1.55}
.h2-err{font-family:'Source Sans 3',sans-serif;background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #dc2626;border-radius:6px;padding:12px 16px;margin:8px 0;font-size:0.84rem;color:#7f1d1d;line-height:1.55}
.h2-rank{font-family:'Source Sans 3',sans-serif;display:flex;align-items:center;gap:12px;padding:10px 14px;margin:5px 0;border:1px solid #e2e8f0;border-radius:6px;background:#fff;font-size:0.84rem;line-height:1.45}
.h2-rn{width:26px;height:26px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0}
.h2-r1{background:#006233;color:#fff}.h2-r2{background:#dcfce7;color:#006233}.h2-r3{background:#e2e8f0;color:#334155}.h2-r4{background:#f1f5f9;color:#94a3b8}
.h2-rank-body{flex:1;min-width:0}.h2-rank-t{font-weight:600;color:#0f172a}.h2-rank-s{font-size:0.76rem;color:#64748b;margin-top:2px}
.h2-route{font-family:'Source Sans 3',sans-serif;display:flex;gap:12px;padding:10px 14px;margin:5px 0;border:1px solid #e2e8f0;border-radius:6px;background:#fff;font-size:0.82rem;line-height:1.5;align-items:flex-start}
.h2-route.best{border-color:#16a34a;background:#f0fdf4}
.h2-route-cost{font-weight:700;font-size:0.88rem;white-space:nowrap;min-width:75px}
.h2-route-body{flex:1;min-width:0}.h2-route-name{font-weight:600;color:#0f172a}.h2-route-det{font-size:0.74rem;color:#64748b;margin-top:2px;word-break:break-word}
.h2-footer{font-family:'Source Sans 3',sans-serif;text-align:center;padding:16px 0 6px 0;margin-top:24px;border-top:1px solid #e2e8f0;font-size:0.7rem;color:#94a3b8;line-height:1.6}
section[data-testid="stSidebar"]>div{background:#003d1f}
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] .stMarkdown{color:#fff !important}
.sb-h{font-family:'Source Sans 3',sans-serif;font-size:1.1rem;font-weight:700;color:#fff}
.sb-p{font-family:'Source Sans 3',sans-serif;font-size:0.72rem;color:rgba(255,255,255,0.5);margin-bottom:4px}
.sb-l{font-family:'Source Sans 3',sans-serif;font-size:0.63rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.4);font-weight:600;margin:14px 0 2px 0}
</style>""", unsafe_allow_html=True)
# ═══════════════════════ SIDEBAR ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-h">H2 Morocco</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-p">Plateforme d\'aide a la decision</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Site de production ───────────────────────────────────────────────
    st.markdown('<div class="sb-l">Site de production</div>', unsafe_allow_html=True)
    site = st.selectbox("Site", list(SITES.keys()), index=0, label_visibility="collapsed")

    with st.expander("Site arbitraire (hors liste)"):
        use_arb = st.checkbox("Utiliser des coordonnées personnalisées")
        if use_arb:
            lat_arb = st.number_input("Latitude (°N)", 21.0, 36.0, 30.0, step=0.1)
            lon_arb = st.number_input("Longitude (°W)", -17.0, -1.0, -8.0, step=0.1)
            nom_arb = st.text_input("Nom du site", "Mon site")
            port_arb = st.number_input("Distance port (km)", 0, 500, 100)
            site_arb = site_arbitraire_vers_dict_app(lat_arb, lon_arb, nom_arb, port_arb)
            SITES[nom_arb] = site_arb
            site = nom_arb
            S = site_arb
            st.markdown(f"""
**Paramètres interpolés (IDW) :**
- GHI : {site_arb['ghi']} kWh/m²/an
- Vent : {site_arb['ws']} m/s
- Eau : {site_arb['eau']} $/m³
- Région proxy : {site_arb['_idw']['region_proxy']} ({site_arb['_idw']['dist_proxy_km']} km)
""")

    st.markdown("---")

    # ── Marché cible — CORRECTION : destination personnalisée possible ───
    st.markdown('<div class="sb-l">Marche cible</div>', unsafe_allow_html=True)
    dests = [k for k in DESTINATIONS if k != site]

    mode_dest = st.radio(
        "Mode destination",
        ["Liste prédéfinie", "Destination personnalisée"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    if mode_dest == "Liste prédéfinie":
        dest = st.selectbox(
            "Destination",
            dests,
            index=dests.index('Rotterdam') if 'Rotterdam' in dests else 0,
            label_visibility="collapsed"
        )
    else:
        with st.expander("Définir une destination personnalisée", expanded=True):
            nom_dest_arb  = st.text_input("Nom de la destination", "Ma destination")
            lat_dest_arb  = st.number_input("Latitude (°N)",  -90.0,  90.0,  51.9, step=0.1)
            lon_dest_arb  = st.number_input("Longitude (°E)", -180.0, 180.0,  4.5, step=0.1)
            pays_dest_arb = st.text_input("Pays", "Pays-Bas")
            type_dest_arb = st.selectbox(
                "Type",
                ["Port H2", "Port", "Marché", "Hub", "Industriel", "Autre"],
                index=0
            )
        # Injection dans DESTINATIONS à la volée
        DESTINATIONS[nom_dest_arb] = {
            'lat':  lat_dest_arb,
            'lon':  lon_dest_arb,
            'pays': pays_dest_arb,
            'type': type_dest_arb,
        }
        dest = nom_dest_arb
        st.markdown(
            f'<div class="h2-ok" style="font-size:0.76rem">'
            f'📍 <b>{nom_dest_arb}</b> — {pays_dest_arb} '
            f'({lat_dest_arb:.2f}°N, {lon_dest_arb:.2f}°E) · {type_dest_arb}'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="sb-l">Parametres</div>', unsafe_allow_html=True)
    annee = st.select_slider("Annee", [2024, 2030, 2035, 2040, 2050], value=2030)
    qh2 = st.number_input("Production (tH2/an)", 100, 500000, 10000)
    mode_demande = st.radio("Mode demande", ["Flexible (suit ENR)", "Constant 24/7 (industriel)"], index=0,
        help="Constant : client industriel necessite H2 en continu")
    scenario_stock = st.selectbox("Scenario eco.", ["optimiste", "central", "pessimiste"], index=1)

    AHP_LABEL_TO_KEY = {
        "EXPORT":          "EXPORT",
        "Pôle_Industriel": "Pôle_Industriel",
        "SITE_ISOLE":      "SITE_ISOLE",
    }
    scenario_ahp_label = st.selectbox(
        "Scenario AHP",
        list(AHP_LABEL_TO_KEY.keys()),
        index=0
    )
    scenario_ahp = AHP_LABEL_TO_KEY[scenario_ahp_label]

    st.markdown("---")
    dev = st.radio("Devise", ["USD", "EUR"], index=0, horizontal=True)
    with st.expander("Avance"):
        use_bat = st.checkbox("Batterie (Etape 2)", True)
        n_mc = st.slider("Tirages Monte Carlo", 500, 5000, 2000)
        if mode_demande.startswith("Constant"):
            buffer_h = st.slider("Buffer H2 (heures)", 4, 72, 24)
        else:
            buffer_h = 0
            # ── Dimensionnement ENR ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sb-l">Dimensionnement ENR</div>', unsafe_allow_html=True)
    
    mode_dim = st.radio(
        "Mode dimensionnement",
        ["Automatique (NSGA-II)", "Manuel"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if mode_dim == "Manuel":
        PV_MW_user   = st.number_input("PV (MW)",           10, 2000, 100, step=10)
        EOL_MW_user  = st.number_input("Éolien (MW)",        0, 2000, 100, step=10)
        ELEC_MW_user = st.number_input("Électrolyseur (MW)", 5,  500,  50, step=5)
        BAT_MWH_user = st.number_input("Batterie (MWh)",     0, 2000, 200, step=50)
            
# ═══════════════════════ CONVERSION DEVISE ═══════════════════════════
# DOIT être placé APRÈS la sélection de 'dev' dans la sidebar
# et AVANT l'affichage des tabs

if 'TAUX_EUR' not in globals():
    TAUX_EUR = 0.92  # Valeur par défaut si non défini dans engine.py

if dev == "EUR":
    fx = float(TAUX_EUR)
    u = "€"
else:
    fx = 1.0
    u = "$"

# Debug optionnel (à supprimer après validation)
# st.sidebar.caption(f"Devise: {dev} | fx={fx} | u={u}")
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════ STABILISATION DU RENDER ═══════════════════════════
# Force un re-render propre quand la devise ou le site change
render_trigger = f"{site}_{dest}_{annee}_{scenario_ahp}_{dev}_{qh2}"
if "last_trigger" not in st.session_state or st.session_state.last_trigger != render_trigger:
    st.session_state.last_trigger = render_trigger
    # Ne pas forcer rerun ici, laisser Streamlit gérer naturellement
# ═════════════════════════════════════════════════════════════════════════

# ═══════════════════════ CALCULS ══════════════════════════════════════════════
S = SITES[site]
dist_cache = load_or_build_distance_cache()
cs = cf_solaire(S['ghi']); ce = cf_eolien(S['ws'])
ls = lcoe_sol(cs,annee); le = lcoe_eol(ce,annee)
ch, we, lh = calc_hybride(ce, cs, le, ls)

lcoh_r = {}
for t, p in ELECS.items():
    cx = capex_lr_unified(t if t in CAPEX_PARAMS else 'PEM', annee)
    lcoh_r[t] = {
        'lcoh': calc_lcoh(lh, ch, cx, p['op'], p['eta'], TURBINE['DR'], p['lt']),
        'capex': cx, 'eta': p['eta'], 'trl': p['trl']
    }
be = min(lcoh_r, key=lambda k: lcoh_r[k]['lcoh'] or 999)
bl = lcoh_r[be]['lcoh'] or 0

qkg = qh2 * 1000
eta_be = lcoh_r[be]['eta']
ch_real = max(ch, 0.01)
# ── Électrolyseur dimensionné sur production cible ──────────────────────────
# ── Dimensionnement ENR : Automatique ou Manuel ──────────────────────────────
if mode_dim == "Manuel":
    PV_MW   = float(PV_MW_user)
    EOL_MW  = float(EOL_MW_user)
    ELEC_MW = float(ELEC_MW_user)
    BAT_MWH = float(BAT_MWH_user) if use_bat else 0.0
else:
    # ── Électrolyseur dimensionné sur production cible ───────────────────────
    # CAPEX électrolyseur dépend de l'année → meilleur CF possible
    cx_be = capex_lr_unified(be if be in CAPEX_PARAMS else 'PEM', annee)
    
    # Facteur de charge cible : on vise 95% de la production demandée
    ELEC_MW = max(5, qkg * (eta_be / 1000) / (ch_real * 8760) * 0.95)
    ELEC_MW_max = qkg / (eta_be / 1000) / 8760 * 1.5   # ← cap réduit (était 2.0)
    ELEC_MW = min(ELEC_MW, ELEC_MW_max)

    # ── Ratios ENR dépendants de l'ANNÉE (courbes d'apprentissage) ───────────
    # Plus l'année avance, plus le système est optimisé → ratio plus bas
    RATIO_BASE = {
        "Dakhla":      2.8,
        "Laayoune":    2.6,
        "Tanger":      2.5,
        "Boujdour":    2.7,
        "Ouarzazate":  2.0,
        "Midelt":      2.0,
        "Guelmim":     2.2,
        "Jorf_Lasfar": 2.0,
        "Agadir":      2.3,
        "Casablanca":  2.2,
        "Nador":       2.4,
        "Marrakech":   2.1,
    }

    # Facteur de réduction par année (systèmes plus efficaces)
    ANNEE_FACTEUR = {
        2024: 1.00,
        2030: 0.95,
        2035: 0.90,
        2040: 0.86,
        2050: 0.80,
    }
    # Interpolation si année intermédiaire
    annees_ref = sorted(ANNEE_FACTEUR.keys())
    if annee in ANNEE_FACTEUR:
        facteur_annee = ANNEE_FACTEUR[annee]
    else:
        # Interpolation linéaire
        for i in range(len(annees_ref) - 1):
            if annees_ref[i] <= annee <= annees_ref[i+1]:
                a, b = annees_ref[i], annees_ref[i+1]
                facteur_annee = (ANNEE_FACTEUR[a] +
                    (annee - a) / (b - a) *
                    (ANNEE_FACTEUR[b] - ANNEE_FACTEUR[a]))
                break
        else:
            facteur_annee = 0.73

    ratio_enr = RATIO_BASE.get(site, 2.0) * facteur_annee
    P_enr_total = ELEC_MW * ratio_enr

    if we > 0.01:
        PV_MW  = max(5, P_enr_total * (1 - we))
        EOL_MW = max(5, P_enr_total * we)
    else:
        PV_MW  = max(5, P_enr_total)
        EOL_MW = 0.0

    BAT_MWH = ELEC_MW * 4.0 if use_bat else 0

# ── Surface requise (coefficients NREL/IEA) ─────────────────────────────────
surface_need_km2 = PV_MW * 0.012 + EOL_MW * 0.06
surface_ok = surface_need_km2 <= S['surface']

is_constant = mode_demande.startswith("Constant")
cal_override = None
if site not in CALIB_PROFILS:
    cal_override = {
        'CF_sol': cf_solaire(S['ghi']),
        'CF_eol': cf_eolien(S['ws'])
    }
if is_constant:
    demande_kg_h = qkg / 8760
    BUFFER_H2_KG = demande_kg_h * buffer_h
    sim = simulate_8760h_constant(site, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, BUFFER_H2_KG, be, demande_kg_h,cal_override=cal_override)
else:
    sim = simulate_8760h(site, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, be,cal_override=cal_override)

lcoh_detail = calc_lcoh_detailed(PV_MW, EOL_MW, ELEC_MW, BAT_MWH, sim['H2_kg_an'], be,annee=annee)
taux_cible = sim['H2_kg_an'] / qkg * 100 if qkg > 0 else 0

if 'h2_profil' in sim and sim['h2_profil'] is not None:
    jours_calcules = jours_stockage_from_profil(sim['h2_profil'])
    source_jours = f"profil 8760h ({jours_calcules}j)"
else:
    jours_calcules = JOURS_STOCKAGE_DEFAULT.get(site, 14)
    source_jours = f"defaut region ({jours_calcules}j)"

stor_results = run_storage_optimizer(site, annee, scenario_stock, qkg, bl, jours_override=jours_calcules,caverne_dispo=S.get('caverne', False))
bs = stor_results[0]['tech'] if stor_results else 'GH2_350bar'
bs_lcos = stor_results[0]['LCOS'] if stor_results else 0.5

if is_constant and BUFFER_H2_KG > 0:
    buffer_capex = 600 * BUFFER_H2_KG
    buffer_ann = buffer_capex * CRF(0.08, 20)
    buffer_cost_per_kg = buffer_ann / max(sim['H2_kg_an'], 1)
    bs_lcos += buffer_cost_per_kg
else:
    buffer_cost_per_kg = 0

ci_enr, ci_grid, ci_ind = get_ci_sources(ch * 100)
co2_data = {}
for tk in [r['tech'] for r in stor_results]:
    e_compr = {
        'GH2_350bar': 3.0,
        'GH2_700bar': 6.0,
        'LH2':        0.5,
        'NH3':        8.0,
        'LOHC':       5.0,
        'Caverne':    1.5,
        'eMethanol':  9.0,
    }.get(tk, 3.0)
    e_synth = {
        'NH3':       8.0,
       'LOHC':      4.0,
       'eMethanol': 6.0,
    }.get(tk, 0.0)
    co2_op = calc_co2_tech(tk, ci_enr, ci_grid, ci_ind, e_compr, e_synth)
    co2_up = CO2_UPSTREAM.get(tk, 1.5); co2_tot = co2_op + co2_up
    zone, _ = co2_zone_label(co2_tot)
    co2_data[tk] = {'op':co2_op,'up':co2_up,'tot':co2_tot,'score':score_co2_zone(co2_tot),'zone':zone}

# ── Compléter co2_data pour toutes les techs (LH2/LOHC absents si pas de caverne) ──
_ALL_TECHS = ['GH2_350bar','GH2_700bar','LH2','NH3','LOHC','Caverne','eMethanol']
for tk in _ALL_TECHS:
    if tk not in co2_data:
        e_compr = {
            'GH2_350bar': 3.0,
            'GH2_700bar': 6.0,
            'LH2':       12.0,
            'NH3':        8.0,   # ← corrigé
            'LOHC':       5.0,
            'Caverne':    1.5,
            'Caverne_saline': 1.5,
            'eMethanol':  9.0,
        }.get(tk, 3.0)
        e_synth = {
            'NH3':       8.0,
            'LOHC':      4.0,
            'eMethanol': 6.0,
        }.get(tk, 0.0)
        co2_op  = calc_co2_tech(tk, ci_enr, ci_grid, ci_ind, e_compr, e_synth)
        co2_up  = CO2_UPSTREAM.get(tk, 1.5)
        co2_tot = co2_op + co2_up
        zone, _ = co2_zone_label(co2_tot)
        co2_data[tk] = {
            'op':co2_op,'up':co2_up,'tot':co2_tot,
            'score':score_co2_zone(co2_tot),'zone':zone
        }
techs_macb = [r['tech'] for r in stor_results]   # ← ligne déjà existante
if len(techs_macb) >= 3:
    AHP_KEY_MAP = {
        'EXPORT':          'EXPORT',
       'Pôle_Industriel': 'Pôle_Industriel',
        'SITE_ISOLE':      'SITE_ISOLE',
    }
    ahp_key = AHP_KEY_MAP.get(scenario_ahp, scenario_ahp)
    w_ahp, cr = ahp_weights(AHP_SCENARIOS[ahp_key])
    w_dict = dict(zip(CRITERIA, w_ahp))
    data_macb = pd.DataFrame(index=techs_macb)
    for r in stor_results:
        tk = r['tech']
        data_macb.loc[tk,'CAPEX'] = storage_capex_lr(
            {"GH2_350bar":"GH2_tank","GH2_700bar":"GH2_tank700","LH2":"LH2_liq",
             "NH3":"NH3_synth","LOHC":"LOHC_sys","Caverne":"Caverne","Caverne_saline":"Caverne","eMethanol":"EMeth_syn"}.get(tk,"GH2_tank"), annee)
        opx_spec, _ = calc_opex_specifique(tk, qkg)
        data_macb.loc[tk,'OPEX'] = data_macb.loc[tk,'CAPEX'] * .02 + opx_spec / qkg
        data_macb.loc[tk,'Densite'] = STORAGE_INFO.get(tk,{}).get('den', 20)
        data_macb.loc[tk,'LCOS'] = r['LCOS']
        data_macb.loc[tk,'TRL'] = STORAGE_INFO.get(tk,{}).get('trl', 7)
        data_macb.loc[tk,'CO2_Real'] = co2_data.get(tk,{}).get('score', 50)
        eau_L = (9 + 0) * (2 if S['eau'] > .8 else 1)
        data_macb.loc[tk,'Eau_Real'] = score_eau_zone(eau_L)
    data_macb = data_macb.fillna(0)
    macb_scores = run_macbeth_full(techs_macb, data_macb, w_dict)
    rob = analyse_robustesse(techs_macb, data_macb, w_dict, n_pert=200)
else:
    macb_scores = {t: 50 for t in techs_macb}; w_dict = {}; cr = 0; rob = {}
routes = find_routes(site, dest, annee, dist_cache)
best_route = routes[0]; lcot_v = best_route['cost']
lcodc = bl + bs_lcos + lcot_v; prix_eu = 4.50; marge = prix_eu - lcodc
# ═══════════════════════ PLOTLY TEMPLATE ══════════════════════════════════════
PL = dict(template='plotly_white', font=dict(family='Source Sans 3, sans-serif', size=12, color='#334155'),
    title_font=dict(size=14, color='#0f172a'), paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=20, t=50, b=40))

# ═══════════════════════ AFFICHAGE ════════════════════════════════════════════
mode_label = "Flexible" if not is_constant else f"Constant 24/7 (buffer {buffer_h}h)"
bs_nom = STORAGE_INFO.get(bs,{}).get('nom', bs)

st.markdown(f"""<div class="h2-hero">
<h2>{site} &#8594; {dest}</h2>
<p>{annee} &middot; {qh2:,} tH2/an &middot; {mode_label}</p>
</div>""", unsafe_allow_html=True)

with st.expander(f"Fiche technique — {site}", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GHI", f"{S['ghi']} kWh/m2"); c1.metric("DNI", f"{S['dni']} kWh/m2")
    c2.metric("Vent", f"{S['ws']} m/s"); c2.metric("Reseau", S['reseau'])
    c3.metric("Eau", f"{S['eau']} $/m3"); c3.metric("Caverne", "Oui" if S['caverne'] else "Non")
    c4.metric("Port", f"{S['port']} km"); c4.metric("Surface", f"{S['surface']:,.0f} km2")

# KPI row — single HTML block, no st.columns overlap
kc_lcodc = "green" if lcodc < prix_eu else "red"
kc_marge = "green" if marge > 0 else "red"
st.markdown(f"""<div class="h2-kpi-row">
<div class="h2-kpi green"><div class="h2-kpi-val">{bl*fx:.2f} {u}</div><div class="h2-kpi-lbl">LCOH ({be})</div></div>
<div class="h2-kpi amber"><div class="h2-kpi-val">{bs_lcos*fx:.2f} {u}</div><div class="h2-kpi-lbl">LCOS ({bs_nom[:10]})</div></div>
<div class="h2-kpi blue"><div class="h2-kpi-val">{lcot_v*fx:.2f} {u}</div><div class="h2-kpi-lbl">LCOT Transport</div></div>
<div class="h2-kpi {kc_lcodc}"><div class="h2-kpi-val">{lcodc*fx:.2f} {u}</div><div class="h2-kpi-lbl">LCODC Total</div></div>
<div class="h2-kpi {kc_marge}"><div class="h2-kpi-val">{marge*fx:+.2f} {u}</div><div class="h2-kpi-lbl">Marge EU /kg</div></div>
</div>""", unsafe_allow_html=True)

if marge > 0:
    co2v = co2_data.get(bs,{}).get('tot', 0); co2z = co2_data.get(bs,{}).get('zone', '')
    st.markdown(f'<div class="h2-ok"><b>Chaine optimale identifiee</b><br>{be} ({lcoh_r[be]["capex"]*fx:.0f} {u}/kW) &#8594; {bs_nom} ({source_jours}) &#8594; {best_route["label"]}<br>LCODC = {lcodc*fx:.2f} {u}/kg &middot; Marge {marge*fx:+.2f} {u}/kg &middot; CO2 = {co2v:.2f} kgCO2/kgH2 ({co2z})</div>', unsafe_allow_html=True)


# ═══════════════════════ ONGLETS ══════════════════════════════════════════════
tabs = st.tabs(["Production", "MACBETH", "Stockage LCOS", "Transport", "Simulation 8760h", "Chaine Complete", "Sensibilite"])

with tabs[0]:
    st.markdown('<div class="h2-sec">Production — Weibull + LCOE + LCOH</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("CF Solaire", f"{cs*100:.1f} %"); c1.metric("LCOE Solaire", f"{(ls or 0)*fx:.4f} {u}/kWh")
    c2.metric("CF Eolien", f"{ce*100:.1f} %"); c2.metric("LCOE Eolien", f"{(le or 0)*fx:.4f} {u}/kWh" if le else "N/A")
    c3.metric("CF Hybride", f"{ch*100:.1f} %"); c3.metric("Mix", f"{we*100:.0f}% eol / {(1-we)*100:.0f}% sol")
    fig = go.Figure()
    for t, r in lcoh_r.items():
        fig.add_trace(go.Bar(x=[t], y=[(r['lcoh'] or 0)*fx], marker_color=ELECS[t]['col'],
            text=f"{(r['lcoh'] or 0)*fx:.2f}<br>CAPEX {(r['capex']*fx):.0f} {u}/kW", textposition='outside'))
    fig.add_hline(y=2*fx, line_dash="dot", line_color="#94a3b8", annotation_text="Ref H2 gris", annotation_font_color="#94a3b8")
    fig.update_layout(**PL, title=f"LCOH par technologie — {site} ({annee})", yaxis_title=f"LCOH ({u}/kg)",yaxis=dict(range=[0, 8]), height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
# ═══════════════════════ ONGLET MACBETH — VERSION GARANTIE ═══════════════════
with tabs[1]:
    # Container avec key unique pour forcer re-render propre
    with st.container(key=f"macbeth_tab_{site}_{scenario_ahp}_{annee}_{dev}"):
        
        st.markdown('<div class="h2-sec">Classement MACBETH — Résultats</div>', unsafe_allow_html=True)


        # ── Poids AHP ────────────────────────────────────────────────────────
        cr_label = "Cohérent ✅" if cr < 0.1 else "Acceptable ⚠️" if cr < 0.15 else "Incohérent ❌"
        st.info(f"📊 Scénario : **{scenario_ahp}** · CR = {cr:.4f} — {cr_label}")
        
        if w_dict:
            cols = st.columns(min(len(w_dict), 4))
            criteria_labels = {'CAPEX':'CAPEX','OPEX':'OPEX','Densite':'Densité','LCOS':'LCOS','TRL':'TRL','CO2_Real':'CO₂','Eau_Real':'Eau'}
            for idx, (crit, weight) in enumerate(w_dict.items()):
                with cols[idx % len(cols)]:
                    st.metric(criteria_labels.get(crit, crit), f"{weight*100:.1f}%")

        # ── AFFICHAGE DIRECT depuis stor_results (FIABLE) ───────────────────
        if not stor_results:
            st.warning("⚠️ Aucune donnée de stockage disponible")
        else:
            st.success(f"✅ {len(stor_results)} technologies analysées")
            
            # Mapping des noms
            tech_map = {
                'GH2_350bar': 'H2 350 bar', 'GH2_700bar': 'H2 700 bar',
                'LH2': 'H2 Liquide', 'NH3': 'Ammoniaque',
                'LOHC': 'LOHC', 'Caverne': 'Caverne Saline', 'eMethanol': 'e-Méthanol'
            }
            
            # 🔹 Construction du tableau AVEC conversion garantie
            rows = []
            for rank, r in enumerate(sorted(stor_results, key=lambda x: x.get('score', 0), reverse=True), 1):
                tk = r.get('tech', 'Inconnu')
                nom = tech_map.get(tk, tk)
                
                # ✅ LCOS : conversion DIRECTE depuis stor_results
                lcos_raw = r.get('LCOS')
                if lcos_raw is not None:
                    lcos_conv = float(lcos_raw) * float(fx)
                    lcos_txt = f"{lcos_conv:.3f} {u}/kg"
                elif tk in ['Caverne', 'Caverne_saline'] and not S.get('caverne', False):
                    lcos_txt = f"⚠️ Non disponible (géologie) ({u}/kg)"
                else:
                    lcos_txt = f"N/D ({u}/kg)"
                
                # CO2
                co2_info = co2_data.get(tk) or {}
                co2_tot = co2_info.get('tot', 0)
                co2_zone = co2_info.get('zone', 'N/D')
                
                # Score MACBETH
                score = macb_scores.get(tk, r.get('score', 0))
                
                rows.append({
                    'Rang': rank,
                    'Technologie': nom,
                    'Score': f"{score:.2f}/100",
                    'LCOS': lcos_txt,  # ✅ CONVERSION GARANTIE ICI
                    'CO₂': f"{co2_tot:.2f} kg/kg ({co2_zone})",
                    'Efficacité': f"{r.get('eff', 0)*100:.1f}%",
                    'Statut': "✅ Oui" if r.get('rentable', False) else "❌ Non"
                })
            
            # Affichage tableau
            df_display = pd.DataFrame(rows)
            
            # Coloration du rang
            def color_rang(val):
                if val == 1: return 'background-color: #dcfce7; color: #006233; font-weight: bold'
                elif val == 2: return 'background-color: #dcfce7; color: #006233'
                elif val == 3: return 'background-color: #f1f5f9'
                return ''
            
            st.dataframe(
                df_display.style.map(color_rang, subset=['Rang']),
                use_container_width=True,
                hide_index=True,
                key=f"macbeth_table_{site}_{scenario_ahp}_{dev}"
            )
            
            # Export CSV
            csv = df_display.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
            st.download_button(
                label="📥 Télécharger (CSV)",
                data=csv,
                file_name=f"MACBETH_{site}_{scenario_ahp}_{annee}_{dev}.csv",
                mime="text/csv",
                key=f"dl_macbeth_{site}_{dev}"
            )

            # Radar chart (optionnel, simplifié)
            if len(stor_results) >= 3 and 'data_macb' in locals() and not data_macb.empty:
                with st.expander("📈 Profil multi-critères", expanded=False):
                    cols_use = [c for c in data_macb.columns if c in ['CAPEX','OPEX','Densite','LCOS','TRL','CO2_Real','Eau_Real']]
                    if cols_use:
                        cats = ['CAPEX','OPEX','Densité','LCOS','TRL','CO₂','Eau'][:len(cols_use)]
                        df_norm = data_macb[cols_use].copy()
                        for col in cols_use:
                            p5, p95 = df_norm[col].quantile(0.05), df_norm[col].quantile(0.95)
                            df_norm[col] = ((df_norm[col].clip(p5, p95) - p5) / (p95 - p5) * 100) if p95 > p5 else 50.0
                        
                        fig_radar = go.Figure()
                        colors = ['#1565C0','#D84315','#2E7D32','#6A1B9A','#00838F','#F9A825','#AD1457']
                        for i, tk in enumerate(stor_results[:5]):
                            tkey = tk.get('tech', '')
                            if tkey not in df_norm.index: continue
                            vals = df_norm.loc[tkey, cols_use].tolist()
                            vals.append(vals[0])  # fermer le polygone
                            fig_radar.add_trace(go.Scatterpolar(
                                r=vals, theta=cats+[cats[0]],
                                name=tech_map.get(tkey, tkey),
                                line=dict(color=colors[i%len(colors)], width=2),
                                fill='toself', opacity=0.15
                            ))
                        fig_radar.update_layout(**PL, height=400, title=f"Profil — {site}")
                        st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{site}_{dev}")
with tabs[2]:  # ← Onglet "Stockage LCOS"
    # ✅ Key simple sans dom_id()
    storage_container = st.container(key=f"storage_tab_{site}_{annee}_{dev}")
    
    with storage_container:
        st.markdown('<div class="h2-sec">LCOS — Jours dynamiques depuis profil 8760h</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="h2-ok"><b>Jours de stockage : {jours_calcules} j</b> — Source : {source_jours}</div>', unsafe_allow_html=True)
        
        if stor_results:
            # ── Graphique LCOS ───────────────────────────────────────────
            fig3 = go.Figure()
            for r in stor_results:
                tk = r['tech']
                info = STORAGE_INFO.get(tk, {})
                rentable_txt = "Rentable" if r.get('rentable', False) else "Non rentable"
                fig3.add_trace(go.Bar(
                    x=[info.get('nom', tk)],
                    y=[r['LCOS'] * fx],
                    marker_color=info.get('col', '#94a3b8'),
                    text=f"{r['LCOS']*fx:.3f}<br>eff={r.get('eff',0):.0%}<br>{rentable_txt}",
                    textposition='outside',
                    name=tk
                ))
            
            jours_affiche = stor_results[0].get('jours', jours_calcules)
            fig3.update_layout(
                **PL,
                title=f"LCOS — {site} ({annee}, {scenario_stock}, {jours_affiche}j)",
                yaxis_title=f"LCOS ({u}/kg)",
                height=430,
                showlegend=False,
                uirevision=f"lcos_{site}_{annee}"
            )
            # ✅ Key simple sans dom_id()
            st.plotly_chart(fig3, use_container_width=True, key=f"lcos_chart_{site}_{dest}_{annee}_{dev}")

            # ── TABLEAU : Conversion + Renommage ─────────────────────────
            cols_display = ['tech', 'LCOS', 'LCOH_total', 'eff', 'co2', 'jours', 'rentable', 'marge']
            cols_available = [c for c in cols_display if c in stor_results[0].keys()]
            df_stor = pd.DataFrame(stor_results)[cols_available].copy()
            
            cols_to_convert = [c for c in ['LCOS', 'LCOH_total', 'marge'] if c in df_stor.columns]
            if cols_to_convert:
                df_stor[cols_to_convert] = (df_stor[cols_to_convert] * fx).round(4)
            
            rename_map = {
                'LCOS': f'LCOS ({u}/kg)',
                'LCOH_total': f'LCOH_total ({u}/kg)',
                'marge': f'marge ({u}/kg)',
                'eff': 'Efficacité (%)',
                'co2': 'CO₂ (kg/kgH₂)',
                'jours': 'Jours stockage',
                'rentable': 'Statut',
                'tech': 'Technologie'
            }
            rename_filtered = {k: v for k, v in rename_map.items() if k in df_stor.columns}
            df_stor = df_stor.rename(columns=rename_filtered)
            
            df_display = df_stor.copy()
            for col in df_display.columns:
                if '/kg' in col or col in ['CAPEX', 'OPEX']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
                elif 'eff' in col.lower() or 'Efficacité' in col:
                    df_display[col] = df_display[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
                elif 'Statut' in col:
                    df_display[col] = df_display[col].apply(lambda x: "✅ Oui" if x else "❌ Non")
            
            # ✅ Key simple sans dom_id()
            table_container = st.container(key=f"table_container_{site}_{annee}_{dev}")
            with table_container:
                st.markdown('<div class="h2-sec">Détails des résultats</div>', unsafe_allow_html=True)
                st.dataframe(df_display, use_container_width=True, hide_index=True, key=f"df_stor_{site}_{annee}_{dev}")
                
                if len(df_display) > 0:
                    csv = df_display.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"LCOS_{site}_{dest}_{annee}_{dev}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{site}_{annee}_{dev}"
                    )

            # ── Monte Carlo ──────────────────────────────────────────────
            # ✅ Key simple sans dom_id()
            mc_container = st.container(key=f"mc_container_{site}_{annee}_{dev}")
            with mc_container:
                st.markdown('<div class="h2-sec">Monte Carlo LCOS</div>', unsafe_allow_html=True)
                mc_results = {}
                for tk in [r['tech'] for r in stor_results[:4]]:
                    mc = run_mc_storage(tk, qkg, n_mc)
                    if mc:
                        mc_results[tk] = mc
                
                if mc_results:
                    figMC = go.Figure()
                    colors_mc = ['#006233', '#d97706', '#1565c0', '#6a1b9a']
                    for idx, (tk, mc) in enumerate(mc_results.items()):
                        nom_tech = STORAGE_INFO.get(tk, {}).get('nom', tk)
                        color = colors_mc[idx % len(colors_mc)]
                        p50 = mc.get('P50', 0) * fx
                        p10 = mc.get('P10', 0) * fx
                        p90 = mc.get('P90', 0) * fx
                        
                        figMC.add_trace(go.Bar(
                            x=[nom_tech],
                            y=[p50],
                            error_y=dict(type='data', symmetric=False,
                                array=[(p90 - p50)], arrayminus=[(p50 - p10)],
                                color=color, thickness=1.5, width=4),
                            marker_color=color, opacity=0.85,
                            text=f"P50={p50:.2f}<br>[{p10:.2f}-{p90:.2f}]<br>CV={mc.get('CV',0)*100:.0f}%",
                            textposition='outside', name=nom_tech
                        ))
                    
                    figMC.update_layout(
                        **PL, title="Analyse de risque LCOS — Intervalles P10/P90",
                        yaxis_title=f"LCOS ({u}/kg)", height=380, showlegend=False,
                        uirevision=f"mc_{site}_{annee}"
                    )
                    # ✅ Key simple sans dom_id()
                    st.plotly_chart(figMC, use_container_width=True, key=f"mc_chart_{site}_{annee}_{dev}")
        else:
            # ✅ Key simple sans dom_id()
            with st.container(key=f"empty_storage_{site}_{annee}_{dev}"):
                st.markdown('<div class="h2-warn">⚠️ Aucun résultat de stockage disponible</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="h2-sec">Itineraires — OSRM + Courbes d\'apprentissage</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(routes)} itineraires** — OSRM/OpenStreetMap — {annee}")
    lats, lons, texts = [], [], []
    for seg in best_route['segs']:
        for n in [seg['de'], seg['a']]:
            nd = SITES.get(n) or DESTINATIONS.get(n)
            if nd: lats.append(nd['lat']); lons.append(nd['lon']); texts.append(n)
    figM = go.Figure()
    mc = ['#006233'] + ['#d97706'] * max(0, len(lats)-2) + ['#c1272d'] if len(lats) > 1 else ['#006233']
    figM.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode='markers+text', text=texts, textposition='top center', marker=dict(size=14, color=mc)))
    for seg in best_route['segs']:
        sl, so = [], []
        for n in [seg['de'], seg['a']]:
            nd = SITES.get(n) or DESTINATIONS.get(n)
            if nd: sl.append(nd['lat']); so.append(nd['lon'])
        figM.add_trace(go.Scattermapbox(lat=sl, lon=so, mode='lines', line=dict(width=3, color='#006233' if seg['type']=='Domestique' else '#c1272d'), name=f"{seg['mode']} ({seg['dist']:.0f} km)"))
    figM.update_layout(mapbox=dict(style='carto-positron', center=dict(lat=np.mean(lats), lon=np.mean(lons)), zoom=3), height=420, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(figM, use_container_width=True)
    for i, r in enumerate(routes[:5]):
        opt = i == 0; cls = "best" if opt else ""
        segs_txt = " &#8594; ".join([f"{s['de']}>{s['a']} ({s['mode']}, {s['dist']:.0f}km, {s['cost']*fx:.2f}{u})" for s in r['segs']])
        tag = " — OPTIMAL" if opt else ""
        st.markdown(f'<div class="h2-route {cls}"><div class="h2-route-cost" style="color:{"#006233" if opt else "#334155"}">{r["cost"]*fx:.2f} {u}/kg</div><div class="h2-route-body"><div class="h2-route-name">{r["label"]}{tag}</div><div class="h2-route-det">{r["dist"]:.0f} km &middot; {segs_txt}</div></div></div>', unsafe_allow_html=True)

with tabs[4]:
    st.markdown(f'<div class="h2-sec">Simulation 8760h — Mode {"Constant 24/7" if is_constant else "Flexible"}</div>', unsafe_allow_html=True)
    if is_constant:
        st.markdown(f'<div class="h2-ok"><b>Mode industriel</b> — Demande {sim.get("demande_kg_h",0):.0f} kg/h = {qh2:,} tH2/an — Buffer {buffer_h}h ({BUFFER_H2_KG:,.0f} kg)</div>', unsafe_allow_html=True)
    if taux_cible < 80:
        st.markdown(f'<div class="h2-warn"><b>Production = {sim["H2_kg_an"]/1e6:.2f} kt/an ({taux_cible:.0f}% cible)</b><br>CF insuffisant. Pistes : meilleur site, reduire cible, ou reseau.</div>', unsafe_allow_html=True)
    if not surface_ok:
        st.markdown(f'<div class="h2-err"><b>Surface requise : {surface_need_km2:.3f} km2 — Disponible : {S["surface"]:.3f} km2</b></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PV", f"{PV_MW:.1f} MW"); c2.metric("Eolien", f"{EOL_MW:.1f} MW")
    c3.metric("Electrolyseur", f"{ELEC_MW:.1f} MW ({be})"); c4.metric("Batterie", f"{BAT_MWH:.0f} MWh" if use_bat else "Sans")
    c1, c2, c3, c4 = st.columns(4)
    if is_constant:
        c1.metric("H2 livre", f"{sim['H2_kg_an']/1e6:.2f} kt/an", delta=f"{sim.get('taux_couverture',0)*100:.0f}% couverture")
        c2.metric("Taux service", f"{sim.get('taux_service',0)*100:.1f}%")
        c3.metric("Buffer H2", f"{BUFFER_H2_KG:,.0f} kg")
        c4.metric("H2 perdu", f"{sim.get('H2_perdu_kg',0)/1000:.1f} t")
    else:
        ecart = taux_cible - 100
        signe = "+" if ecart > 0 else ""
        c1.metric(
            "H2 produit",
            f"{sim['H2_kg_an']/1e6:.2f} kt/an",
            delta=f"{signe}{ecart:.0f}% vs cible",
            delta_color="normal" if -20 <= ecart <= 20 else "inverse"
        )
        c2.metric("Fiabilite", f"{sim['fiabilite']*100:.1f}%")
        c3.metric("Curtailment", f"{sim['E_curtail_MWh']/1000:.1f} GWh")
        c4.metric("Heures pleine charge", f"{sim['h_full_load']}/8760")
    # SURFACE : 3 decimales pour ne jamais afficher 0
    surf_st = "OK" if surface_ok else "Insuffisant"
    st.metric("Surface ENR", f"{surface_need_km2:.2f} km2", delta=f"{surf_st} ({S['surface']:,.0f} km2 dispo)")
    st.metric("Jours stockage", f"{jours_calcules} j", delta=f"vs {JOURS_STOCKAGE_DEFAULT.get(site,14)}j defaut")
    if lcoh_detail:
        measures = ["relative"]*5+["total"]
        labels = ["PV","Eolien","Electrolyseur","Batterie","Eau","LCOH Total"]
        values = [lcoh_detail['PV']*fx, lcoh_detail['Eolien']*fx, lcoh_detail['Electrolyseur']*fx, lcoh_detail['Batterie']*fx, lcoh_detail['Eau']*fx, 0]
        if is_constant and buffer_cost_per_kg > 0:
            measures = ["relative"]*6+["total"]
            labels = ["PV","Eolien","Electrolyseur","Batterie","Eau","Buffer H2","LCOH Total"]
            values = [lcoh_detail['PV']*fx, lcoh_detail['Eolien']*fx, lcoh_detail['Electrolyseur']*fx, lcoh_detail['Batterie']*fx, lcoh_detail['Eau']*fx, buffer_cost_per_kg*fx, 0]
        figD = go.Figure(go.Waterfall(orientation="v", measure=measures, x=labels, y=values,
            text=[f"{v:.3f}" for v in values[:-1]]+[f"{sum(values[:-1]):.3f}"], textposition="outside",
            increasing={"marker":{"color":"#006233"}}, totals={"marker":{"color":"#c1272d"}}))
        figD.update_layout(**PL, title=f"Decomposition LCOH ({mode_label})", yaxis_title=f"{u}/kgH2", height=400)
        st.plotly_chart(figD, use_container_width=True)

with tabs[5]:
    st.markdown('<div class="h2-sec">LCODC — Chaine de valeur complete</div>', unsafe_allow_html=True)
    lv, sv, tv, dv = bl*fx, bs_lcos*fx, lcot_v*fx, lcodc*fx
    fW = go.Figure(go.Waterfall(orientation="v", measure=["relative","relative","relative","total"],
        x=[f"LCOH\n({be})", f"LCOS\n({bs_nom[:10]})", f"LCOT\n({best_route['segs'][0]['mode'][:12]})", "LCODC"],
        y=[lv, sv, tv, 0], text=[f"{lv:.2f}", f"{sv:.2f}", f"{tv:.2f}", f"{dv:.2f}"], textposition="outside",
        increasing={"marker":{"color":"#006233"}}, totals={"marker":{"color":"#c1272d" if lcodc > prix_eu else "#006233"}}))
    fW.add_hline(y=prix_eu*fx, line_dash="dash", line_color="#c1272d", annotation_text=f"Seuil EU {prix_eu*fx:.1f}", annotation_font_color="#c1272d")
    fW.update_layout(**PL, height=420)
    st.plotly_chart(fW, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fP = go.Figure(go.Pie(labels=['Production','Stockage','Transport'], values=[lv, sv, tv],
            marker_colors=['#006233','#d97706','#1565c0'], hole=.5, textinfo='label+percent'))
        fP.update_layout(**PL, height=320)
        st.plotly_chart(fP, use_container_width=True)
    with c2:
        co2 = co2_data.get(bs, {})
        st.markdown(f"""
| Composante | Technologie | Cout |
|:---|:---|:---|
| Production | {be} | {lv:.2f} {u}/kg |
| Stockage | {bs_nom} | {sv:.2f} {u}/kg |
| Transport | {best_route['label'][:25]} | {tv:.2f} {u}/kg |
| **TOTAL** | | **{dv:.2f} {u}/kg** |

**CO2 :** {co2.get('tot',0):.2f} kgCO2/kgH2 ({co2.get('zone','')}) | **Stockage :** {jours_calcules}j ({source_jours}) | **Mode :** {mode_label}
""")

with tabs[6]:
    st.markdown('<div class="h2-sec">Comparaison 12 sites + Projection</div>', unsafe_allow_html=True)
    rows = []
    for sk, sv_s in SITES.items():
        c_s = cf_solaire(sv_s['ghi']); c_e = cf_eolien(sv_s['ws'])
        c_h, _, l_h = calc_hybride(c_e, c_s, lcoe_eol(c_e,annee), lcoe_sol(c_s,annee))
        p = ELECS[be]; cx = capex_lr(p['c24'], p['l1'], p['l2'], p['f'], annee)
        lh2 = calc_lcoh(l_h, c_h, cx, p['op'], p['eta'], TURBINE['DR'], p['lt']) or 0
        rows.append({'Site':sk, 'GHI':sv_s['ghi'], 'Vent':sv_s['ws'], 'CF_hyb%':round(c_h*100,1), f'LCOH_{be}':round(lh2*fx,2), 'Port_km':sv_s['port']})
    df_c = pd.DataFrame(rows).sort_values(f'LCOH_{be}')
    colors = [('#006233' if r['Site'] == site else '#cbd5e1') for _, r in df_c.iterrows()]
    figC = go.Figure(go.Bar(x=df_c['Site'], y=df_c[f'LCOH_{be}'], marker_color=colors,
        text=[f"{v:.2f}" for v in df_c[f'LCOH_{be}']], textposition='outside'))
    figC.add_hline(y=2*fx, line_dash="dot", line_color="#94a3b8", annotation_text="Ref H2 gris", annotation_font_color="#94a3b8")
    figC.update_layout(**PL, title=f"LCOH {be} — 12 sites ({annee})", yaxis_title=f"LCOH ({u}/kg)", height=400, showlegend=False)
    st.plotly_chart(figC, use_container_width=True)
    st.dataframe(df_c.set_index('Site'), use_container_width=True)

    yrs = [2024, 2030, 2035, 2040, 2050]; dL = {'LCOH':[], 'LCOS':[], 'LCOT':[]}
    for yr in yrs:
        p = ELECS[be]; cx = capex_lr(p['c24'], p['l1'], p['l2'], p['f'], yr)
        dL['LCOH'].append((calc_lcoh(lh, ch, cx, p['op'], p['eta'], TURBINE['DR'], p['lt']) or 0)*fx)
        sr = run_storage_optimizer(site, yr, scenario_stock, qkg, dL['LCOH'][-1]/fx, jours_override=jours_calcules)
        dL['LCOS'].append(sr[0]['LCOS']*fx if sr else .5)
        dL['LCOT'].append(best_route['cost']*lcot_year_factor(best_route['segs'][0]['mode'], yr)*fx)
    tots = [a+b+c for a, b, c in zip(dL['LCOH'], dL['LCOS'], dL['LCOT'])]
    f8 = go.Figure()
    f8.add_trace(go.Bar(x=yrs, y=dL['LCOH'], name='LCOH', marker_color='#006233'))
    f8.add_trace(go.Bar(x=yrs, y=dL['LCOS'], name='LCOS', marker_color='#d97706'))
    f8.add_trace(go.Bar(x=yrs, y=dL['LCOT'], name='LCOT', marker_color='#1565c0'))
    f8.add_trace(go.Scatter(x=yrs, y=tots, mode='lines+markers+text', name='LCODC',
        line=dict(color='#c1272d', width=2.5), marker=dict(size=8),
        text=[f"{t:.2f}" for t in tots], textposition='top center', textfont=dict(size=11)))
    f8.add_hline(y=prix_eu*fx, line_dash="dash", line_color="#c1272d", annotation_text="Seuil EU", annotation_font_color="#c1272d")
    f8.update_layout(**PL, barmode='stack', title="Projection LCODC 2024-2050", yaxis_title=f"{u}/kg", height=430)
    st.plotly_chart(f8, use_container_width=True)
    for i, yr in enumerate(yrs):
        if tots[i] <= prix_eu*fx:
            st.markdown(f'<div class="h2-ok"><b>Competitif des {yr}</b> — LCODC = {tots[i]:.2f} {u}/kg</div>', unsafe_allow_html=True); break
    else:
        st.markdown(f'<div class="h2-warn"><b>Non competitif avant 2050</b></div>', unsafe_allow_html=True)

st.markdown('<div class="h2-footer">H2 Morocco &middot; Corrections A4/A6/A7 &middot; Modules : base_de_donnees, Etape1_macbeth, modele_stockage, etape2, Etape4_transport<br>Moujane Nisrine, Larhni Maroua &middot; Enc. Meryeme Azaroual</div>', unsafe_allow_html=True)