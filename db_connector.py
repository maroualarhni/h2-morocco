# -*- coding: utf-8 -*-
"""
H2 Morocco — Connecteur Unifié (db_connector.py)
═════════════════════════════════════════════════
Fait le lien entre TOUS vos modules et PostgreSQL.
Fonctionne aussi en mode CSV si PostgreSQL est indisponible.

Usage:
  python db_connector.py --load          # Charge les CSV dans PostgreSQL
  python db_connector.py --calculs       # Lance tous les calculs -> stocke résultats
  python db_connector.py --check         # Vérifie l'état de la base
  python db_connector.py --all           # load + calculs
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Forcer UTF-8 sur stdout/stderr (Windows PowerShell utilise charmap par défaut)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# Rendre le handler robuste contre les erreurs d'encodage Windows
for handler in logging.root.handlers:
    if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
        try:
            handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(handler, 'setFormatter'):
        handler.errors = 'replace'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (même que vos fichiers existants)
# ─────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/h2morocco_db")
SCHEMA  = "h2morocco"

BASE_DIR   = Path.home() / "Downloads"
CSV_CLEAN  = BASE_DIR / "h2pipeline" / "outputs" / "clean_csv"
CSV_RAW    = BASE_DIR / "H2Morocco222_Outputs" / "csv"
OUTPUT_DIR = BASE_DIR / "h2pipeline" / "outputs"

ANNEES = [2024, 2030, 2035, 2040, 2050]
TAUX_EUR = 0.9217  # 1 USD = 0.9217 EUR

# ─────────────────────────────────────────────────────────────────────────────
# PRIX ÉLECTRICITÉ PAR RÉGION (manquant dans MODELESTOCKAGE.py !)
# Basé sur tarifs ONEE HT option MU, prix moyen pondéré HP/HPL/HC en EUR/kWh
# ─────────────────────────────────────────────────────────────────────────────
PRIX_ELEC_REGION = {
    "Dakhla":          0.065,   # Site EnR optimal -> autoconso + PPA bas
    "Laayoune":        0.068,
    "Guelmim":         0.072,
    "Souss_Massa":     0.075,
    "Marrakech_Safi":  0.078,
    "Casablanca":      0.085,   # Zone industrielle -> tarif réseau plus élevé
    "Draa_Tafilalet":  0.070,
    "Beni_Mellal":     0.078,
    "Fes_Meknes":      0.080,
    "Rabat_Sale":      0.082,
    "Oriental":        0.076,
    "Tanger":          0.079,
    # Anciens noms (compatibilité MODELESTOCKAGE)
    "Ouarzazate":      0.070,
    "Jorf_Lasfar":     0.080,
    "_default":        0.078,
}


# ═════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

class H2Database:
    """
    Interface unifiée vers la base H2 Morocco.
    Mode PostgreSQL si disponible, sinon fallback CSV.
    """

    def __init__(self, db_url=DB_URL, schema=SCHEMA):
        self.schema = schema
        self.engine = None
        self.mode = "csv"

        try:
            from sqlalchemy import create_engine, text
            self.engine = create_engine(db_url)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                conn.commit()
            self.mode = "postgresql"
            log.info(f"[OK] Connecté à PostgreSQL — schéma '{schema}'")
        except Exception as e:
            log.warning(f"[WARN] PostgreSQL non disponible ({e})")
            log.info("   -> Mode CSV activé")

    # ─── LECTURE ──────────────────────────────────────────────────────────

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Lit une table depuis PostgreSQL ou CSV (fallback)."""
        if self.mode == "postgresql":
            try:
                return pd.read_sql_table(table_name, self.engine, schema=self.schema)
            except Exception:
                pass

        # Fallback CSV
        for csv_dir in [CSV_CLEAN, CSV_RAW]:
            if not csv_dir.exists():
                continue
            for pattern in [f"{table_name}_clean.csv", f"{table_name}.csv",
                            f"{table_name.upper()}.csv"]:
                fpath = csv_dir / pattern
                if fpath.exists():
                    return pd.read_csv(fpath, encoding="utf-8-sig")

        return pd.DataFrame()

    def get_t1(self):  return self.get_table("t1_ressources_energetiques")
    def get_t2(self):  return self.get_table("t2_technologies_production")
    def get_t3(self):  return self.get_table("t3_technologies_stockage")
    def get_t4_corridors(self): return self.get_table("t4_corridors_resume")
    def get_t4_segments(self): return self.get_table("t4_segments_detail")
    def get_t9(self):  return self.get_table("t9_scenarios_temporels")

    # ─── ÉCRITURE ─────────────────────────────────────────────────────────

    def save_results(self, df: pd.DataFrame, table_name: str):
        """Sauvegarde les résultats en CSV + PostgreSQL."""
        # CSV toujours
        csv_dir = OUTPUT_DIR / "resultats"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info(f"  [SAVE] CSV: {csv_path} ({len(df)} lignes)")

        # PostgreSQL si disponible
        if self.mode == "postgresql":
            try:
                df.to_sql(table_name, self.engine, schema=self.schema,
                          if_exists="replace", index=False)
                log.info(f"  [SAVE] PostgreSQL: {self.schema}.{table_name}")
            except Exception as e:
                log.error(f"  [ERR] PostgreSQL {table_name}: {e}")

    # ─── CHARGEMENT CSV -> PostgreSQL ──────────────────────────────────────

    def load_all_csv(self):
        """Charge tous les CSV nettoyés dans PostgreSQL."""
        if self.mode != "postgresql":
            log.error("PostgreSQL non disponible")
            return

        loaded = 0
        for csv_dir in [CSV_CLEAN, CSV_RAW]:
            if not csv_dir.exists():
                continue
            for fpath in sorted(csv_dir.glob("*.csv")):
                tname = fpath.stem.lower().replace("_clean", "")
                try:
                    df = pd.read_csv(fpath, encoding="utf-8-sig")
                    df.to_sql(tname, self.engine, schema=self.schema,
                              if_exists="replace", index=False)
                    loaded += 1
                    log.info(f"  [OK] {tname}: {len(df)} lignes")
                except Exception as e:
                    log.error(f"  [ERR] {tname}: {e}")

        log.info(f"\n[OK] {loaded} tables chargées dans {self.schema}")

    # ─── VÉRIFICATION ─────────────────────────────────────────────────────

    def check_status(self):
        """Affiche l'état complet de la base."""
        tables = [
            "t1_ressources_energetiques", "t2_technologies_production",
            "t3_technologies_stockage", "t4_corridors_resume",
            "t4_segments_detail", "t9_scenarios_temporels",
            "resultats_lcoh", "resultats_lcos", "resultats_transport",
        ]
        print(f"\n{'═'*60}")
        print(f"  H2 MOROCCO — Mode: {self.mode.upper()}")
        print(f"{'═'*60}")
        for t in tables:
            df = self.get_table(t)
            icon = "[OK]" if not df.empty else "[ERR]"
            statut = f"{len(df):>6} lignes" if not df.empty else "  vide — lancez --calculs"
            print(f"  {icon} {t:<40} {statut}")
        print(f"{'═'*60}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CALCULS -> RÉSULTATS EN BASE
# ═════════════════════════════════════════════════════════════════════════════

def run_all_calculs(db: H2Database):
    """Lance TOUS les moteurs de calcul et stocke les résultats."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Forcer UTF-8 sur stdout avant tout import de module externe ───────────
    # Les modules ETAPE2, MODELESTOCKAGE, ETAPE4TRANSPORT contiennent des print()
    # avec emojis qui plantent sur Windows si l'encodage n'est pas UTF-8.
    import io
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    else:
        # Python < 3.7 — wrapper manuel
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # ── Ajouter le dossier parent au path pour imports ────────────────
    project_dir = Path(__file__).parent
    sys.path.insert(0, str(project_dir))

    # ─── 1. LCOH Production ───────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("[>>>] ÉTAPE 1 — LCOH Production (via ETAPE2)")
    log.info("="*60)

    try:
        # Import depuis votre ETAPE2.py
        import importlib
        etape2 = importlib.import_module("ETAPE2")
        log.info("  [OK] ETAPE2 importé")

        t1 = db.get_t1()
        rows_lcoh = []

        # Détecter colonnes
        col_region = next((c for c in t1.columns if c.lower() in ["region", "région"]), None)
        col_ghi = next((c for c in t1.columns if "ghi" in c.lower()), None)
        col_vent = next((c for c in t1.columns if "vent" in c.lower()), None)

        if col_region and col_ghi:
            for _, row in t1.iterrows():
                region = row[col_region]
                ghi = float(row[col_ghi]) if pd.notna(row[col_ghi]) else 2000
                vent = float(row[col_vent]) if col_vent and pd.notna(row.get(col_vent)) else 3.0

                # Utiliser charger_profils_T10 de ETAPE2
                for annee in ANNEES:
                    try:
                        profils = etape2.charger_profils_T10(region, annee, force_synthetic=True)
                        cf_pv = profils['CF_PV_h'].mean()
                        cf_eol = profils['CF_eol_h'].mean()

                        for tech_elec in ["PEM", "AEL", "SOEC"]:
                            params = etape2.PARAMS_TECHNO
                            capex = params.get(f"CAPEX_{tech_elec}", 900)
                            opex_frac = params.get(f"OPEX_{tech_elec}", 0.03)
                            eff = params.get(f"EFF_{tech_elec}", 55)
                            dr = etape2.PARAMS_FIN['DR']
                            lt = etape2.PARAMS_FIN['LT_ELEC']

                            # LCOE hybride simplifié
                            lcoe_pv = (550 * 0.08 + 12) / (cf_pv * 8760) if cf_pv > 0.01 else 999
                            lcoe_eol = (1100 * 0.08 + 35) / (cf_eol * 8760) if cf_eol > 0.05 else 999

                            viable_eol = cf_eol >= 0.20
                            if viable_eol:
                                w_eol = 0.4
                                lcoe_hyb = w_eol * lcoe_eol + (1 - w_eol) * lcoe_pv
                                cf_hyb = w_eol * cf_eol + (1 - w_eol) * cf_pv
                            else:
                                w_eol = 0.0
                                lcoe_hyb = lcoe_pv
                                cf_hyb = cf_pv

                            # LCOH
                            crf = (dr * (1 + dr)**lt) / ((1 + dr)**lt - 1)
                            lcoh = eff * lcoe_hyb + (capex * crf + capex * opex_frac) / (cf_hyb * 8760) if cf_hyb > 0 else 999

                            rows_lcoh.append({
                                "region": region, "annee": annee, "electrolyseur": tech_elec,
                                "ghi": ghi, "vent_100m": vent,
                                "cf_pv": round(cf_pv, 4), "cf_eol": round(cf_eol, 4),
                                "cf_hybride": round(cf_hyb, 4), "w_eolien": round(w_eol, 3),
                                "lcoe_hybride": round(lcoe_hyb, 5),
                                "lcoh_usd_kg": round(lcoh, 3),
                                "lcoh_eur_kg": round(lcoh * TAUX_EUR, 3),
                                "_timestamp": timestamp,
                            })
                    except Exception as e:
                        log.warning(f"    [WARN] {region}/{annee}: {e}")

        df_lcoh = pd.DataFrame(rows_lcoh)
        if not df_lcoh.empty:
            db.save_results(df_lcoh, "resultats_lcoh")
            log.info(f"  [OK] LCOH: {len(df_lcoh)} résultats sauvegardés")

    except Exception as e:
        log.error(f"  [ERR] ETAPE2 échoué: {e}")
        df_lcoh = pd.DataFrame()

    # ─── 2. LCOS Stockage ─────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("[>>>] ÉTAPE 2 — LCOS Stockage (via MODELESTOCKAGE)")
    log.info("="*60)

    try:
        stockage = importlib.import_module("MODELESTOCKAGE")

        # INJECTION du PRIX_ELEC_REGION manquant !
        if not hasattr(stockage, 'PRIX_ELEC_REGION'):
            stockage.PRIX_ELEC_REGION = PRIX_ELEC_REGION
            log.info("  [FIX] PRIX_ELEC_REGION injecté dans MODELESTOCKAGE")

        log.info("  [OK] MODELESTOCKAGE importé")

        rows_lcos = []
        regions_stockage = [r for r in stockage.JOURS_STOCKAGE_DEFAULT if r != "_default"]

        for region in regions_stockage:
            for annee in ANNEES:
                for scenario in ["optimiste", "central", "pessimiste"]:
                    # Meilleur LCOH pour cette région/année
                    best_lcoh = 3.5  # default
                    if not df_lcoh.empty:
                        mask = (df_lcoh["region"].str.contains(region, case=False, na=False)) & \
                               (df_lcoh["annee"] == annee)
                        if mask.any():
                            best_lcoh = df_lcoh.loc[mask, "lcoh_eur_kg"].min()

                    try:
                        model = stockage.StorageOptimizer(region, annee, scenario)

                        # Profil 8760h dynamique
                        profil = stockage.generer_profil_synthetique(region, 10_000_000)
                        model.set_profil_8760h(profil)

                        df_s = model.run_all(10_000_000, best_lcoh)
                        df_s["_timestamp"] = timestamp
                        rows_lcos.append(df_s)
                    except Exception as e:
                        log.warning(f"    [WARN] LCOS {region}/{annee}/{scenario}: {e}")

        if rows_lcos:
            df_lcos = pd.concat(rows_lcos, ignore_index=True)
            db.save_results(df_lcos, "resultats_lcos")
            log.info(f"  [OK] LCOS: {len(df_lcos)} résultats sauvegardés")

    except Exception as e:
        log.error(f"  [ERR] MODELESTOCKAGE échoué: {e}")

    # ─── 3. MACBETH Classement (ETAPE1) ───────────────────────────────
    log.info("\n" + "="*60)
    log.info("[>>>] ÉTAPE 3 — Classement MACBETH (via ETAPE1MACBETH)")
    log.info("="*60)

    try:
        macbeth = importlib.import_module("ETAPE1MACBETH")
        log.info("  [OK] ETAPE1MACBETH importé")
        # Le module s'exécute en connexion directe à PostgreSQL
        # Les résultats sont déjà dans la base
        log.info("  -> MACBETH lit directement depuis PostgreSQL")
    except Exception as e:
        log.warning(f"  [WARN] ETAPE1MACBETH: {e}")

    # ─── 4. Transport MILP (ETAPE4) ──────────────────────────────────
    log.info("\n" + "="*60)
    log.info("[>>>] ÉTAPE 4 — Transport MILP (via ETAPE4TRANSPORT)")
    log.info("="*60)

    try:
        transport = importlib.import_module("ETAPE4TRANSPORT")
        log.info("  [OK] ETAPE4TRANSPORT importé")

        rows_transport = []

        # Cas 1 — Exportateur National
        try:
            mp1 = transport.cas1_exportateur_national_mp(scenario="central")
            r1 = mp1.resultats
            for t, rp in r1.get("par_periode", {}).items():
                rows_transport.append({
                    "cas": "Cas1_Exportateur",
                    "scenario": "central",
                    "periode": t,
                    "statut": r1.get("status"),
                    "demande_ktH2": rp.get("demande_ktH2"),
                    "livraison_ktH2": rp.get("livraison_ktH2"),
                    "LCOT_EUR_kg": rp.get("LCOT_EUR_kg"),
                    "capex_MUSD": rp.get("capex_arcs_MUSD", 0) + rp.get("capex_hubs_MUSD", 0),
                    "VAN_totale_Mrd_EUR": round(r1.get("VAN_totale_EUR", 0) / 1e9, 3),
                    "_timestamp": timestamp,
                })
            log.info(f"  [OK] Cas 1 : {r1.get('status')}")
        except Exception as e:
            log.warning(f"  [WARN] Cas 1 : {e}")

        # Cas 2 — Hub Industriel OCP
        try:
            mp2 = transport.cas2_hub_industriel_mp(scenario="central")
            r2 = mp2.resultats
            for t, rp in r2.get("par_periode", {}).items():
                rows_transport.append({
                    "cas": "Cas2_Hub_OCP",
                    "scenario": "central",
                    "periode": t,
                    "statut": r2.get("status"),
                    "demande_ktH2": rp.get("demande_ktH2"),
                    "livraison_ktH2": rp.get("livraison_ktH2"),
                    "LCOT_EUR_kg": rp.get("LCOT_EUR_kg"),
                    "capex_MUSD": rp.get("capex_arcs_MUSD", 0) + rp.get("capex_hubs_MUSD", 0),
                    "VAN_totale_Mrd_EUR": round(r2.get("VAN_totale_EUR", 0) / 1e9, 3),
                    "_timestamp": timestamp,
                })
            log.info(f"  [OK] Cas 2 : {r2.get('status')}")
        except Exception as e:
            log.warning(f"  [WARN] Cas 2 : {e}")

        # Cas 3 — Site Isolé Midelt
        try:
            mp3 = transport.cas3_site_isole_mp(scenario="central")
            r3 = mp3.resultats
            for t, rp in r3.get("par_periode", {}).items():
                rows_transport.append({
                    "cas": "Cas3_Site_Isole",
                    "scenario": "central",
                    "periode": t,
                    "statut": r3.get("status"),
                    "demande_ktH2": rp.get("demande_ktH2"),
                    "livraison_ktH2": rp.get("livraison_ktH2"),
                    "LCOT_EUR_kg": rp.get("LCOT_EUR_kg"),
                    "capex_MUSD": rp.get("capex_arcs_MUSD", 0) + rp.get("capex_hubs_MUSD", 0),
                    "VAN_totale_Mrd_EUR": round(r3.get("VAN_totale_EUR", 0) / 1e9, 3),
                    "_timestamp": timestamp,
                })
            log.info(f"  [OK] Cas 3 : {r3.get('status')}")
        except Exception as e:
            log.warning(f"  [WARN] Cas 3 : {e}")

        if rows_transport:
            df_transport = pd.DataFrame(rows_transport)
            db.save_results(df_transport, "resultats_transport")
            log.info(f"  [OK] Transport : {len(df_transport)} lignes sauvegardées en base")

    except Exception as e:
        log.warning(f"  [WARN] ETAPE4TRANSPORT: {e}")

    log.info("\n" + "="*60)
    log.info("[OK] TOUS LES CALCULS TERMINÉS")
    log.info("="*60)


# ═════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="H2 Morocco — Connecteur Base de Données",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python db_connector.py --check       -> Vérifier l'état des tables\n"
            "  python db_connector.py --load        -> Charger les CSV dans PostgreSQL\n"
            "  python db_connector.py --calculs     -> Lancer ETAPE2 + STOCKAGE + MILP\n"
            "  python db_connector.py --all         -> load + calculs (pipeline complet)\n"
        )
    )
    parser.add_argument("--load",    action="store_true", help="Charger CSV -> PostgreSQL")
    parser.add_argument("--calculs", action="store_true", help="Lancer tous les calculs et sauvegarder en base")
    parser.add_argument("--check",   action="store_true", help="Vérifier l'état de la base")
    parser.add_argument("--all",     action="store_true", help="load + calculs (pipeline complet)")
    args = parser.parse_args()

    db = H2Database()

    # Par défaut (aucun argument) -> check + aide
    if not any([args.load, args.calculs, args.check, args.all]):
        db.check_status()
        print("  [TIP] Les tables 'resultats_*' sont vides ?")
        print("     -> Lancez :  python db_connector.py --calculs")
        print("     -> Ou tout : python db_connector.py --all\n")
        return

    if args.check:
        db.check_status()

    if args.load or args.all:
        db.load_all_csv()

    if args.calculs or args.all:
        run_all_calculs(db)
        # Afficher l'état final après calculs
        print()
        db.check_status()


if __name__ == "__main__":
    main()
    