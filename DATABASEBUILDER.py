# -*- coding: utf-8 -*-
"""

Fonction : Connexion + Création des tables manquantes + Test
"""

import pandas as pd
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION DE LA CONNEXION
# ─────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/h2morocco_db")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FONCTION POUR CRÉER LES TABLES MANQUANTES
# ─────────────────────────────────────────────────────────────────────────────
def create_missing_tables(engine):
    """Crée les tables T5, T7, T8 et T10 si elles n'existent pas."""
    print("\n🔧 Vérification et création des tables manquantes...")
    
    tables_sql = {
        't5_parametres_economiques': """
            CREATE TABLE IF NOT EXISTS h2morocco.t5_parametres_economiques (
                id SERIAL PRIMARY KEY,
                parametre VARCHAR(80),
                valeur NUMERIC,
                unite VARCHAR(30),
                source VARCHAR(200),
                devise_originale VARCHAR(10),
                valeur_eur NUMERIC,
                devise_ref VARCHAR(10),
                annee_ref INTEGER,
                source_taux VARCHAR(100)
            );
        """,
        't7a_emissions_co2': """
            CREATE TABLE IF NOT EXISTS h2morocco.t7a_emissions_co2 (
                id SERIAL PRIMARY KEY,
                filiere VARCHAR(60),
                emissions_kgco2_kgh2_min NUMERIC,
                emissions_kgco2_kgh2_mode NUMERIC,
                emissions_kgco2_kgh2_max NUMERIC,
                conforme_eu_rfnbo BOOLEAN,
                certifiable_certifhy BOOLEAN,
                source VARCHAR(100)
            );
        """,
        't7b_certifications': """
            CREATE TABLE IF NOT EXISTS h2morocco.t7b_certifications (
                id SERIAL PRIMARY KEY,
                certification VARCHAR(50),
                seuil_kgco2_kgh2 NUMERIC,
                seuil_gco2_mj NUMERIC,
                premium_prix_pct NUMERIC,
                marche_cible VARCHAR(100),
                importance_maroc VARCHAR(100),
                source VARCHAR(200)
            );
        """,
        't8_projets_reference_maroc': """
            CREATE TABLE IF NOT EXISTS h2morocco.t8_projets_reference_maroc (
                id SERIAL PRIMARY KEY,
                projet VARCHAR(100),
                type VARCHAR(60),
                capacite_mw NUMERIC,
                capex_total_musd NUMERIC,
                lcoe_ou_lcoh NUMERIC,
                unite_lcoe_lcoh VARCHAR(50),
                cf_reel_pct NUMERIC,
                annee_commission INTEGER,
                statut VARCHAR(50),
                developpeur VARCHAR(80),
                source VARCHAR(200)
            );
        """,
        't10_profils_horaires': """
            CREATE TABLE IF NOT EXISTS h2morocco.t10_profils_horaires (
                id SERIAL PRIMARY KEY,
                region VARCHAR(50) NOT NULL,
                annee INTEGER NOT NULL,
                datetime_utc TIMESTAMP NOT NULL,
                heure_annee INTEGER,
                cf_pv_h REAL,
                cf_eol_h REAL,
                ghi_w_m2 REAL,
                ws90m_m_s REAL,
                t2m_c REAL,
                UNIQUE(region, annee, heure_annee)
            );
            CREATE INDEX IF NOT EXISTS idx_t10_region ON h2morocco.t10_profils_horaires(region);
            CREATE INDEX IF NOT EXISTS idx_t10_datetime ON h2morocco.t10_profils_horaires(datetime_utc);
        """
    }
    
    with engine.connect() as conn:
        for table_name, sql in tables_sql.items():
            try:
                # Exécution de la requête SQL
                conn.execute(text(sql))
                conn.commit()
                print(f"   ✅ Table '{table_name}' vérifiée/créée avec succès.")
            except Exception as e:
                print(f"   ️ Erreur lors de la création de '{table_name}': {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXÉCUTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
try:
    # Création du moteur de connexion
    engine = create_engine(DB_URL)
    print("✅ Connexion à PostgreSQL établie avec succès !\n")

    # APPEL DE LA FONCTION DE CRÉATION DES TABLES
    create_missing_tables(engine)

    # 4. Tester la table T1 (Ressources énergétiques)
    print("\n Chargement des ressources énergétiques (T1)...")
    df_ressources = pd.read_sql(
        "SELECT region, cf_hybride_pct, lcoe_hybride_eur_kwh FROM h2morocco.t1_ressources ORDER BY region", 
        engine
    )
    print(df_ressources.to_string(index=False))
    print(f"   → {len(df_ressources)} régions chargées.\n")

    # 5. Tester la table T9 (Scénarios 2030)
    print("📈 Chargement des scénarios pour l'année 2030 (T9)...")
    df_scenarios = pd.read_sql(
        "SELECT annee, variable, valeur, valeur_eur FROM h2morocco.t9_scenarios_long WHERE annee=2030 LIMIT 10", 
        engine
    )
    print(df_scenarios.to_string(index=False))
    print(f"   → {len(df_scenarios)} lignes affichées.\n")

    # 6. Vérifier que les nouvelles tables existent
    print("🔍 Vérification des tables créées...")
    new_tables = ['t5_parametres_economiques', 't7a_emissions_co2', 't7b_certifications',
                  't8_projets_reference_maroc', 't10_profils_horaires']
    for t in new_tables:
        try:
            count = pd.read_sql(f"SELECT COUNT(*) FROM h2morocco.{t}", engine).iloc[0, 0]
            print(f"    ✅ {t}: {count} lignes")
        except Exception as e:
            print(f"    ⚠️ {t}: table existe mais vide ou erreur ({e})")

    print("\n Tout fonctionne ! Votre base est complète et prête pour l'analyse.")

except Exception as e:
    print(f" Erreur de connexion ou d'exécution : {e}")
    print("\nVérifications à faire :")
    print("  1. Le service PostgreSQL est-il démarré ?")
    print("  2. Le mot de passe 'root' est-il correct dans l'URL ?")
    print("  3. La base 'h2morocco_db' existe-t-elle ?")