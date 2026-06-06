# -*- coding: utf-8 -*-


"""
Chargement PostgreSQL — H2 Morocco Database
Usage : python load_to_postgres.py
"""
from sqlalchemy import create_engine, text
import pandas as pd
from pathlib import Path
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — À ADAPTER SELON VOTRE ENVIRONNEMENT
# ═══════════════════════════════════════════════════════════════════

# URL de connexion PostgreSQL
# Format : postgresql+psycopg2://user:password@host:port/database
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/h2morocco_db")

# Dossier contenant les CSV nettoyés générés par pipeline_final.py
CLEAN_DIR = Path.home() / "Downloads" / "h2pipeline" / "outputs" / "clean_csv"

# Fallback : si le dossier principal n'existe pas, chercher alternatives
CLEAN_DIR_ALT = Path.home() / "Downloads" / "H2Morocco222_Outputs" / "csv"

# Nom du schéma PostgreSQL où charger les tables
SCHEMA = "h2morocco"

# Mapping table → colonne clé primaire (pour UPSERT optionnel)
PK_MAPPING = {
    "dim_region": "region_id",
    "dim_technologie": "techno_id",
    "t1_ressources": "region",
    "t2_technologies_production": "t2_id",
    "t3_technologies_stockage": "t3_id",
    "t4_corridors_resume": "id",
    "t6a_demande_nationale": "id",
    "t6b_benchmark_competiteurs": "id",
    "t9_scenarios_long": "scenario_id",
}


def create_schema_if_not_exists(engine):
    """Crée le schéma PostgreSQL s'il n'existe pas déjà."""
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        conn.commit()
    logger.info(f"✓ Schéma '{SCHEMA}' vérifié/créé")


def get_csv_dir():
    """Retourne le dossier CSV existant (principal ou fallback)."""
    if CLEAN_DIR.exists() and any(CLEAN_DIR.glob("*.csv")):
        return CLEAN_DIR
    if CLEAN_DIR_ALT.exists() and any(CLEAN_DIR_ALT.glob("*.csv")):
        logger.warning(f"  ⚠️ Dossier principal absent, fallback vers : {CLEAN_DIR_ALT}")
        return CLEAN_DIR_ALT
    return None


def load_table(engine, table_name: str, df: pd.DataFrame):
    """
    Charge un DataFrame dans PostgreSQL via pandas.to_sql().
    
    Mode utilisé : if_exists='replace' (simple, recrée la table)
    Pour un mode UPSERT (idempotent), utiliser execute_values + ON CONFLICT.
    """
    # Préparation : convertir NaN pandas → None pour compatibilité PostgreSQL
    df_clean = df.where(pd.notna(df), None)
    
    try:
        df_clean.to_sql(
            name=table_name,
            con=engine,
            schema=SCHEMA,
            if_exists="replace",      # 'replace' = supprime et recrée
            index=False,
            method="multi",           # INSERT groupés pour performance
            chunksize=500             # Taille des batches
        )
        logger.info(f"✓ {SCHEMA}.{table_name} : {len(df_clean)} lignes chargées")
        return True
    except Exception as e:
        logger.error(f"✗ Erreur chargement {table_name} : {e}")
        return False


def main():
    """Fonction principale de chargement."""
    logger.info("🚀 Démarrage du chargement PostgreSQL")
    
    # 1. Créer le moteur de connexion SQLAlchemy
    try:
        engine = create_engine(DB_URL, echo=False)
        # Test de connexion rapide
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✓ Connexion PostgreSQL établie")
    except Exception as e:
        logger.error(f"✗ Échec connexion : {e}")
        logger.error("Vérifiez : user/password, base existante, PostgreSQL en écoute")
        return
    
    # 2. Créer le schéma si nécessaire
    create_schema_if_not_exists(engine)
    
    # 3. Trouver le dossier CSV
    csv_dir = get_csv_dir()
    if csv_dir is None:
        logger.error(f"✗ Aucun dossier CSV trouvé :")
        logger.error(f"  - Principal : {CLEAN_DIR}")
        logger.error(f"  - Fallback  : {CLEAN_DIR_ALT}")
        logger.error("  Exécutez d'abord pipeline_final.py ou base_de_donnees.py")
        return

    logger.info(f" Dossier CSV : {csv_dir}")
    
    # 4. Charger chaque fichier CSV
    loaded_count = 0
    error_count = 0
    total_rows = 0
    
    for csv_file in sorted(csv_dir.glob("*.csv")):
        # Extraire le nom de table depuis le nom de fichier
        # Ex: "t1_ressources_clean.csv" → "t1_ressources"
        table_name = csv_file.stem.replace("_clean", "")
        
        try:
            # Lecture du CSV avec encodage UTF-8-SIG (compatible Excel/Windows)
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
            
            # Chargement dans PostgreSQL
            if load_table(engine, table_name, df):
                loaded_count += 1
                total_rows += len(df)
            else:
                error_count += 1
                
        except Exception as e:
            logger.error(f"✗ Erreur lecture {csv_file.name} : {e}")
            error_count += 1
    
    # 4. Résumé final
    logger.info("="*50)
    logger.info(f"Chargement terminé : {loaded_count} tables OK, {error_count} erreurs, {total_rows:,} lignes totales")
    logger.info(f"Schéma : {SCHEMA} dans la base h2morocco_db")
    
    # 5. Vérification : lister les tables chargées
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                f"SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = '{SCHEMA}' ORDER BY table_name"
            ))
            tables = [row[0] for row in result]
            logger.info(f" Tables dans {SCHEMA} : {', '.join(tables)}")
    except Exception as e:
        logger.warning(f"  Vérification échouée : {e}")
    
    # 5. Fermer la connexion
    engine.dispose()
    logger.info(" Connexion fermée")


if __name__ == "__main__":
    main()