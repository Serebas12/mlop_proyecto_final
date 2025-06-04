import os
from datetime import datetime, timedelta
import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from airflow.exceptions import AirflowSkipException
from sklearn.model_selection import train_test_split

BATCH_SIZE = 1500
RAW_DB_URI = os.getenv("RAW_DB_CONN")
CLEAN_DB_URI = os.getenv("CLEAN_DB_CONN")  
API_URL = "http://10.43.101.108:80/data?group_number=6&day=Wednesday"
API_URL_reset = "http://10.43.101.108:80/restart_data_generation?group_number=6&day=Wednesday"
SCHEMA_RAW     = "raw_data"
SCHEMA_CLEAN   = "clean_data"
TABLE_NAME = "raw_data_init"
TABLE_NAME_CLEAN = "clean_data_init"

default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    dag_id="data_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 5, 1),
    schedule_interval="@hourly",
    catchup=False,
    tags=["raw","ingestion","dataSource"],
) as dag:

    def create_schema_raw():
        """Crea los esquema si no existe."""
        engine = create_engine(RAW_DB_URI)
        ddl = f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_RAW};"
        with engine.begin() as conn:
            conn.execute(text(ddl))

    def create_table_raw():
        """Crea tabla si no existen."""
        engine = create_engine(RAW_DB_URI)
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {SCHEMA_RAW};
        CREATE TABLE IF NOT EXISTS {SCHEMA_RAW}.{TABLE_NAME} (
            brokered_by     NUMERIC,
            status          VARCHAR(50),
            price           NUMERIC,
            bed             NUMERIC,
            bath            NUMERIC,
            acre_lot        NUMERIC,
            street          VARCHAR(200),
            city            VARCHAR(100),
            state           VARCHAR(100),
            zip_code        VARCHAR(20),
            house_size      NUMERIC,
            prev_sold_date  DATE,
            load_date       TIMESTAMP NOT NULL
        );
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))
    
    def load_raw_batch():
        """
        1) Llama a la API y obtiene el JSON.
        2) Construye un DataFrame solo con payload["data"].
        3) Abre una conexión psycopg2 vía engine.raw_connection()
           y vuela el DataFrame en batch con execute_values.
        """
        # Traer datos de la API (timeout 5 min)
        try:
             resp = requests.get(API_URL, timeout=300)
             resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
             if e.response is not None and e.response.status_code == 400:
                 # fin normal: no más datos que recolectar
                 raise AirflowSkipException("API devolvió 400 – ya no hay datos nuevos")
             else:
                 raise
        payload = resp.json()

        records = payload.get("data", [])
        if not records:
            return

        df = pd.DataFrame(records)
        df["load_date"] = datetime.utcnow()

        engine = create_engine(RAW_DB_URI)
        raw_conn = engine.raw_connection()
        try:
            cur = raw_conn.cursor()
            cols = list(df.columns)
            insert_sql = f"""
                INSERT INTO {SCHEMA_RAW}.{TABLE_NAME} ({','.join(cols)})
                VALUES %s
            """
            values = [
                tuple(row) for row in df[cols].itertuples(index=False, name=None)
            ]
            execute_values(cur, insert_sql, values, page_size=BATCH_SIZE)
            raw_conn.commit()
        finally:
            cur.close()
            raw_conn.close()

    def create_schema_clean():
        engine = create_engine(CLEAN_DB_URI)
        ddl = f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_CLEAN};"
        with engine.begin() as conn:
            conn.execute(text(ddl))
    
    def create_table_clean():
        """Crea la tabla clean_data.clean_data_init si no existe."""
        engine = create_engine(CLEAN_DB_URI)
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {SCHEMA_CLEAN};
        CREATE TABLE IF NOT EXISTS {SCHEMA_CLEAN}.{TABLE_NAME_CLEAN} (
            brokered_by     NUMERIC,
            status          VARCHAR(50),
            price           NUMERIC,
            bed             NUMERIC,
            bath            NUMERIC,
            acre_lot        NUMERIC,
            street          VARCHAR(200),
            city            VARCHAR(100),
            state          VARCHAR(100),
            zip_code        VARCHAR(20),
            house_size      NUMERIC,
            prev_sold_date  DATE,
            load_date       TIMESTAMP NOT NULL,
            split           VARCHAR(10) NOT NULL
        );
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))

    def transform_and_load_clean():
        """Lee raw, filtra por load_date nuevo, limpia, particiona y carga en clean_data."""
        engine_c = create_engine(CLEAN_DB_URI)
        # 1) última fecha procesada en clean_data
        with engine_c.connect() as conn_c:
            result = conn_c.execute(text(f"""
                SELECT MAX(load_date) AS maxd
                  FROM {SCHEMA_CLEAN}.{TABLE_NAME_CLEAN}
            """))
            max_date = result.scalar()

        engine_r = create_engine(RAW_DB_URI)
        # 2) leo sólo raws nuevos, usando raw_connection para pandas
        raw_conn = engine_r.raw_connection()
        try:
            if max_date:
                df = pd.read_sql_query(
                    f"SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME} WHERE load_date > %s",
                    con=raw_conn,
                    params=[max_date],
                )
            else:
                df = pd.read_sql_query(
                    f"SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME}",
                    con=raw_conn,
                )
        finally:
            raw_conn.close()

        if df.empty:
            return

        # 3) limpieza avanzada
        for c in ["street", "city", "state", "status"]:
            df[c] = df[c].astype(str).str.strip().str.lower()
        df["zip_code"] = df["zip_code"].astype(str).str.replace(r"\D+", "", regex=True)
        df = df[
            (df["price"] >= 0) &
            (df["house_size"] >= 0) &
            (df["acre_lot"] >= 0) &
            (pd.to_datetime(df["prev_sold_date"], errors="coerce") <= df["load_date"])
        ]

        # 4) partición train/test
        df["split"] = "train"
        _, test_idx = train_test_split(df.index, test_size=0.3, random_state=42)
        df.loc[test_idx, "split"] = "test"

        # 5) vuelco en batches al clean_data
        conn_c = engine_c.raw_connection()
        try:
            cur = conn_c.cursor()
            cols = list(df.columns)
            insert_sql = f"""
                INSERT INTO {SCHEMA_CLEAN}.{TABLE_NAME_CLEAN}
                ({','.join(cols)}) VALUES %s
            """
            values = [tuple(r) for r in df[cols].itertuples(index=False, name=None)]
            execute_values(cur, insert_sql, values, page_size=BATCH_SIZE)
            conn_c.commit()
        finally:
            cur.close()
            conn_c.close()

    t1 = PythonOperator(
        task_id="create_schema_raw",
        python_callable=create_schema_raw,
    )

    t2 = PythonOperator(
        task_id="create_table_raw",
        python_callable=create_table_raw,
    )

    t3 = PythonOperator(
        task_id="load_raw_batch",
        python_callable=load_raw_batch,
        execution_timeout=timedelta(minutes=5),
    )

    t4 = PythonOperator(
        task_id="create_schema_clean",
        python_callable=create_schema_clean,
    )
    
    t5 = PythonOperator(
        task_id="create_table_clean",
        python_callable=create_table_clean,
    )

    t6 = PythonOperator(
        task_id="transform_and_load_clean",
        python_callable=transform_and_load_clean,
        execution_timeout=timedelta(minutes=5),
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 
