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

# Configuración
BATCH_SIZE = 1500
RAW_DB_URI = os.getenv("RAW_DB_CONN")
CLEAN_DB_URI = os.getenv("CLEAN_DB_CONN")  
API_URL = os.getenv("DB_GET_DATA", "http://fastapi:8989/data")
API_URL_reset = os.getenv("DB_FORMAT_DATA", "http://fastapi:8989/reset")
SCHEMA_RAW = "raw_data"
SCHEMA_CLEAN = "clean_data"
TABLE_NAME = "raw_data_init"
TABLE_NAME_CLEAN = "clean_data_init"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
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
        """Crea el esquema raw si no existe."""
        engine = create_engine(RAW_DB_URI)
        ddl = f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_RAW};"
        with engine.begin() as conn:
            conn.execute(text(ddl))

    def create_table_raw():
        """Crea la tabla raw si no existe."""
        engine = create_engine(RAW_DB_URI)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA_RAW}.{TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            price FLOAT,
            brokered_by TEXT,
            status TEXT,
            bed INTEGER,
            bath INTEGER,
            acre_lot FLOAT,
            street TEXT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            house_size FLOAT,
            prev_sold_date DATE,
            load_date TIMESTAMP WITHOUT TIME ZONE
        );
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))

    def load_raw_batch():
        """Carga un batch de datos desde la API al esquema raw."""
        # Obtener datos de la API
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        records = response.json()
        
        if not records:
            raise AirflowSkipException("No hay datos nuevos para cargar")

        # Convertir a DataFrame y agregar timestamp
        df = pd.DataFrame(records)
        df["load_date"] = datetime.utcnow()

        # Cargar a la base de datos
        engine = create_engine(RAW_DB_URI)
        raw_conn = engine.raw_connection()
        try:
            cur = raw_conn.cursor()
            cols = list(df.columns)
            insert_sql = f"""
                INSERT INTO {SCHEMA_RAW}.{TABLE_NAME} ({','.join(cols)})
                VALUES %s
            """
            values = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
            execute_values(cur, insert_sql, values, page_size=BATCH_SIZE)
            raw_conn.commit()
        finally:
            cur.close()
            raw_conn.close()

    def create_schema_clean():
        """Crea el esquema clean si no existe."""
        engine = create_engine(CLEAN_DB_URI)
        ddl = f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_CLEAN};"
        with engine.begin() as conn:
            conn.execute(text(ddl))

    def create_table_clean():
        """Crea la tabla clean si no existe."""
        engine = create_engine(CLEAN_DB_URI)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA_CLEAN}.{TABLE_NAME_CLEAN} (
            id SERIAL PRIMARY KEY,
            price FLOAT,
            brokered_by TEXT,
            status TEXT,
            bed INTEGER,
            bath INTEGER,
            acre_lot FLOAT,
            street TEXT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            house_size FLOAT,
            prev_sold_date DATE,
            split TEXT,
            load_date TIMESTAMP WITHOUT TIME ZONE
        );
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))

    def transform_and_load_clean():
        """Transforma los datos raw y los carga en la tabla clean."""
        # Leer datos raw
        engine_r = create_engine(RAW_DB_URI)
        raw_conn = engine_r.raw_connection()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME} WHERE load_date >= NOW() - INTERVAL '1 hour'",
                con=raw_conn
            )
        finally:
            raw_conn.close()

        if df.empty:
            raise AirflowSkipException("No hay datos nuevos para procesar")

        # Limpiar datos
        df = df.dropna()
        
        # Convertir tipos de datos
        df["bed"] = df["bed"].astype(int)
        df["bath"] = df["bath"].astype(int)
        df["acre_lot"] = df["acre_lot"].astype(float)
        df["house_size"] = df["house_size"].astype(float)
        df["price"] = df["price"].astype(float)
        
        # Dividir en train/test
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        df_train["split"] = "train"
        df_test["split"] = "test"
        df = pd.concat([df_train, df_test])

        # Cargar a clean
        engine_c = create_engine(CLEAN_DB_URI)
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

    # Definir tareas
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

    # Definir dependencias
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 
