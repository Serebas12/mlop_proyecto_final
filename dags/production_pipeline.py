import os
from datetime import datetime, timedelta

import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.exceptions import AirflowSkipException
from sqlalchemy import create_engine, text
from mlflow.tracking import MlflowClient

# Configuración
CLEAN_DB_URI = os.getenv("CLEAN_DB_CONN")
MLFLOW_MODEL = "my_model"
FASTAPI_HOOK = "http://fastapi:8989/hooks/model_update"
RAW_SCHEMA = "raw_data"
RAW_TABLE = "inference_logs"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="production_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 5, 1),
    schedule_interval="@hourly",
    catchup=False,
    tags=["production"],
) as dag:

    def ensure_rawdata_table_fn():
        """Crea la tabla de logs de inferencia si no existe."""
        engine = create_engine(CLEAN_DB_URI)
        with engine.begin() as conn:
            # Crear esquema
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA};"))
            # Crear tabla
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {RAW_SCHEMA}.{RAW_TABLE} (
                    id             SERIAL PRIMARY KEY,
                    requested_at   TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
                    model_name     TEXT    NOT NULL,
                    model_version  INTEGER NOT NULL,
                    run_id         TEXT    NOT NULL,
                    input_data     JSONB   NOT NULL,
                    prediction     JSONB   NOT NULL
                );
            """))

    ensure_rawdata_table = PythonOperator(
        task_id="ensure_rawdata_table",
        python_callable=ensure_rawdata_table_fn,
    )

    wait_for_history = ExternalTaskSensor(
        task_id="wait_for_model_history",
        external_dag_id="modeling_pipeline",
        external_task_id="record_model_history",
        mode="reschedule",
        poke_interval=30,
        timeout=10*60,
    )

    def notify_fastapi_fn():
        """Notifica a FastAPI sobre el nuevo modelo en producción."""
        client = MlflowClient()
        # Obtener la última versión en Production
        prod_versions = client.get_latest_versions(MLFLOW_MODEL, stages=["Production"])
        if not prod_versions:
            raise AirflowSkipException("No hay modelo en Production")

        latest = prod_versions[0]
        model_name = latest.name
        model_version = int(latest.version)
        run_id = latest.run_id

        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id
        }
        resp = requests.post(FASTAPI_HOOK, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"✅ Notified FastAPI of new production model: {payload}")

    notify_fastapi = PythonOperator(
        task_id="notify_fastapi",
        python_callable=notify_fastapi_fn,
    )

    # Definir dependencias
    ensure_rawdata_table >> wait_for_history >> notify_fastapi 