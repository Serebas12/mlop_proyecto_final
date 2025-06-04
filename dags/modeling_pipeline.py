import os
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.exceptions import AirflowSkipException
from sqlalchemy import create_engine, text
import mlflow
from mlflow.tracking import MlflowClient
from mlflow import sklearn as mlflow_sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap

# ─── Configuración ─────────────────────────────────────────────────────────────
CLEAN_DB_URI    = os.getenv("CLEAN_DB_CONN")
SCHEMA_CLEAN    = "clean_data"
TABLE_CLEAN     = "clean_data_init"
EXPERIMENT_NAME = "modeling_pipeline"
SHARED_TMP      = "/opt/airflow/dags/tmp"
MAX_SHAP        = 50000

os.makedirs(SHARED_TMP, exist_ok=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ─── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="modeling_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 5, 1),
    schedule_interval="@hourly",
    # schedule_interval="@once",
    # schedule_interval="*/5 * * * *", # 5 minutos
    catchup=False,
    tags=["modeling","mlflow"],
) as dag:

    wait_for_clean = ExternalTaskSensor(
        task_id="wait_for_clean_data",
        external_dag_id="data_pipeline",
        external_task_id="transform_and_load_clean",
        mode="reschedule",
        poke_interval=30,
        timeout=5*60,
    )

    # 1) Creamos la tabla con shap_uri
    def ensure_history_table_fn():
        engine = create_engine(CLEAN_DB_URI)
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_CLEAN}.model_history (
                    id             SERIAL PRIMARY KEY,
                    experiment_id  BIGINT    NOT NULL,
                    model_name     TEXT       NOT NULL,
                    run_id         TEXT       NOT NULL,
                    run_name       TEXT      NOT NULL,
                    model_version  INTEGER    NOT NULL,
                    new_mse        DOUBLE PRECISION,
                    new_rmse       DOUBLE PRECISION,
                    new_mae        DOUBLE PRECISION,
                    new_r2         DOUBLE PRECISION,
                    prod_mse       DOUBLE PRECISION,
                    prod_rmse      DOUBLE PRECISION,
                    prod_mae       DOUBLE PRECISION,
                    prod_r2        DOUBLE PRECISION,
                    promoted       BOOLEAN    NOT NULL,
                    shap_uri       TEXT,
                    trained_at     TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
                );
            """))

    ensure_history_table = PythonOperator(
        task_id="ensure_history_table",
        python_callable=ensure_history_table_fn,
    )

    # 2) Extracción de datos
    def extract_data_fn():
        engine   = create_engine(CLEAN_DB_URI)
        raw_conn = engine.raw_connection()
        try:
            df = pd.read_sql_query(f"SELECT * FROM {SCHEMA_CLEAN}.{TABLE_CLEAN}", con=raw_conn)
        finally:
            raw_conn.close()

        if df.empty:
            raise AirflowSkipException("No hay datos en clean_data_init para extraer")

        df_train = df[df["split"]=="train"].drop(columns=["split"])
        df_test  = df[df["split"]=="test"].drop(columns=["split"])
        df_train.to_csv(f"{SHARED_TMP}/train.csv", index=False)
        df_test .to_csv(f"{SHARED_TMP}/test.csv",  index=False)
        print(f"✅ Extracted {len(df)} rows → {len(df_train)} train / {len(df_test)} test")

    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data_fn,
    )

    # 3) Entrenamiento y logging
    def train_and_log_fn(ti):
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run() as run:
            df_train = pd.read_csv(f"{SHARED_TMP}/train.csv")
            FEATURES = [
                "brokered_by", "status", "bed", "bath", "acre_lot",
                "street", "city", "state", "zip_code", "house_size", "prev_sold_date"
            ]
            for c in ["street", "city", "state", "status", "zip_code"]:
                if c in df_train.columns:
                    df_train[c] = df_train[c].astype(str)
            print("Dtypes en X_train antes de entrenar:")
            print(df_train.dtypes)
            print("Ejemplo de valores de street:", df_train['street'].head(10))
            y_train = df_train.pop("price")
            # Filtrar solo las columnas válidas
            X_train = df_train[[col for col in FEATURES if col in df_train.columns]].copy()
            # Validación de columnas faltantes
            missing = [c for c in FEATURES if c not in X_train.columns]
            if missing:
                print(f"⚠️  Warning: faltan columnas esperadas en los datos: {missing}")
            # Guardar features usadas en MLflow
            import json
            with open(f"{SHARED_TMP}/features.json", "w") as f:
                json.dump(FEATURES, f)
            mlflow.log_artifact(f"{SHARED_TMP}/features.json", artifact_path="features")
            # Revisa NaN
            assert not X_train.isnull().any().any(), "Hay NaN en X_train"
            assert not y_train.isnull().any(), "Hay NaN en y_train"
            # Definición de columnas categóricas y numéricas
            cats = X_train.select_dtypes(include=["object","category"]).columns.tolist()
            low_card = [c for c in cats if df_train[c].nunique()<=8]
            high_card= [c for c in cats if df_train[c].nunique()>8]
            if high_card:
                print(f"Descartando cat cols alta cardinalidad: {high_card}")
            ti.xcom_push("high_card", high_card)
            X_train = X_train.drop(columns=high_card)
            num_cols = [c for c in X_train.columns if c not in low_card]
            print(f"X_train shape: {X_train.shape}, num_cols: {num_cols}, low_card: {low_card}")
            preproc = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), low_card),
                ("num", StandardScaler(), num_cols),
            ], remainder="drop")
            X_train_trans = preproc.fit_transform(X_train)
            assert X_train_trans.shape[0] == X_train.shape[0]
            pipe = Pipeline([
                ("preproc", preproc),
                ("reg", GammaRegressor(max_iter=200))
            ])
            pipe.fit(X_train, y_train)
            final_features = list(X_train.columns)
            with open(f"{SHARED_TMP}/final_features.json", "w") as f:
                json.dump(final_features, f)
            mlflow.log_artifact(f"{SHARED_TMP}/final_features.json", artifact_path="features")
            mlflow_sklearn.log_model(pipe, "model", registered_model_name="my_model")
            df_test = pd.read_csv(f"{SHARED_TMP}/test.csv")
            for c in ["street", "city", "state", "status", "zip_code"]:
                if c in df_test.columns:
                    df_test[c] = df_test[c].astype(str)
            X_test = df_test[[col for col in FEATURES if col in df_test.columns]].copy()
            y_test = df_test["price"]
            X_test = X_test.drop(columns=high_card, errors="ignore")
            preds = pipe.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            run_name = run.data.tags.get("mlflow.runName", None)
            ti.xcom_push("run_name", run_name)
            for k, v in {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}.items():
                mlflow.log_metric(k, v)

    train_and_log = PythonOperator(
        task_id="train_and_log",
        python_callable=train_and_log_fn,
    )

    # 4) Evaluación y promoción
    def evaluate_and_promote_fn(ti):
        client = MlflowClient()
        exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
        run    = client.search_runs([exp.experiment_id], order_by=["attributes.start_time desc"], max_results=1)[0]

        mv = int(client.search_model_versions(f"name='my_model' and run_id='{run.info.run_id}'")[0].version)

        metrics = {k: run.data.metrics[k] for k in ["mse","rmse","mae","r2"]}
        df_train = pd.read_csv(f"{SHARED_TMP}/train.csv")
        high_card = [c for c in df_train.select_dtypes(include=["object","category"]).columns if df_train[c].nunique()>8]

        prod = client.get_latest_versions("my_model", stages=["Production"])
        if prod:
            prod_pipe = mlflow.sklearn.load_model(prod[0].source)
            df_test  = pd.read_csv(f"{SHARED_TMP}/test.csv")
            df_test  = df_test.drop(columns=["price"]+high_card, errors="ignore")
            y_test   = pd.read_csv(f"{SHARED_TMP}/test.csv")["price"]
            p2       = prod_pipe.predict(df_test)
            prod_metrics = {
                "prod_mse":  mean_squared_error(y_test, p2),
                "prod_rmse": None,  # lo calculas igual que arriba
                "prod_mae":  mean_absolute_error(y_test, p2),
                "prod_r2":   r2_score(y_test, p2)
            }
        else:
            prod_metrics = {"prod_mse":None,"prod_rmse":None,"prod_mae":float("inf"),"prod_r2":float("-inf")}

        promoted = metrics["mae"] < prod_metrics["prod_mae"]
        if promoted:
            client.transition_model_version_stage(
                name="my_model",
                version=mv,
                stage="Production",
                archive_existing_versions=True
            )

        promoted = int(promoted)
        ti.xcom_push("cmp", {**metrics, **prod_metrics, "exp_id": exp.experiment_id,
                             "run_id": run.info.run_id, "mv": mv, "promoted": promoted})

    evaluate_and_promote = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote_fn,
    )

    # 5) Computar SHAP y registrar URI
    def compute_shap_fn(ti):
        # 1) Tiramos del diccionario completo que guardó evaluate_and_promote
        cmp = ti.xcom_pull(task_ids="evaluate_and_promote", key="cmp")
        if not cmp:
            raise AirflowSkipException("No se encontró información de métricas (cmp) para compute_shap")

        # 2) Tiramos de la lista de columnas de alta cardinalidad que guardó train_and_log
        high_card = ti.xcom_pull(task_ids="train_and_log", key="high_card") or []

        run_id = cmp["run_id"]
        pipeline = mlflow_sklearn.load_model(f"runs:/{run_id}/model")
        preproc  = pipeline.named_steps["preproc"]
        reg      = pipeline.named_steps["reg"]

        df_test = pd.read_csv(f"{SHARED_TMP}/test.csv")
        X_test  = df_test.drop(columns=["price"] + high_card, errors="ignore")
        if len(X_test) > MAX_SHAP:
            X_test = X_test.sample(n=MAX_SHAP, random_state=42)
        X_trans = preproc.transform(X_test)

        explainer = shap.LinearExplainer(reg, X_trans, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_trans)
        cols = preproc.get_feature_names_out()
        shap_df = pd.DataFrame(shap_vals, columns=cols)

        path = f"{SHARED_TMP}/shap_values.parquet"
        shap_df.to_parquet(path, index=False)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(path, artifact_path="shap_values")
            shap_uri = mlflow.get_artifact_uri("shap_values/shap_values.parquet")
        ti.xcom_push(key="shap_uri", value=shap_uri)
        print("✅ SHAP generado y URI registrada:", shap_uri)

    compute_shap = PythonOperator(
        task_id="compute_shap",
        python_callable=compute_shap_fn,
        execution_timeout=timedelta(minutes=10),
    )

    # 6) Finalmente: insertar todo en model_history, incluyendo shap_uri
    def record_history_fn(ti):
        # 1) Pull de XComs con task_ids y key
        cmp = ti.xcom_pull(task_ids="evaluate_and_promote", key="cmp")
        shap_uri = ti.xcom_pull(task_ids="compute_shap", key="shap_uri")
        run_name = ti.xcom_pull(task_ids="train_and_log", key="run_name")

        # 2) Validaciones
        if cmp is None:
            raise AirflowSkipException("No se encontró información de métricas en XCom (cmp).")
        if shap_uri is None:
            raise AirflowSkipException("No se encontró shap_uri en XCom.")

        # 3) Preparar payload
        payload = {
            "exp_id":    int(cmp["exp_id"]),
            "model_name": "my_model",
            "run_id":    cmp["run_id"],
            "run_name": run_name,
            "mv":        int(cmp["mv"]),
            "new_mse":   cmp["mse"],
            "new_rmse":  cmp["rmse"],
            "new_mae":   cmp["mae"],
            "new_r2":    cmp["r2"],
            "prod_mse":  cmp["prod_mse"],
            "prod_rmse": cmp["prod_rmse"],
            "prod_mae":  cmp["prod_mae"],
            "prod_r2":   None if cmp["prod_r2"] == float("-inf") else cmp["prod_r2"],
            "promoted":  bool(cmp["promoted"]),
            "shap_uri":  shap_uri
        }

        # 4) Inserción en la tabla
        engine = create_engine(CLEAN_DB_URI)
        with engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {SCHEMA_CLEAN}.model_history (
                  experiment_id, model_name, run_id, run_name, model_version,
                  new_mse, new_rmse, new_mae, new_r2,
                  prod_mse, prod_rmse, prod_mae, prod_r2,
                  promoted, shap_uri
                ) VALUES (
                  :exp_id, :model_name, :run_id, :run_name, :mv,
                  :new_mse, :new_rmse, :new_mae, :new_r2,
                  :prod_mse, :prod_rmse, :prod_mae, :prod_r2,
                  :promoted, :shap_uri
                )
            """), payload)

        print("✅ Registro en model_history insertado con shap_uri:", shap_uri)

    record_history = PythonOperator(
        task_id="record_model_history",
        python_callable=record_history_fn,
    )

    # ─── Flujo de dependencias ────────────────────────────────────────────────
    wait_for_clean \
        >> ensure_history_table \
        >> extract_data \
        >> train_and_log \
        >> evaluate_and_promote \
        >> compute_shap \
        >> record_history
