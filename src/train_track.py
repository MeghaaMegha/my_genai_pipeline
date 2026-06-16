import os
import logging
import pickle
import mlflow

logger = logging.getLogger(__name__)


def log_pipeline_experiment(
    run_name: str,
    parameters: dict,
    metrics: dict,
    tensor_artifacts: dict = None,
    vector_db_dir: str = None,
):
    """
    Log structural engineering parameters, execution metrics, and serialized binary artifacts to MLflow.
    """
    try:
        mlflow.set_experiment("BERT_Text_Prep_Pipeline")

        with mlflow.start_run(run_name=run_name):
            logger.info(f"MLflow Session opened. Logging metadata for run: {run_name}")

            # 1. Log metadata variables and metric parameters
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)

            # 2. Serialize and log Deep Learning Tensors as binary artifacts
            if tensor_artifacts:
                logger.info(
                    "Serializing processed BERT token tensors to local binary buffers."
                )
                temp_tensor_path = "bert_features.pkl"
                with open(temp_tensor_path, "wb") as f:
                    pickle.dump(tensor_artifacts, f)

                # Push the binary file directly into the MLflow artifact container registry
                mlflow.log_artifact(temp_tensor_path, artifact_path="model_features")

                # Clean up local temporary file buffer
                if os.path.exists(temp_tensor_path):
                    os.remove(temp_tensor_path)
                logger.info(
                    "[SUCCESS] Serialized PyTorch tensor assets successfully archived in MLflow."
                )

            # 3. Snapshot and version the entire Chroma DB index directory
            if vector_db_dir and os.path.exists(vector_db_dir):
                logger.info(
                    f"Archiving local persistent Vector Database folder: {vector_db_dir}"
                )
                # Log the entire directory of SQLite and parquet files into MLflow tracking storage
                mlflow.log_artifacts(vector_db_dir, artifact_path="chroma_vector_db")
                logger.info(
                    "[SUCCESS] Chroma Vector DB snapshot successfully versioned in MLflow."
                )

            # Apply operational compliance tracking tags
            mlflow.set_tag("pipeline_tier", "Production_RAG_V1")
            mlflow.set_tag("framework", "Poetry_PySpark_Transformers_Chroma")

            logger.info(
                "[SUCCESS] Session run values and heavy file assets written into local registry layer."
            )

    except Exception as e:
        logger.error(
            f"Critical failure logging enterprise pipeline assets to MLflow: {e}"
        )
        raise e
