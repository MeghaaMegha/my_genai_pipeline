import os
import mlflow
import logging
from src.utils import get_production_logger

logger = get_production_logger(__name__)


def log_pipeline_experiment(run_name: str, parameters: dict, metrics: dict):
    """Orchestrate and log pipeline telemetry details inside a local MLflow registry."""
    try:
        # Establish or fetch a centralized named experiment category space
        mlflow.set_experiment("BERT_Text_Prep_Pipeline")

        logger.info(
            f"Starting MLflow automated run session tracking under signature: {run_name}"
        )
        with mlflow.start_run(run_name=run_name):
            # Log structural hyperparameters configurations
            mlflow.log_params(parameters)

            # Log calculated accuracy/loss evaluations metrics signatures
            mlflow.log_metrics(metrics)

            # Explicitly capture metadata references
            mlflow.set_tag("pipeline_phase", "Tokenization_Prep")

        logger.info(
            "[SUCCESS] Session run values written into local registry metrics layer."
        )

    except Exception as e:
        logger.error(f"MLflow experiment lifecycle step encountered an issue: {e}")
        raise e


if __name__ == "__main__":
    # Mock data execution test loop parameters block
    sample_params = {"max_sequence_length": 16, "vocab_target": "bert-base-uncased"}
    sample_metrics = {"data_ingest_records": 100, "token_extraction_efficiency": 0.98}

    log_pipeline_experiment(
        run_name="Initial_Local_Run", parameters=sample_params, metrics=sample_metrics
    )
