import sys
import logging
from src.utils import get_production_logger
from src.data_ingest import get_spark_session, clean_text_data, tokenize_text
from src.bert_prep import tokenize_for_bert
from src.train_track import log_pipeline_experiment
from src.vector_store import generate_and_store_embeddings, retrieve_relevant_context


# Initialize our standardized production logging system
logger = get_production_logger("main_pipeline")


def run_end_to_end_pipeline():
    """Orchestrate the GenAI data pipeline from raw ingestion to model features."""
    logger.info("Initializing complete Milestone 1 GenAI Pipeline workflow.")

    # 1. Initialize our Spark environment
    spark = get_spark_session()

    # 2. Mock input data stream representing uncleaned documents
    raw_data = [
        ("1", "Enterprise Spark data-processing pipeline! 🚀"),
        ("2", "BERT model sequence ingestion metrics verification..."),
        ("3", "MLflow tracking configuration is completely functional."),
    ]
    schema = ["id", "raw_text"]

    logger.info(f"Ingesting raw batch containing {len(raw_data)} text sequences.")
    df_raw = spark.createDataFrame(raw_data, schema)

    # 3. Apply PySpark text normalization transformations
    df_cleaned = clean_text_data(df_raw, target_column="raw_text")

    # 4. Apply PySpark array tokenization splits
    df_tokenized = tokenize_text(df_cleaned, target_column="cleaned_raw_text")

    # 5. Extract our normalized column to process inside our Deep Learning engine
    logger.info(
        "Extracting data frames out of PySpark to prepare deep learning features."
    )
    collected_rows = df_tokenized.select("cleaned_raw_text").collect()
    clean_strings = [row["cleaned_raw_text"].strip() for row in collected_rows]

    # 6. Generate deep learning features using our BERT tokenizer
    max_seq_len = 16
    bert_features = tokenize_for_bert(clean_strings, max_length=max_seq_len)
    logger.info("Indexing cleaned text fragments into persistent vector layers.")
    generate_and_store_embeddings(clean_strings, persist_directory="./chroma_db")

    # 7. Package metrics and run variables for MLOps indexing
    pipeline_params = {
        "max_sequence_length": max_seq_len,
        "tokenizer_dictionary": "bert-base-uncased",
        # Force cast to a string or default to an integer 2 if it reads 'default'
        "spark_partitions": str(spark.conf.get("spark.sql.shuffle.partitions", "2")),
    }

    pipeline_metrics = {
        "processed_records_count": len(clean_strings),
        "tensor_batch_dimension": bert_features["input_ids"].shape[0],
        "tensor_sequence_dimension": bert_features["input_ids"].shape[1],
    }

    # 8. Record the experiment telemetry run inside MLflow with deep artifact tracking
    logger.info("Writing run logs and performance data to local MLflow registry.")
    log_pipeline_experiment(
        run_name="Milestone_Project_1_Run",
        parameters=pipeline_params,
        metrics=pipeline_metrics,
        tensor_artifacts=bert_features,
        vector_db_dir="./chroma_db"
    )
    logger.info("Verifying RAG Knowledge Layer via Context Retrieval Routing...")
    sample_search_prompt = "Tell me about the distributed Spark data processing components."
    
    contexts = retrieve_relevant_context(query=sample_search_prompt, persist_directory="./chroma_db", k=1)
    
    for idx, context in enumerate(contexts):
        logger.info(f"[RETRIEVED CONTEXT #{idx+1}]: {context}")

    logger.info("[PIPELINE COMPLETE] Milestone Project 1 executed successfully.")


if __name__ == "__main__":
    try:
        run_end_to_end_pipeline()
    except Exception as error:
        logger.critical(
            f"Pipeline execution aborted due to unhandled exception: {error}"
        )
        sys.exit(1)
