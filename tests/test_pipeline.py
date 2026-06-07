import logging
import os
import pytest
from pyspark.sql import SparkSession
from src.utils import get_production_logger
from src.data_ingest import clean_text_data, tokenize_text


# ==========================================
# DAY 1: LOGGER UTILITY TESTS
# ==========================================
def test_get_production_logger():
    """Verify production logger builds and initializes correctly."""
    logger = get_production_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO


# ==========================================
# DAYS 3-7: PYSPARK INGESTION FIXTURES & TESTS
# ==========================================
@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Fixture to initialize a local, isolated Spark Session bypassing remote cluster requirements."""
    spark = (
        SparkSession.builder.remote(
            "local[*]"
        )  # Forces Databricks Connect to run locally
        .appName("pipeline-unit-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_clean_text_data_normalization(spark_session: SparkSession):
    """Verify that text is correctly lowercased and special characters are removed."""
    input_data = [
        ("1", "Hello, World! 🚀"),
        ("2", "Data-Engineering_101..."),
    ]
    schema = ["id", "raw_text"]
    input_df = spark_session.createDataFrame(input_data, schema)

    output_df = clean_text_data(input_df, target_column="raw_text")
    results = {row["id"]: row["cleaned_raw_text"] for row in output_df.collect()}

    assert "hello world " in results["1"]
    assert "dataengineering101" in results["2"]


def test_tokenize_text_splitting(spark_session: SparkSession):
    """Verify that tokenization correctly splits strings into arrays of words."""
    # Arrange: Set up clean input data with words separated by single and multiple spaces
    input_data = [
        ("1", "hello world"),
        ("2", "data    engineering    pipeline"),
    ]
    schema = ["id", "cleaned_text"]
    input_df = spark_session.createDataFrame(input_data, schema)

    # Act: Run the tokenization function
    output_df = tokenize_text(input_df, target_column="cleaned_text")
    results = {row["id"]: row["tokenized_cleaned_text"] for row in output_df.collect()}

    # Assert: Verify the resulting string lists match perfectly
    assert results["1"] == ["hello", "world"]
    assert results["2"] == ["data", "engineering", "pipeline"]


from src.bert_prep import tokenize_for_bert


def test_tokenize_for_bert_dimensions():
    """Verify BERT Tokenizer correctly generates tensor shapes."""
    # Arrange: Build a sample text sequence batch
    sample_texts = [
        "Testing machine learning engineering structures.",
        "BERT model ingestion validation.",
    ]

    # Act: Encode text strings with a restricted sequence cutoff
    encoded = tokenize_for_bert(sample_texts, max_length=16)

    # Assert: Confirm critical transformer dictionary keys exist
    assert "input_ids" in encoded
    assert "attention_mask" in encoded

    # Confirm shape sizing patterns: 2 sequences, 16 length tokens long
    assert encoded["input_ids"].shape == (2, 16)
    assert encoded["attention_mask"].shape == (2, 16)


from src.vector_store import generate_and_store_embeddings


def test_vector_store_persistence(tmp_path):
    """Verify that text items are cleanly converted into persistent Chroma vector layers."""
    # Arrange: Build test documents and map a safe temporary directory path
    test_corpus = ["validation text sequence alpha", "testing vector embedding omega"]
    temp_db_dir = os.path.join(tmp_path, "temp_chroma")

    # Act: Run the embedding and storage persistence engine loop
    db_instance = generate_and_store_embeddings(
        test_corpus, persist_directory=temp_db_dir
    )

    # Assert: Confirm the database object initializes and records the correct sizes
    assert db_instance is not None

    # Query the local vector collection to verify the items exist inside Chroma
    collection_data = db_instance.get()
    assert len(collection_data["documents"]) == 2
    assert "testing vector embedding omega" in collection_data["documents"]
