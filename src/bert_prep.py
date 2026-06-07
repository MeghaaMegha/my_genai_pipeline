import logging
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def tokenize_for_bert(text_list: list, max_length: int = 128) -> dict:
    """Convert clean string tokens into numeric tensor inputs for BERT."""
    try:
        logger.info("Initializing HuggingFace pre-trained BERT tokenizer.")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        logger.info(f"Encoding text batch with strict truncation limit: {max_length}")
        encoded_inputs = tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return dict(encoded_inputs)

    except Exception as e:
        logger.error(f"Failed to generate BERT tokens: {e}")
        raise e
