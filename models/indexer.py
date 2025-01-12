# models/indexer.py

import os
from byaldi import RAGMultiModalModel
from models.converters import convert_docs_to_pdfs
from logger import get_logger
import base64
from models.model_loader import load_model

logger = get_logger(__name__)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_images(image_paths):
    """
    Extracts text from a list of image paths using a vision model.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        str: Concatenated extracted text from all images.
    """
    client = load_model('groq-llama-vision')
    generated_text = ""

    for img_path in image_paths:
        if os.path.exists(img_path):
            base64_image = encode_image(img_path)
            content = [
                {
                    "type": "text",
                    "text": (
                        "You are a text extraction tool. Extract and return ONLY the exact text visible in the image. "
                        "Do not describe the image, background or add any commentary. Format the text exactly as it appears, "
                        "maintaining headings, lists and structure. Do not add any additional formatting or descriptions. "
                        "For tables, output each row in this exact format: {row_number}. {column1} - {column2}. "
                        "Do not add any other text or descriptions."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                model="llama-3.2-90b-vision-preview",
                temperature=0.1,
            )
            generated_text += chat_completion.choices[0].message.content
    return generated_text


def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali'):
    """
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update.
        index_path (str): The path where the index should be saved.
        indexer_model (str): The name of the indexer model to use.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    try:
        logger.info(f"Starting document indexing in folder: {folder_path}")
        # Convert non-PDF documents to PDFs
        convert_docs_to_pdfs(folder_path)
        logger.info("Conversion of non-PDF documents to PDFs completed.")

        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
        logger.info(f"RAG model initialized with {indexer_model}.")

        # Prepare metadata for each document
        metadata_list = []
        for doc_id, doc_path in enumerate(os.listdir(folder_path)):
            doc_full_path = os.path.join(folder_path, doc_path)
            if os.path.isfile(doc_full_path):
                # Extract text from images in the document
                extracted_text = extract_text_from_images([doc_full_path])
                metadata = {
                    "doc_id": doc_id,
                    "file_name": doc_path,
                    "extracted_text": extracted_text,
                    # Add other metadata fields as needed
                }
                metadata_list.append(metadata)

        # Index the documents in the folder
        RAG.index(
            input_path=folder_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True,
            metadata=metadata_list
        )

        logger.info(f"Indexing completed. Index saved at '{index_path}'.")

        return RAG
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise