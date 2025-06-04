# backend/blip_logic/model_handler.py
# ... (other imports)
import numpy as np
import logging
from .plugir_blip_retriever import PlugIRStyleRetriever
from .Questioner import LLM_Connector
import pickle
import os
from PIL import Image

logger = logging.getLogger(__name__)



# For retrieval
g_retrieval_image_ids_list = []
g_retrieval_image_embeddings_np = np.array([])

# --- Configuration ---
RETRIEVER_MODEL_NAME = "Salesforce/blip-itm-large-coco"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-large"

# Path for embeddings used in the image retrieval step
RETRIEVAL_GALLERY_PICKLE_PATH = '/home/shangrong/research/demo_website/backend/blip_logic/embeddings/visdial_val2018_gallery_embeddings.pkl'

# Path for embeddings/data potentially related to captioning (from your original script)
# NOTE: The effective logic in your terminal script did not directly use data from
# this specific pickle for the caption_instance's operation variables.
# If specific data from here (e.g., a distinct list of captionable image IDs)
# is needed, its usage will need to be explicitly defined.
CAPTION_GALLERY_PICKLE_PATH = '/home/shangrong/research/demo_website/backend/blip_logic/embeddings/visdial_val2018_embeddings_caption_large.pkl' # Currently noted, but not loaded by default unless a specific use is defined.

IMAGE_BASE_DIRECTORY = "/home/shangrong/research/datasets/VisualDialog_val2018" # User specific
# LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY_HERE")
# LLM_BASE_URL = os.getenv("LLM_BASE_URL", "YOUR_LLM_BASE_URL_HERE")
# LLM_API_KEY = "sk-mR9xMzFRbOWD21nzrCC8mnjTXYeIxEaKVho0ftGXw5VzQ3VM"
# LLM_BASE_URL = "https://tbnx.plus7.plus/v1"
LLM_API_KEY = "xai-E2jU2fHXhbqB6uxaUHsS24Ou0ECXWpnbdkNOydZsUuSNldvBdrBizADkSBnijXQY5hijMcIqq4WsvPEm"
LLM_BASE_URL = "https://api.x.ai/v1"

# --- Global Variables for Models and Data ---
retriever_instance = PlugIRStyleRetriever(model_name_or_path=RETRIEVER_MODEL_NAME)
caption_instance = PlugIRStyleRetriever(model_name_or_path=CAPTION_MODEL_NAME)
llm_connector_instance = LLM_Connector(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


DEFAULT_N_CANDIDATES = 50
DEFAULT_M_REPRESENTATIVES = 10


def initialize_models_and_data(
    retriever_model: str = RETRIEVER_MODEL_NAME,
    caption_model: str = CAPTION_MODEL_NAME,
    # Explicitly use the retrieval gallery path for loading retrieval embeddings
    retrieval_gallery_path: str = RETRIEVAL_GALLERY_PICKLE_PATH,
    llm_api_key: str = LLM_API_KEY,
    llm_base_url: str = LLM_BASE_URL
):
    global retriever_instance, caption_instance, llm_connector_instance
    global g_retrieval_image_ids_list, g_retrieval_image_embeddings_np

    logger.info("Initializing models and data...")

    # 1. Initialize Retrievers and Captioner
    try:
        logger.info(f"Loading retriever model: {retriever_model}")
        retriever_instance = PlugIRStyleRetriever(model_name_or_path=retriever_model)
        logger.info(f"Loading captioning model: {caption_model}")
        caption_instance = PlugIRStyleRetriever(model_name_or_path=caption_model)
        logger.info("BLIP models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading BLIP models: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load BLIP models: {e}")

    # 2. Initialize LLM Connector (omitted for brevity, same as before)
    # ...

    # 3. Load Gallery Embeddings and IDs for RETRIEVAL
    if not os.path.exists(retrieval_gallery_path):
        logger.error(f"ERROR: Retrieval gallery pickle file not found at {retrieval_gallery_path}.")
        raise FileNotFoundError(f"Retrieval gallery pickle file not found: {retrieval_gallery_path}")

    try:
        with open(retrieval_gallery_path, 'rb') as f:
            gallery_data = pickle.load(f)
        g_retrieval_image_ids_list = gallery_data['image_ids'] # Specifically for retrieval
        g_retrieval_image_embeddings_np = gallery_data['embeddings'] # Specifically for retrieval
        if not isinstance(g_retrieval_image_embeddings_np, np.ndarray):
            g_retrieval_image_embeddings_np = np.array(g_retrieval_image_embeddings_np)
        logger.info(f"Loaded RETRIEVAL gallery from '{retrieval_gallery_path}' with {len(g_retrieval_image_ids_list)} images, embedding dim: {g_retrieval_image_embeddings_np.shape}")
    except Exception as e:
        logger.error(f"Error loading retrieval gallery data from {retrieval_gallery_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load retrieval gallery data: {e}")

    # If CAPTION_GALLERY_PICKLE_PATH contains data that needs to be loaded and used
    # (e.g., a different set of image IDs, or specific metadata for captioning),
    # that loading logic would be added here. For now, we stick to the observed effective
    # logic from your terminal script where retrieval embeddings were primary for candidate selection.

    logger.info("Models and data initialization complete.")

# ...

def load_pil_image_from_id_handler(image_id: str, image_base_dir: str = IMAGE_BASE_DIRECTORY):
    potential_filenames = [
        f"{image_id}.jpg",
        f"VisualDialog_val2018_{image_id}.jpg",
    ]
    if isinstance(image_id, int) or image_id.isdigit():
        try:
            formatted_id_str = f"{int(image_id):012d}"
            potential_filenames.append(f"VisualDialog_val2018_{formatted_id_str}.jpg")
        except ValueError:
            pass

    img_path_to_load = None
    for fname in potential_filenames:
        test_path = os.path.join(image_base_dir, fname)
        if os.path.exists(test_path):
            img_path_to_load = test_path
            break
    
    if not img_path_to_load:
        last_try_path = os.path.join(image_base_dir, image_id) # If image_id might already include extension
        if os.path.exists(last_try_path):
            img_path_to_load = last_try_path
        else:
            logger.error(f"Image path not found for ID {image_id} in {image_base_dir} after checking common patterns.")
            return None, None

    try:
        image = Image.open(img_path_to_load).convert("RGB")
        logger.debug(f"Successfully loaded image: {img_path_to_load}")
        return image, img_path_to_load
    except Exception as e:
        logger.error(f"Error loading image {img_path_to_load}: {e}")
        return None, None

        

# --- Main function for retrieval and captioning ---
def get_representative_images_and_captions(
    query_text: str,
    num_candidates: int = DEFAULT_N_CANDIDATES,
    num_representatives: int = DEFAULT_M_REPRESENTATIVES
):
    """
    Performs retrieval, representative selection, and captioning for a given query.
    Returns:
        List of dicts, each like: {"image_id": str, "caption": str, "image_url": str}
        Returns an empty list if any critical step fails.
    """
    if not retriever_instance or not caption_instance:
        logger.error("Models not initialized. Call initialize_models_and_data() first.")
        return []
    if not g_retrieval_image_ids_list or g_retrieval_image_embeddings_np.size == 0:
        logger.error("Gallery data not loaded.")
        return []

    logger.info(f"Processing query for retrieval & captioning: '{query_text[:100]}...'")

    # Step 1: Get query embedding
    try:
        query_embedding_batch = retriever_instance.get_text_embeddings([query_text.strip()], batch_size=1)
        if query_embedding_batch is None or query_embedding_batch.shape[0] == 0:
            logger.error("Failed to generate query embedding.")
            return []
        current_query_embedding = query_embedding_batch[0]
    except Exception as e:
        logger.error(f"Error getting text embeddings: {e}", exc_info=True)
        return []

    # Step 2: Retrieve top N candidate images
    try:
        candidate_indices, _ = retriever_instance.retrieve_top_k_indices_and_scores(
            current_query_embedding,
            g_retrieval_image_embeddings_np,
            top_k=num_candidates
        )
        if len(candidate_indices) == 0:
            logger.warning("No candidate images retrieved.")
            return []
        top_n_candidate_ids = [g_retrieval_image_ids_list[idx] for idx in candidate_indices]
        top_n_candidate_embeddings = g_retrieval_image_embeddings_np[candidate_indices]
        logger.info(f"Retrieved {len(top_n_candidate_ids)} candidates.")
    except Exception as e:
        logger.error(f"Error retrieving top k candidates: {e}", exc_info=True)
        return []

    # Step 3: Select M representative images
    try:
        if not top_n_candidate_ids: # Should be caught by len(candidate_indices) but good to double check
             logger.warning("No candidates available for representative selection.")
             return []
        representative_image_ids = retriever_instance.select_representative_images(
            candidate_ids=top_n_candidate_ids,
            candidate_embeddings_normalized=top_n_candidate_embeddings, # Assumes embeddings are normalized by retriever
            num_representatives=min(num_representatives, len(top_n_candidate_ids)) # Ensure not asking for more than available
        )
        if not representative_image_ids:
            logger.warning("No representative images selected.")
            return []
        logger.info(f"Selected {len(representative_image_ids)} representative IDs: {representative_image_ids}")
    except Exception as e:
        logger.error(f"Error selecting representative images: {e}", exc_info=True)
        return []

    # Step 4: Load PIL images for selected representatives and prepare for captioning
    pil_images_for_captioning = []
    # Store details to match captions back to IDs and URLs later
    representative_details_for_captioning = []

    for img_id in representative_image_ids:
        pil_image, actual_img_path = load_pil_image_from_id_handler(img_id)
        if pil_image and actual_img_path:
            pil_images_for_captioning.append(pil_image)
            try:
                # Construct a relative path for the API URL.
                # Example: IMAGE_BASE_DIRECTORY = /path/to/VisualDialog_val2018
                # actual_img_path = /path/to/VisualDialog_val2018/VisualDialog_val2018_000000xxxxxx.jpg
                # relative_path = VisualDialog_val2018_000000xxxxxx.jpg
                relative_image_path = os.path.relpath(actual_img_path, IMAGE_BASE_DIRECTORY)
                # Sanitize for URL (e.g., convert backslashes if on Windows)
                api_image_url = f"/static_images/{relative_image_path.replace(os.sep, '/')}"
            except ValueError:
                api_image_url = None # Should not happen if actual_img_path is from IMAGE_BASE_DIRECTORY
                logger.warning(f"Could not make image path {actual_img_path} relative to {IMAGE_BASE_DIRECTORY}")

            representative_details_for_captioning.append({
                "id": img_id,
                "url": api_image_url
            })
        else:
            logger.warning(f"Could not load PIL image for representative ID {img_id} for captioning.")

    if not pil_images_for_captioning:
        logger.warning("No PIL images could be loaded for the selected representatives. Cannot generate captions.")
        # Return empty or just IDs if partial results are okay, for now returning empty.
        return []

    # Step 5: Generate captions for these M representative images
    final_results = []
    try:
        logger.info(f"Generating captions for {len(pil_images_for_captioning)} representative images...")
        generated_captions = caption_instance.generate_captions(pil_images_for_captioning)

        if len(generated_captions) != len(representative_details_for_captioning):
            logger.error(f"Mismatch in number of generated captions ({len(generated_captions)}) and representative images ({len(representative_details_for_captioning)}).")
            # Handle mismatch, e.g., by returning what's available or erroring out
            # For now, we'll proceed cautiously and only pair up what we can.
            # A more robust solution might involve error captions or skipping problematic ones.
            
        for i, detail in enumerate(representative_details_for_captioning):
            if i < len(generated_captions):
                caption_text = generated_captions[i]
                final_results.append({
                    "image_id": detail["id"],
                    "caption": caption_text,
                    "image_url": detail["url"]
                })
                logger.info(f"  ID: {detail['id']}, URL: {detail['url']}, Caption: {caption_text}")
            else:
                # This case handles if fewer captions were returned than images sent for captioning.
                logger.warning(f"No caption generated for image ID: {detail['id']}")
                final_results.append({
                    "image_id": detail["id"],
                    "caption": "Error: Caption not generated for this image.", # Or None
                    "image_url": detail["url"]
                })
        
    except Exception as e:
        logger.error(f"Error generating captions: {e}", exc_info=True)
        # If captioning fails, you might still want to return images without captions
        # or handle this as a complete failure for the items being captioned.
        # For now, we'll return what we have, potentially with error captions.
        if not final_results and representative_details_for_captioning: # If captioning failed entirely
            logger.warning("Caption generation failed. Returning images without captions.")
            for detail in representative_details_for_captioning:
                final_results.append({
                    "image_id": detail["id"],
                    "caption": "Caption generation failed.",
                    "image_url": detail["url"]
                })
        elif not final_results: # No images were processable for captioning and captioning failed
             return []


    if not final_results:
        logger.warning("No representative images with captions were successfully processed.")

    return final_results


def reformulate_query_with_llm(original_query: str, dialogue_history: list):
    """
    Uses LLM to reformulate the query based on dialogue history.
    dialogue_history is a list of dicts, e.g., [{"role": "user/assistant", "content": "text"}]
    """
    if not llm_connector_instance:
        logger.warning("LLM Connector not initialized. Returning original query.")
        return original_query

    # Adapt dialogue_history to the format expected by your LLM_Connector.reformulate
    # Your LLM_Connector.reformulate expects (description_example, dialogue)
    # where dialogue seems to be a list of {"Question": q, "Answer": a}
    # We need to map FastAPI's session dialogue history to this format.
    # Assuming dialogue_history in FastAPI will be like:
    # [ {"role": "user", "content": "first query"},
    #   {"role": "assistant", "content": "llm question 1"},
    #   {"role": "user", "content": "answer to q1"}, ... ]
    
    # For LLM_Connector, it might be simpler if it takes the current query and the history.
    # Let's assume `dialogue_history` for LLM_Connector is the [{"Question":q, "Answer":a}, ...] format
    # The API will need to maintain this structure in its session.
    
    logger.info(f"Reformulating query: '{original_query}' with dialogue: {dialogue_history}")
    try:
        reformulated = llm_connector_instance.reformulate(original_query, dialogue_history)
        if reformulated:
            logger.info(f"Reformulated query: {reformulated}")
            return reformulated
        else:
            logger.warning("LLM failed to reformulate query. Using original.")
            return original_query
    except Exception as e:
        logger.error(f"Error during LLM query reformulation: {e}", exc_info=True)
        return original_query


def generate_question_with_llm(retrieved_captions_dict: dict, original_query: str, dialogue_history: list):
    """
    Uses LLM to generate a follow-up question.
    retrieved_captions_dict: {"image_id1": "caption1", ...}
    dialogue_history: list of {"Question": q, "Answer": a}
    """
    if not llm_connector_instance:
        logger.warning("LLM Connector not initialized. Cannot generate question.")
        return None, "LLM not available." # Return None for question, and a system message

    logger.info(f"Generating LLM question based on captions: {list(retrieved_captions_dict.values())[:2]}...")
    try:
        # LLM_Connector.generate_question expects (retrieved_img_with_generated_caption, description_example, dialogue)
        question = llm_connector_instance.generate_question(
            retrieved_captions_dict,
            original_query,
            dialogue_history
        )
        if question:
            # Your script then checks `filter_question`. Let's include that concept.
            # The filter logic in your script is: if "Unvertain" or "uncertain" in filter1: pass
            # This suggests filter1 IS the question, and it's checking for uncertainty keywords
            # in the question itself to decide if it's a good question.
            # Let's assume filter_question just returns the question if good, or a specific marker if bad.
            
            # For simplicity here, let's assume generate_question already returns a usable question string or None.
            # The filter logic might be more complex in your Questioner.py.
            # We can add a call to filter_question if needed.
            # For now, we directly use the generated question.
            # filter_result = llm_connector_instance.filter_question(original_query, dialogue_history + [{"Question": question, "Answer": ""}]) # Tentative answer
            # if "uncertain" in filter_result.lower(): # Your condition
            #    logger.info(f"Generated LLM question (passed filter): {question}")
            #    return question, None
            # else:
            #    logger.info(f"Generated LLM question did NOT pass filter: {filter_result}. Not using this question.")
            #    return None, "I'm not sure how to follow up on that. Could you rephrase or try a different aspect?"
            logger.info(f"Generated LLM question: {question}")
            return question, None # Question, no system message
        else:
            logger.warning("LLM failed to generate a question.")
            return None, "I couldn't think of a good follow-up question. What would you like to ask or refine?"
    except Exception as e:
        logger.error(f"Error during LLM question generation: {e}", exc_info=True)
        return None, "An error occurred while trying to generate a follow-up question."


def filter_generated_llm_question(original_query: str, dialogue_history: list, generated_question_text: str):
    """
    Uses LLM to filter/validate a generated question.
    original_query: The query context for the LLM.
    dialogue_history: List of {"Question": q, "Answer": a}
    generated_question_text: The question generated by the LLM that needs filtering.

    Returns:
        The validated question text if it passes the filter,
        or None if it doesn't pass or an error occurs.
    """
    if not llm_connector_instance:
        logger.warning("LLM Connector not initialized. Cannot filter question.")
        return generated_question_text # Pass through if no filter available

    if not generated_question_text: # No question was generated to filter
        return None

    try:
        # The filter_question in your script takes (description_example, dialogue)
        # The dialogue passed to filter_question should ideally include the question being filtered
        # to see if the LLM thinks the current state (including this new question) is "uncertain".
        # We'll append a temporary entry for the new question without an answer for filtering context.
        temp_dialogue_for_filtering = dialogue_history + [{"Question": generated_question_text, "Answer": ""}]
        
        filter_response = llm_connector_instance.filter_question(original_query, temp_dialogue_for_filtering)
        logger.info(f"LLM filter_question response for '{generated_question_text}': {filter_response}")

        # Your original condition: if "Unvertain" or "uncertain" in filter1:
        # This condition `bool("Unvertain")` is always True.
        # Corrected to check for presence of "uncertain" (case-insensitive).
        # The meaning of filter_response needs to be clear:
        # - If filter_response IS the question text itself when it passes,
        # - Or if filter_response is some status string.
        # Assuming filter_response is a status string or the question itself,
        # and we're checking if "uncertain" is mentioned by the LLM *about* the state.
        # Let's assume if "uncertain" (or similar negative keyword) IS NOT in the filter_response, the question is good.
        # Or, if your filter_question returns the question if good, and something else if bad.
        # Based on your terminal script's `if "Unvertain" or "uncertain" in filter1: print(f'filtering pass: {filter1}')`
        # it seems if these keywords are present, it's considered a "pass" (which is counter-intuitive for "uncertain").
        # Let's reinterpret: "filter1" is the result of the filtering. If it contains "uncertain", the LLM might be expressing doubt.
        # Let's assume your LLM_Connector.filter_question() returns the question if it's good,
        # or a string containing "uncertain" (or similar) if the LLM advises against asking it.

        # Re-evaluating your original logic:
        # if "Unvertain" or "uncertain" in filter1: -> This means if the filter_response contains "uncertain", it's a PASS.
        # This is unusual. Usually, "uncertain" would mean the question is not good.
        # Let's proceed with your script's apparent logic: "uncertain" in filter_response means the question is okay to ask.
        # You may need to adjust this logic based on the actual behavior of your LLM_Connector.filter_question.
        
        # For robustness, let's assume filter_question returns the original question if it's good,
        # or a modified string/None if it's bad.
        # If your filter_question returns the question itself and its presence in the string means it passed:
        if filter_response and ("uncertain" in filter_response.lower() or "unvertain" in filter_response.lower()): # Corrected keyword check
            logger.info(f"LLM question '{generated_question_text}' passed filter with response: {filter_response}")
            return generated_question_text # Return the original generated question if filter criteria met
        else:
            logger.warning(f"LLM question '{generated_question_text}' did NOT pass filter. Filter response: {filter_response}")
            return None # Question did not pass filter

    except Exception as e:
        logger.error(f"Error during LLM question filtering: {e}", exc_info=True)
        return None # Default to not using the question if filtering errors