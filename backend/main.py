# backend/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles # For serving images
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
import uuid # For session IDs

# Import functions and variables from your model_handler
# Assuming model_handler.py is in a subdirectory blip_logic
from blip_logic import model_handler

# If model_handler.py is in the same directory (backend/):
# from blip_logic import model_handler

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO) # Set to DEBUG for more verbosity
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="BLIP Interactive Image Retrieval API",
    description="API for dialogue-based interactive text-to-image retrieval using BLIP and LLM.",
    version="0.2.0",
)

# --- CORS (Cross-Origin Resource Sharing) ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://localhost:8081",  # <--- ADD THIS LINE
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8081",
    # Add your frontend's actual origin in production
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State / Session Management (Simple In-Memory) ---
# WARNING: This is for demonstration. For production, use Redis, a DB, or more robust session management.
# Session data will store:
#   "dialogue_history_llm": [{"Question": q, "Answer": a}, ... ] for LLM_Connector
#   "last_query": str
#   "last_system_question": str (e.g. LLM question or system message)
SESSION_STORAGE: Dict[str, Dict[str, Any]] = {}

# --- Application Startup: Load Models and Data ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing models and data...")
    try:
        # You might need to pass actual paths/keys if not using defaults from model_handler
        model_handler.initialize_models_and_data(
            llm_api_key=os.getenv("LLM_API_KEY", model_handler.LLM_API_KEY), # Example: Use env var if set
            llm_base_url=os.getenv("LLM_BASE_URL", model_handler.LLM_BASE_URL)
        )
        logger.info("Models and data initialized successfully.")

        # Mount static directory for images AFTER model_handler.IMAGE_BASE_DIRECTORY is known
        # This allows serving images directly from their original location if desired.
        # The URL path will be /static_images/...
        if model_handler.IMAGE_BASE_DIRECTORY and os.path.isdir(model_handler.IMAGE_BASE_DIRECTORY):
            app.mount(
                "/static_images",
                StaticFiles(directory=model_handler.IMAGE_BASE_DIRECTORY),
                name="static_images"
            )
            logger.info(f"Mounted static image directory: {model_handler.IMAGE_BASE_DIRECTORY} at /static_images")
        else:
            logger.warning(f"IMAGE_BASE_DIRECTORY ('{model_handler.IMAGE_BASE_DIRECTORY}') is not valid. Static images may not be served.")

    except Exception as e:
        logger.critical(f"Failed to initialize application: {e}", exc_info=True)
        # Depending on severity, you might want to exit or prevent app from fully starting
        raise RuntimeError(f"Application initialization failed: {e}")


# --- Pydantic Models for API Request/Response ---
class RetrievedImageInfo(BaseModel):
    image_id: str
    caption: str
    image_url: Optional[str] = None # e.g., /static_images/VisualDialog_val2018_000000xxxxxx.jpg

class RetrieveRequest(BaseModel):
    initial_query: str = Field(..., min_length=1, description="The initial text query from the user.")
    session_id: str = Field(None, description="existing session ID to resume.")

class InitiateResponse(BaseModel):
    session_id: str
    retrieved_images: List[RetrievedImageInfo]
    # llm_question: Optional[str] = None # First question from LLM
    system_message: Optional[str] = None # Any message from the system
    

class LLMQuestionResponse(BaseModel):
    session_id: str
    question: Any
    system_message: Optional[str] = None # Any message from the system

class LLMQuestionInput(BaseModel): # Renamed for clarity as it's a request
    session_id: str = Field(..., description="The active session ID.")
    query: str = Field(..., description="User's query for retrieval.")
    dialogue: list = Field(..., description="User's dialogue history with LLM questioner")
    image_caption: list = Field(..., description="a text list of represenative images' captions")
    # 'system_message' from your original LLMQuestionResponse seems out of place for a request, so omitted.

class InteractRequest(BaseModel):
    session_id: str = Field(..., description="The active session ID.")

class InteractResponse(BaseModel):
    session_id: str
    retrieved_images: List[RetrievedImageInfo]
    llm_question: Optional[str] = None # Next question from LLM
    system_message: Optional[str] = None

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the d BLIP Interactive Image Retrieval API!"}

@app.get("/initiate_session")
async def create_session():
    session_id = str(uuid.uuid4())
    SESSION_STORAGE[session_id] = {
            "dialogue_history_llm": [], # For LLM_Connector's format
            "last_system_question": None
        }
    return PlainTextResponse(content=session_id)

@app.post("/initiate_retrieve", response_model=InitiateResponse)
async def retrieve_images(request: RetrieveRequest):
    """
    Starts a new dialogue session or resumes an old one with an initial query.
    """
    session_id = request.session_id if request.session_id and request.session_id in SESSION_STORAGE else str(uuid.uuid4())
   
    if session_id not in SESSION_STORAGE:
        SESSION_STORAGE[session_id] = {
            "dialogue_history_llm": [], # For LLM_Connector's format
            "last_query": request.initial_query,
            "last_system_question": None
        }
    else: # Resuming session
        SESSION_STORAGE[session_id]["last_query"] = request.initial_query
        # Optionally clear parts of history or reset state if resuming means starting "fresh" with new query

    logger.info(f"[{session_id}] Retrieving images with query: '{request.initial_query}'")

    # 1. Get representative images and captions for the initial query
    # For the first turn, no reformulation is typically done.
    retrieved_data = model_handler.get_representative_images_and_captions(
        query_text=request.initial_query,
        num_candidates=model_handler.DEFAULT_N_CANDIDATES,
        num_representatives=model_handler.DEFAULT_M_REPRESENTATIVES
    )
    print(f'retrieved_data: {retrieved_data}')
    SESSION_STORAGE[session_id]["retrieved_data"] = retrieved_data
    if not retrieved_data: # If retrieval fails or returns nothing
        logger.warning(f"[{session_id}] No images retrieved for initial query.")
        # Fallback: maybe ask LLM to generate a question based on query alone?
        # Or just return an empty set. For now, let's try to generate a question.

    # Prepare captions dict for LLM
    # captions_for_llm = {item["image_id"]: item["caption"] for item in retrieved_data}

    # 2. Generate the first LLM question
    # The dialogue_history_llm is empty for the very first turn for generate_question
    # llm_question, system_msg = model_handler.generate_question_with_llm(
    #     retrieved_captions_dict=captions_for_llm,
    #     original_query=request.initial_query,
    #     dialogue_history=SESSION_STORAGE[session_id]["dialogue_history_llm"] # initially empty
    # )
    # SESSION_STORAGE[session_id]["last_system_question"] = llm_question or system_msg

    return InitiateResponse(
        session_id=session_id,
        retrieved_images=retrieved_data, # Already in List[RetrievedImageInfo] like format
        # llm_question=llm_question,
    #     system_message=system_msg if not llm_question else None # Only send system_msg if no question
    )



@app.post("/generate_question_via_LLM", response_model=LLMQuestionResponse)
async def generate_llm_question(request: InteractRequest): # Using renamed request model
    """
    Receives user's answer to a previous LLM question, updates dialogue,
    and generates a new LLM question based on current context.
    """
    session_id = request.session_id
    # query = request.query
    # dialogue = request.dialogue
    # caption = request.image_caption

    query = SESSION_STORAGE[session_id]["last_query"] 
    if "dialogue_history_llm" in SESSION_STORAGE[session_id]:
        dialogue = SESSION_STORAGE[session_id]["dialogue_history_llm"] 
    else:
        SESSION_STORAGE[session_id]["dialogue_history_llm"] = list()
    # dialogue = SESSION_STORAGE[session_id]["dialogue_history_llm"] if "dialogue_history_llm" in SESSION_STORAGE[session_id] else SESSION_STORAGE[session_id]["dialogue_history_llm"] = list()
    if session_id not in SESSION_STORAGE:
        raise HTTPException(status_code=404, detail="Session not found. Please initiate dialogue first.")

    session_data = SESSION_STORAGE[session_id]
    logger.info(f"[{session_id}] Generating new LLM question. User dialogue history: '{dialogue}'")

    # 1. Retrieve necessary context from session
    last_retrieved_list = session_data.get("retrieved_data")
    last_retrieved_captions_dict = dict()
    for item in last_retrieved_list:
        last_retrieved_captions_dict[item["image_id"]] = item["caption"]
    print(f'last_retrieved_captions_dict: {last_retrieved_captions_dict}')
    # This is the List[RetrievedImageInfo]
    # last_retrieved_images_full_info = session_data.get("last_retrieved_images_full_info", [])
    current_query_for_context = session_data.get("last_query", "") # Query that led to these images/captions
    dialogue_history_llm = session_data.get("dialogue_history_llm", [])
    last_llm_question_asked = session_data.get("last_system_question") # The question user is answering 

    if last_retrieved_captions_dict is None:
        logger.warning(f"[{session_id}] No retrieved image captions found in session to generate question from.")
        # Decide how to handle: error, or try to generate question without image context?
        # For now, let's return the current images with a message.
        return LLMQuestionResponse(
            session_id=session_id,
            question=None,
            system_message="I don't have any specific image context to ask about. What would you like to do next?"
        )

    # # 2. Update dialogue history with the user's current answer
    # if last_llm_question_asked: # Ensure there was a question to answer
    #     dialogue_history_llm.append(
    #         {"Question": last_llm_question_asked, "Answer": user_answer}
    #     )
    # else:
    #     # If no question was pending, user might be providing a general statement.
    #     # How to handle this depends on your desired dialogue flow.
    #     # For now, we'll add it as a statement if no prior question.
    #     # Or you might decide this endpoint is ONLY for answering questions.
    #     dialogue_history_llm.append(
    #         {"Question": "User statement (no prior LLM question)", "Answer": user_answer}
    #     )
    # session_data["dialogue_history_llm"] = dialogue_history_llm


    # 2. Generate a new question using the model_handler
    new_llm_question_raw, system_msg_from_gen = model_handler.generate_question_with_llm(
        retrieved_captions_dict=last_retrieved_captions_dict,
        original_query=current_query_for_context, # The query that led to the current image set
        dialogue_history=dialogue_history_llm
    )

    final_llm_question = ''
    system_message_for_response = system_msg_from_gen # Message from generation step

    # 3. Filter the generated question
    if new_llm_question_raw:
        filtered_question = model_handler.filter_generated_llm_question(
            original_query=current_query_for_context,
            dialogue_history=dialogue_history_llm, # Pass the most up-to-date history
            generated_question_text=new_llm_question_raw
        )
        
        if filtered_question:
            final_llm_question = filtered_question
            system_message_for_response = None # Clear system message if question is good
            session_data["last_system_question"] = final_llm_question
            logger.info(f"[{session_id}] Filtered LLM question: '{final_llm_question}'")
        else:
            # Question did not pass filter
            system_message_for_response = system_message_for_response or "I thought of a question, but I'm not sure it's the right one to ask now. What would you like to focus on?"
            logger.warning(f"[{session_id}] Raw LLM question '{new_llm_question_raw}' did not pass filter.")
    else:
        # No raw question was generated, system_msg_from_gen might already have a reason.
        system_message_for_response = system_message_for_response or "I couldn't think of a follow-up question right now."


    # 5. Update session with the new LLM utterance
    session_data["last_system_question"] = final_llm_question or system_message_for_response
    SESSION_STORAGE[session_id] = session_data # Save updated session data

    # 6. Return response
    return LLMQuestionResponse(
        session_id=session_id,
        question=final_llm_question, # Return the images that were context for this question
        system_message=system_message_for_response
    )


@app.post("/reformulate_query_via_LLM_and_retrieve", response_model=InitiateResponse)
async def reformulate_query_and_retrieve(session_id: str, answer: str): # Using renamed request model
    """
    Receives user's answer to a previous LLM question, updates dialogue,
    and reformulate a new query for retrieving images based on dialogue history
    """

    query = SESSION_STORAGE[session_id]["last_query"] 
    if "dialogue_history_llm" in SESSION_STORAGE[session_id]:
        dialogue = SESSION_STORAGE[session_id]["dialogue_history_llm"] 
    else:
        SESSION_STORAGE[session_id]["dialogue_history_llm"] = list()
    # dialogue = SESSION_STORAGE[session_id]["dialogue_history_llm"] if "dialogue_history_llm" in SESSION_STORAGE[session_id] else SESSION_STORAGE[session_id]["dialogue_history_llm"] = list()
    if session_id not in SESSION_STORAGE:
        raise HTTPException(status_code=404, detail="Session not found. Please initiate dialogue first.")

    session_data = SESSION_STORAGE[session_id]
    logger.info(f"[{session_id}] Generating new LLM question. User dialogue history: '{dialogue}'")

    
    # This is the List[RetrievedImageInfo]
    # last_retrieved_images_full_info = session_data.get("last_retrieved_images_full_info", [])
    current_query_for_context = session_data.get("last_query", "") # Query that led to these images/captions
    last_llm_question_asked = session_data.get("last_system_question") # The question user is answering 
    current_dialogue = {'Question': last_llm_question_asked, 'Answer': answer}
    dialogue.append(current_dialogue)
    
    SESSION_STORAGE[session_id]["dialogue_history_llm"] = dialogue
    print(f'dialogue_history_llm: {dialogue}')

    # if last_retrieved_captions_dict is None:
    #     logger.warning(f"[{session_id}] No retrieved image captions found in session to generate question from.")
    #     # Decide how to handle: error, or try to generate question without image context?
    #     # For now, let's return the current images with a message.
    #     return LLMQuestionResponse(
    #         session_id=session_id,
    #         question=None,
    #         system_message="I don't have any specific image context to ask about. What would you like to do next?"
    #     )

    # # 2. Update dialogue history with the user's current answer
    # if last_llm_question_asked: # Ensure there was a question to answer
    #     dialogue_history_llm.append(
    #         {"Question": last_llm_question_asked, "Answer": user_answer}
    #     )
    # else:
    #     # If no question was pending, user might be providing a general statement.
    #     # How to handle this depends on your desired dialogue flow.
    #     # For now, we'll add it as a statement if no prior question.
    #     # Or you might decide this endpoint is ONLY for answering questions.
    #     dialogue_history_llm.append(
    #         {"Question": "User statement (no prior LLM question)", "Answer": user_answer}
    #     )
    # session_data["dialogue_history_llm"] = dialogue_history_llm


    # 2. Generate a new question using the model_handler
    reformulated_query = model_handler.reformulate_query_with_llm(
        original_query=current_query_for_context,
        dialogue_history=dialogue
    )
    print(f'reformulated_query: {reformulated_query}')

    # 3. Filter the generated question
    try:
        if reformulated_query:
            retrieve_request = RetrieveRequest(
                initial_query=reformulated_query,
                session_id=session_id
            )
            logger.info(f"[{session_id}] Filtered LLM question: '{reformulated_query}'")
            retrieve_response =  await retrieve_images(request=retrieve_request)
            return retrieve_response
        else:
            # No raw question was generated, system_msg_from_gen might already have a reason.
            system_message_for_response = "I couldn't reformulate the query right now."
            return InitiateResponse(
                session_id=session_id,
                retrieved_images=[], 
                system_message=system_message_for_response
            )
    except Exception as e:
        logger.critical(f"Failed to retrieve image: {e}", exc_info=True)
        # Depending on severity, you might want to exit or prevent app from fully starting
        raise RuntimeError(f"Retrieving images failed: {e}")


        # 5. Update session with the new LLM utterance
        session_data["last_system_question"] = final_llm_question or system_message_for_response
        SESSION_STORAGE[session_id] = session_data # Save updated session data

    # 6. Return response
    return LLMQuestionResponse(
        session_id=session_id,
        question=final_llm_question, # Return the images that were context for this question
        system_message=system_message_for_response
    )


@app.post("/interact", response_model=InteractResponse)
async def api_interact_in_dialogue(request: InteractRequest):
    """
    Handles a user's response in an ongoing dialogue.
    """
    session_id = request.session_id
    user_answer = request.user_answer

    if session_id not in SESSION_STORAGE:
        raise HTTPException(status_code=404, detail="Session not found. Please initiate dialogue first.")

    logger.info(f"[{session_id}] Interaction. User answer: '{user_answer}'")

    # 1. Update dialogue history for LLM
    # The last system utterance was the question the user is answering.
    last_llm_question = SESSION_STORAGE[session_id].get("last_system_question")
    if last_llm_question: # Only add if there was a question to answer
         SESSION_STORAGE[session_id]["dialogue_history_llm"].append(
             {"Question": last_llm_question, "Answer": user_answer}
         )
    else: # User might be providing a refinement without a direct question
        # How to handle this depends on your LLM_Connector logic.
        # For now, we assume there was a previous question.
        # If not, the original_query for reformulation might just be the user_answer itself.
        pass


    # 2. Reformulate query (optional, based on your logic for when to reformulate)
    # Let's assume the last *original* query stored in the session is the base for reformulation.
    # Or, perhaps the user_answer itself IS the new query if no LLM question was pending.
    query_to_reformulate = SESSION_STORAGE[session_id].get("last_query", user_answer)

    current_query = model_handler.reformulate_query_with_llm(
        original_query=query_to_reformulate, # Or use user_answer if it's a new refinement
        dialogue_history=SESSION_STORAGE[session_id]["dialogue_history_llm"]
    )
    SESSION_STORAGE[session_id]["last_query"] = current_query # Update the query for next turn

    # 3. Get new representative images and captions based on (reformulated) query
    retrieved_data = model_handler.get_representative_images_and_captions(
        query_text=current_query,
        num_candidates=model_handler.DEFAULT_N_CANDIDATES,
        num_representatives=model_handler.DEFAULT_M_REPRESENTATIVES
    )
    if not retrieved_data:
        logger.warning(f"[{session_id}] No images retrieved for reformulated query: {current_query}")


    # Prepare captions dict for LLM
    captions_for_llm = {item["image_id"]: item["caption"] for item in retrieved_data}

    # 4. Generate next LLM question
    llm_question, system_msg = model_handler.generate_question_with_llm(
        retrieved_captions_dict=captions_for_llm,
        original_query=current_query, # Use the (potentially reformulated) query
        dialogue_history=SESSION_STORAGE[session_id]["dialogue_history_llm"]
    )
    SESSION_STORAGE[session_id]["last_system_question"] = llm_question or system_msg

    return InteractResponse(
        session_id=session_id,
        retrieved_images=retrieved_data,
        llm_question=llm_question,
        system_message=system_msg if not llm_question else None
    )





# --- Main execution for development ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    # Ensure LLM_API_KEY and LLM_BASE_URL are set as environment variables or in model_handler.py defaults
    # e.g. export LLM_API_KEY="your_key"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)