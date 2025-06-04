# Interactive Text-to-Image Retrieval Website


This project uses the BLIP model with LLM API to extract and retrieve image contexts based on text queries, leveraging image embeddings and dialogue-based retrieval.

## Quick Start

### Prerequisites
- Python 3.8+
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure you have a [Visdial 2018 validation image dataset](https://visualdialog.org/data) on your local directory


### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/freshfish15/interactive_retrieval.git
   cd interactive_retrieval
   ```

2. Configure the `model_handler.py` file:
   - Update the paths for your image directory:
     ```python
     IMAGE_BASE_DIRECTORY = "your/path/to/VisualDialog_val2018"
     ```
   - Set your API key and base URL:
     ```python
     LLM_API_KEY = "your-api-key"
     LLM_BASE_URL = "your-base-url"
     ```
   - Customize the query and retrieval parameters:
     ```python
     query = "your text query here"  # e.g., "people playing sports"
     num_top_similar = 16  # Number of initial similar images
     num_representatives = 8  # Number of representative images
     dialogue_round = 2  # Number of dialogue rounds
     ```

### Running the Project
Run the main script:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001
cd frontend
python python3 -m http.server 8081
```

