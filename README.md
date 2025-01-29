# Nuitee – Room Matching Project

## Table of Contents
- [Project Description](#project-description)
- [Datasets Explanation](#datasets-explanation)
- [How to Run](#how-to-run)
- [Data Preprocessing & Feature Extraction for Room Matching](#data-preprocessing--feature-extraction-for-room-matching)
- [Model Development](#model-development)
- [API Development](#api-development)
- [Testing](#testing)
- [Next Steps for Scalability](#next-steps-for-scalability)

---

## Project Description
This is a comprehensive solution for matching hotel room information between different suppliers. It aims to handle messy, inconsistent, or differently labeled room names by normalizing and extracting key features, then computing similarity and providing a robust matching mechanism.

## Datasets Explanation
There are two main datasets in this project:

1. **reference_rooms.xlsx**
   - Contains reference (canonical) hotel room information
   - Columns: `hotel_id`, `lp_id`, `room_id`, `room_name`
   - Serves as the "reference catalog" for matching

2. **updated_core_rooms.csv**
   - Contains supplier (input) room information
   - Columns: `core_room_id`, `core_hotel_id`, `lp_id`, `supplier_room_id`, `supplier_name`, `supplier_room_name`
   - Serves as the "input catalog" for matching

## How to Run

1. **Clone the Repository & Install Requirements**
   ```bash
   git clone <repo-url>
   cd nuitee
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   # or
   venv\Scripts\activate  # For Windows

   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   - Copy `.env.example` to `.env` or create a new `.env` file.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=<YOUR-OPENAI-KEY>
     ```

3. **Data Setup**
   ```bash
   # Create needed directories
   mkdir -p data/raw data/processed logs
   ```
   - Download the reference room file [here](https://docs.google.com/spreadsheets/d/1p-sqMUdtqFQJGc6v9tzqETIuxu8Gpf7r/edit?usp=drive_link&ouid=100361813368908650921&rtpof=true&sd=true) and place it in `data/raw` as `reference_rooms.xlsx`
   - Download the supplier room file [here](https://drive.google.com/file/d/1Nrh9_8xyidzMkbLH52RIZZZ2mU-30rAe/view?usp=drive_link) and place it in `data/raw` as `supplier_rooms.csv`

4. **Run Batch Processing** (data cleanup, optional benchmark creation, etc.)
   ```bash
   python batch_process.py \
       --reference_file="data/raw/reference_rooms.xlsx" \
       --supplier_file="data/raw/supplier_rooms.csv" \
       --output_dir="data/processed" \
       --generate_benchmark=True \
       --benchmark_size=100 \
       --noise_level=0.2
   ```
5. **Run the API**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Test the Endpoints**
   - Go to `http://localhost:8000/docs` to see the automatically generated Swagger UI.
   - Run the health check endpoint:
     ```bash
     curl http://localhost:8000/health
     ```

### Using Docker

1. **Make scripts executable** (if needed):
   ```bash
   chmod +x scripts/run_tests.sh scripts/cleanup.sh
   ```

2. **Build and start services**:
   ```bash
   # Build the images
   docker-compose build

   # Start the services
   docker-compose up
   ```
3. **Run just the batch process**:
   ```bash
   docker-compose run --rm app python batch_process.py
   ```
4. **Run the API service**:
   ```bash
   docker-compose up room-matching
   ```
5. **Run tests**:
   ```bash
   ./scripts/run_tests.sh
   ```
6. **View logs**:
   ```bash
   # View all logs
   docker-compose logs

   # View specific service logs
   docker-compose logs app
   docker-compose logs api

   # Follow logs
   docker-compose logs -f
   ```
7. **Clean up**:
   ```bash
   # Stop services
   docker-compose down

   # Clean up resources
   ./scripts/cleanup.sh
   ```

### Project Structure (Example)

```
nuitee/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .dockerignore
├── scripts/
│   ├── run_tests.sh
│   └── cleanup.sh
├── data/
│   ├── raw/
│   │   ├── reference_rooms.xlsx
│   │   └── supplier_rooms.csv
│   └── processed/
├── logs/
├── src/
│   └── ...
├── tests/
│   └── ...
├── batch_process.py
└── main.py
```

**To develop with Docker**:

- Start development environment:
  ```bash
  # Start services in development mode
  docker-compose up -d

  # View logs while developing
  docker-compose logs -f
  ```
- Make changes and test:
  ```bash
  # Run tests
  ./scripts/run_tests.sh

  # Run specific test file
  ./scripts/run_tests.sh tests/unit/test_feature_extractor.py
  ```
- Rebuild after changes:
  ```bash
  # Rebuild services
  docker-compose build

  # Restart services
  docker-compose restart
  ```
- Access the API:
  ```bash
  # API will be available at:
  http://localhost:8000/docs
  ```

## Data Preprocessing & Feature Extraction for Room Matching

Below is a practical, step-by-step guide for **data preprocessing** and **feature extraction** on your `room_name` and `supplier_room_name` fields before they are passed into the matching pipeline. These steps help standardize text data, ensure consistency, and allow the model (and optional LLM-based NER) to yield better accuracy.

---

### 1. Data Cleaning

1. **Lowercasing**
   - Convert all text to lowercase to ensure case-insensitive comparisons.
   - Example: "Classic Room ADA – ROOM ONLY" → "classic room ada – room only"
2. **Remove Extra Whitespace & Non-ASCII Characters**
   - Trim leading/trailing spaces.
   - Replace multiple spaces with a single space.
   - Remove or normalize any unusual characters (e.g., "\t", "\n", or exotic Unicode punctuation).
   - Example: " superior room " → "superior room"
3. **Remove or Replace Punctuation / Symbols**
   - Common punctuation like commas, semicolons, periods, hyphens, slashes, parentheses can often be normalized.
   - Decide whether to remove them or turn them into spaces. (In many cases, replacing them with spaces or a delimiter helps tokenization.)
   - Example: "Deluxe/Double - 2 Pax!" → "deluxe double 2 pax"
4. **Normalize Common Abbreviations and Typos**
   - Standardize known short forms:
     - "rm" → "room"
     - "ro" → "room only" or "room-only"
     - "nr" → "non-refundable"
     - "ada" → "accessible"
   - Example: "CLASSIC ROOM ADA - RO" → "classic room accessible room-only"
5. **Strip Stopwords That Add No Meaning** (Optional)
   - Words like "and", "with", "the", "of", "a", etc. can be removed if they don’t carry domain-specific meaning.
   - However, be mindful that words like "non-smoking" might be considered an important attribute.

---

### 2. Domain-Specific Normalization

For **hotel rooms**, certain repeated concepts are valuable to parse or normalize. These often map well to the entities we want to extract:

1. **Bed Types**
   - Examples: "king bed", "queen bed", "twin bed", "double bed", etc.
   - Normalize synonyms or slightly different expressions.
   - E.g., detect strings like "1 king", "king-size", or "king-sized" and convert them to something consistent like "king-bed".
2. **View Types**
   - Examples: "city view", "sea view", "courtyard view".
   - Convert to "city-view", "sea-view", etc. if you want consistent tokens.
3. **Room Class / Category**
   - Terms like "standard", "superior", "deluxe", "premium", "junior suite", "suite".
   - Create a mapping if you prefer shorter or uniform tokens (e.g., "junior suite" → "jr-suite").
4. **Occupancy / Number of People**
   - "double", "triple", "quadruple", or mentions like "2 pax" or "3 adults".
   - Convert to a single token if needed ("2pax", "3pax", etc.).
5. **Smoking / Non-Smoking**
   - Terms like "non-smoking", "smoking", "smoke-free".
   - Convert them to "non-smoking" vs. "smoking".
6. **Board / Meal Plans**
   - "room only", "breakfast included", "half board", "full board", "all inclusive".
   - Standardize them: "room-only", "breakfast-included", "half-board", etc.
7. **Refundability / Cancellation Policy**
   - "non-refundable", "free cancellation", etc.
   - Standardize if they appear as part of the naming convention.
8. **Other Room Attributes**
   - "ADA", "accessible", "balcony", "terrace", etc.

---

### 3. Tokenization & Feature Extraction

1. **Tokenize**
   - Split the cleaned string by spaces (or other delimiters) into tokens.
   - Example: "classic room accessible room-only" → ["classic", "room", "accessible", "room-only"]
2. **Domain Tagging / Parsing** (optional LLM-based NER)
   - Instead of building a custom parser or using simple regex to identify bed type, room type, etc., you can leverage a **prompt-based approach** with an LLM.
   - Example prompt to LLM:
     ```
     "Extract the following domain-specific attributes from this room description:
     1) bedType
     2) roomType
     3) viewType
     4) boardType
     5) accessibility
     6) occupancy

     The text is: 'classic room accessible room-only'
     "
     ```
   - Parse the structured response (JSON or bullet points) to fill out fields:
     ```json
     {
       "bedType": null,
       "roomType": "classic",
       "viewType": null,
       "boardType": "room-only",
       "accessibility": "accessible",
       "occupancy": null,
       "additionalAttributes": []
     }
     ```
3. **Use Model or `run_ner`**
   - If you have a built-in NER routine, it will parse text for bed type, room class, etc.
   - Pre-normalizing or partially tagging data helps if original text is inconsistent.

---

### 4. Example End-to-End Preprocessing

Suppose you have:
- **Reference Room Name**: "Classic Room - Olympic Queen Bed - ROOM ONLY"
- **Supplier Room Name**: "Classic Room Ada R.O."

A possible transformation pipeline:

1. **Lowercase**:
   - "classic room - olympic queen bed - room only"
   - "classic room ada r.o."
2. **Remove / Replace punctuation**:
   - "classic room olympic queen bed room only"
   - "classic room ada ro"
3. **Normalize abbreviations**:
   - "ada" → "accessible"
   - "ro" → "room-only"
   - Results:
     - "classic room olympic queen bed room-only"
     - "classic room accessible room-only"
4. **Domain synonyms** (optional):
   - "olympic queen bed" → "queen-bed"
   - Final tokens:
     - ["classic", "room", "queen-bed", "room-only"]
     - ["classic", "room", "accessible", "room-only"]
5. **(Optional) LLM Prompt**:
   - Ask the LLM to classify or label the text with bed type, room type, etc.
6. **Send Through Matching Pipeline**:
   - Provide the cleaned (or tokenized) strings to the matching system.

---

### 5. Putting It All Together

When you assemble data for the matching engine, ensure that each `roomName` or `supplierRoomName` is consistently cleaned and normalized. If your engine supports an NER parameter, enable it as desired. This helps ensure the best results from the text-similarity-based approach.

---

### 6. Summary of Steps

1. **Read** your Excel/CSV files for reference and supplier data.
2. **Clean & Normalize** each `room_name`:
   - Lowercase, remove punctuation/spaces, map abbreviations.
   - (Optional) remove or unify stopwords.
3. **Apply Domain-Specific Rules**
   - Bed type, view, room class, occupancy, etc.
4. **Tokenize** and (optional) **run LLM-based NER**
5. **Construct** the final dataset or request payload for matching.
6. **Analyze** results and refine normalization rules.

By replacing ad-hoc regex solutions with a more robust pipeline and (optionally) LLM-based extraction, you can handle more variations in room naming while preserving accuracy.

## Model Development
1. **Similarity Calculation**
   - Combines text similarity (cosine TF-IDF, Jaccard overlap) and feature-level matching.
   - Weighted approach: 40% feature similarity, 30% text similarity, 30% token overlap.
2. **Threshold Tuning**
   - We evaluate different thresholds (e.g., 0.5–0.8) to choose the best F1 score.

## API Development
1. **Endpoints**
   - `/process_rooms`: Clean and extract features from given room data.
   - `/match_rooms`: Match reference rooms to supplier rooms based on the similarity approach.
   - `/health`: Check the API health.
2. **Main Libraries**
   - `FastAPI` for building the web service.
   - `pydantic` models for request/response validation.

## Testing
1. **Unit Tests**: Tests for cleaners, feature extraction, etc.
2. **Integration Tests**: End-to-end tests calling the FastAPI endpoints.
3. **Benchmarking**: Synthetic dataset is used to measure precision, recall, F1.

## Next Steps for Scalability
1. **Blocking & Candidate Selection**
   - For large catalogs, do approximate blocking or hashing to reduce the candidate pool.
   - Then run the current approach only on the candidate set.
2. **LLM-Generated Training Data**
   - Fine-tune a specialized NER or classification model on LLM-labeled data.
   - Improves performance and reduces reliance on direct LLM calls in production.
3. **Celery & Redis for Task Queue**
   - Handle larger, parallelizable tasks by pushing them into a distributed queue.
   - Each batch of room pairs is processed independently, and results are aggregated.
4. **Microservices**
   - Eventually break out the matching engine into a separate microservice.
   - Each service can scale independently.
5. **Versioning and Observability with Langfuse & MLflow**
   - Use **Langfuse** for prompt versioning and analyzing prompt performance over time, allowing you to iterate on your LLM prompts effectively.
   - Employ **MLflow** to track experiment runs, model versions, and performance metrics, ensuring reproducibility and easier rollback to older model iterations.
