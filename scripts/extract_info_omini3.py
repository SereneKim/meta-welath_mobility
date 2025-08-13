import logging
import json
import time
import openai as OpenAI
from dotenv import load_dotenv
import os
path = path = os.getcwd()

# Enable logging
logging.basicConfig(level=logging.INFO)

# Define OpenAI client
# OpenAI
load_dotenv() 
openai_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_API_KEY)

gpt_model = "o3-mini"

from pydantic import BaseModel

class GetInfo(BaseModel):
    Q1: str
    Q1_1: str
    Q2: str
    Q2_1: str
    Q2_2: str
    Q3: str
    Q4: str
    

BATCH_FILE = "batch_requests.jsonl"  # File for batch upload

# Function to generate batch data
def generate_batch_data(df):
    """Generates a JSONL file with batch requests."""
    with open(BATCH_FILE, "w") as file:
        for row in df.itertuples(index=False):
            user_prompt = f"""
            Consider this academic paper in the Social Sciences domain: 
            - Title: {row.title}
            - Year: {row.year}
            - Abstract: {row.abstract}
            - DOI: {row.doi}

            Use this information to answer the following questions:
            Q1. Does the paper discuss inter/multigenerational wealth/income/earning mobility? Yes or No.
            Q1_1. If 'yes' in Q1, give me a sentence or two about what the paper says. If 'no', what is the main focus?
            Q2. If 'yes' in Q1, does this paper mention measures to study inter/multigenerational mobility? Yes or No.
            Q2_1. If 'yes' in Q2, list the measures. If 'no', return 'N/A'.
            Q2_2. If 'no' in Q2, explain why. If 'yes', return 'N/A'.
            Q3. If 'yes' in Q2, does this paper empirically apply these measures to real data? Yes or No.
            Q4. If 'No' in Q3, does this paper theoretically study the measures? Yes or No.

            Answer concisely with 'Yes', 'No', or 'N/A' and brief explanations where required. Don't hallucinate.
            """

            batch_entry = {
                "messages": [
                    {"role": "system", "content": "Extract relevant data."},
                    {"role": "user", "content": user_prompt},
                ],
                "model": gpt_model,
                "response_format": GetInfo 
            }

            file.write(json.dumps(batch_entry) + "\n")  # Write request as JSONL format

# Step 1: Generate JSONL file for batch processing
df = pd.read_csv(f'{path}/data_abstracts/abstracts_citations_2025-03-05.csv')
generate_batch_data(df)
logging.info(f"Batch file {BATCH_FILE} created successfully.")

# Step 2: Submit batch job to OpenAI
batch_job = openai_client.batches.create(
    input_file=BATCH_FILE,  # File with batch requests
    endpoint="/v1/chat/completions",
    completion_window="24h"  # Adjust as needed
)

batch_id = batch_job.id
logging.info(f"Batch job submitted. Batch ID: {batch_id}")

# Step 3: Wait for batch job completion
while True:
    job_status = openai_client.batches.retrieve(batch_id)
    logging.info(f"Batch Status: {job_status.status}")
    
    if job_status.status == "completed":
        break  # Exit loop if batch is done
    
    time.sleep(30)  # Wait before checking again

# Step 4: Retrieve results
results_file = job_status.output_file
logging.info(f"Retrieving results from {results_file}")

results = openai_client.files.retrieve(results_file)
processed_results = [json.loads(line) for line in results.splitlines()]

# Step 5: Store or process results
logging.info(f"Processed {len(processed_results)} responses.")
for res in processed_results:
    res.to_csv(f'{path}/data_abstracts')  # Process as needed (e.g., save to a CSV)
