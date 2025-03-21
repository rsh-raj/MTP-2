import os
import re
import time
import logging
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
from datasets import load_dataset

# Load API Key from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# # Logging
# logging.basicConfig(filename="gemini_inference.log", level=logging.INFO, 
#                     format="%(asctime)s - %(levelname)s - %(message)s")

# Prompt Template
prompt = """Below is an instruction describing a task and an input providing context. Write a commit message that appropriately summarizes the changes in the code diff.

### Instruction:
You are commit message description generator specializing in Conventional Commits. Given a commit body and code diff, generate a description which
is basically a short summary of the code changes. Only generate one line commit description, you don't need to generate type and body.
###  Example:
- Commit Body:
    Allow the querier's circuit breaker thresholds to be configured in test
    runs - this helps speed up tests that involve hitting offline ingesters.
- Code Diff:
    diff --git a/test_helpers_end_to_end/src/config.rs b/test_helpers_end_to_end/src/config.rs
    index eaac901727..66feeec99c 100644
    --- a/test_helpers_end_to_end/src/config.rs
    +++ b/test_helpers_end_to_end/src/config.rs
    @@ -155,6 +155,16 @@ impl TestConfig {{
            Self::new(ServerType::AllInOne, dsn, random_catalog_schema_name()).with_new_object_store()
        }}
    
    +    /// Set the number of failed ingester queries before the querier considers
    +    /// the ingester to be dead.
    +    pub fn with_querier_circuit_breaker_threshold(self, count: usize) -> Self {{
    +        assert!(count > 0);
    +        self.with_env(
    +            "INFLUXDB_IOX_INGESTER_CIRCUIT_BREAKER_THRESHOLD",
    +            count.to_string(),
    +        )
    +    }}
    +
        /// Configure tracing capture
        pub fn with_tracing(self, udp_capture: &UdpCapture) -> Self {{
            self.with_env("TRACES_EXPORTER", "jaeger")
- Description is: 
    configurable querier circuit breakers

### Input:
- Commit Body:
{}
- Code Diff:
{}

"""

# Load dataset
test_dataset = load_dataset("rsh-raj/influxdb-with-body", split="test")

# Output File
OUTPUT_FILE = "gemini_output.txt"

# Rate limit handling
max_requests_per_minute = 13
request_counter = 0
start_time = time.time()

# Open file to write results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, sample in enumerate(test_dataset):
        print(i,end=" ")

        commit_body = sample["Body"][:500] if sample["Body"] else ""
        description = sample["Description"][:100] if sample["Description"] else ""
        git_diff = ""

        if sample["Git Diff"] is not None:
            git_diff = sample["Git Diff"][:1900 - (len(commit_body) - len(description))]


        # Rate limit check
        if request_counter >= max_requests_per_minute:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                # print(f"Sleeping for {sleep_time:.2f} seconds to avoid rate limit...")
                time.sleep(sleep_time)

            # Reset counter after sleep
            request_counter = 0
            start_time = time.time()

        # Constructing prompt
        input_prompt = prompt.format(commit_body, git_diff)

        try:
            # Generate response from Gemini
            response = completion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": input_prompt}],
            )

            # Extract response text
            output_text = response.choices[0].message.content if response.choices else ""
            # print(output_text)

            # Extract only the generated description using regex
            match = re.search(r"^[a-zA-Z]+(?:\([^)]*\))?:\s*(.*)", output_text)     
            extracted_text = match.group(1) if match else ""
            if len(extracted_text) <=2:
                print(output_text)
                f.write(f"{output_text}\n")

            else:
                print(extracted_text)
                f.write(f"{extracted_text}\n")

            # Save to file

            # Log success
            # logging.info(f"Successfully processed {i}: {extracted_text}")

        except Exception as e:
            # logging.error(f"Error processing {i}: {str(e)}")
            f.write("\n")  # Write empty result in case of failure

        request_counter += 1
        time.sleep(4)

print(f"✅ Inference complete. Results saved in '{OUTPUT_FILE}'.") 