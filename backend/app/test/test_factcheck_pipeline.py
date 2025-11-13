"""
test_complete_pipeline.py - Test complete fact-checking pipeline
"""

import json
import os
# hide the TF CPU message 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# print("Current working directory:", os.getcwd())

from app.core.public_config import PublicConfig
from app.models.claim import Claim
from app.models.evidence import Evidence
from app.models.factcheck import FactCheckResponse
from app.features.factcheck_pipe import FactCheckPipe
from app.utils.configuration_syncing import sync

# --- test the config for testing ---
test_cfg = PublicConfig(
    # embedding_model_name="all-MiniLM-L12-v2",
    embedding_model_name="paraphrase-mpnet-base-v2",
    claim_confidence_threshold=0.7,
    evidence_top_k=3,
    evidence_min_similarity=0.4,
    supports_threshold=0.75
)
sync(test_cfg)

# --- create text for claim ---
text = "The Eiffel Tower is in Paris. It is very tall."
pipe = FactCheckPipe()

# --- run the process ----------
response = pipe.process(text)

# --- Step 7: Print results ---
print(json.dumps(response.dict(), indent=2))
