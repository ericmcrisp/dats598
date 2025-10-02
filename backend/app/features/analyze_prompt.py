import spacy
import random
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")


def analyze_prompt(text: str) -> dict:
    doc = nlp(text)
    blob = TextBlob(text)

    # init
    sent_type = "Unknown"

    # get components
    entities = [ent.text for ent in doc.ents]
    main_verb = next((t.text for t in doc if t.pos_ == "VERB"), None)

    # classify sentence
    if text.endswith("?"):
        sent_type = "Question"
    elif text.endswith("!"):
        sent_type = "Exclamation"
    elif doc[0].pos_ == "VERB":
        sent_type = "Command"
    else:
        sent_type = "Statement"

    # categorize it as a fact or not (placeholder)
    is_factual = random.random() < 0.5

    # THIS is what the frontend will be recieving
    return {
        "sentence_type": sent_type,
        "is_factual": is_factual,
        "subjectivity": blob.sentiment.subjectivity,
        "entities": entities,
        "main_verb": main_verb
    }
