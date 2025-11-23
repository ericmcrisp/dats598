from fastcoref import FCoref

def main():
    print("Loading model...")
    model = FCoref(device="cpu")  # change to "cuda" if you have GPU

    text = (
        "Barack Obama was born in Hawaii. "
        "He was elected president in 2008."
    )

    print("\nInput:")
    print(text)

    print("\nRunning coreference resolution...")
    preds = model.predict(texts=[text])

    resolved = preds[0].get_resolved_utterance()

    print("\nResolved:")
    print(resolved)

if __name__ == "__main__":
    main()
