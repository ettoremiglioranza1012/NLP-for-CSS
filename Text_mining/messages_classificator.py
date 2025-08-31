import os
import pandas as pd
from tqdm import tqdm
from classification_model import ModelLoader


def main():
    # load the TweetNLP hate detection model
    loader = ModelLoader()  # no path parameter needed anymore
    model, tokenizer = loader.load_model()  # tokenizer will be None for TweetNLP
    device = loader.device

    # read your scraped comments
    input_csv = os.path.join("../Database", "comments_category1.csv")
    df = pd.read_csv(input_csv)

    # extract fields
    timestamps = df["published_at"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    upvotes = df["upvotes"].fillna(0).astype(int).tolist()

    # classify with progress bar
    # TweetNLP processes texts individually, so we'll iterate through them
    labels = []
    total_texts = len(texts)
    pbar = tqdm(total=total_texts, desc="Classifying messages")

    for text in texts:
        # Use TweetNLP model to classify
        result = model.predict(text)
        # Extract the predicted label (assuming it returns a structured result)
        # TweetNLP typically returns the label with highest confidence
        if isinstance(result, dict):
            # If result contains 'label' key
            if 'label' in result:
                # Convert label to numeric if needed (adjust based on your needs)
                # For hate detection: typically 'hate' vs 'not_hate' or similar
                label = 1 if result['label'].lower() in ['hate', 'hateful', 'offensive'] else 0
            else:
                # If result is a list of predictions, take the first one
                label = 1 if str(result).lower() in ['hate', 'hateful', 'offensive'] else 0
        else:
            # If result is a simple string/label
            label = 1 if str(result).lower() in ['hate', 'hateful', 'offensive'] else 0

        labels.append(label)
        pbar.update(1)

    pbar.close()

    # write results.csv with published_at, upvotes, label, and text (in that order)
    result_df = pd.DataFrame({
        "published_at": timestamps,
        "upvotes": upvotes,
        "label": labels,
        "text": texts  # text as last field
    })
    result_df.to_csv("../Database/classification_results.csv", index=False)
    print(f"✅ Saved {len(result_df)} rows to classification_results.csv")


if __name__ == "__main__":
    main()