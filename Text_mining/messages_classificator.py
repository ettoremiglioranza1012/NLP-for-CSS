import os
import pandas as pd
from tqdm import tqdm
from classification_model import ModelLoader


def main():
    # --- Variabile da cambiare: category1 | spillover | ecc. ---
    title = "spillover"

    # load the TweetNLP hate detection model
    loader = ModelLoader()
    model, tokenizer = loader.load_model()
    device = loader.device

    # read your scraped comments
    input_csv = os.path.join("../Database", f"comments_{title}.csv")
    df = pd.read_csv(input_csv)

    # extract fields
    timestamps = df["published_at"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    upvotes = df["upvotes"].fillna(0).astype(int).tolist()

    # classify with progress bar
    labels = []
    total_texts = len(texts)
    pbar = tqdm(total=total_texts, desc=f"Classifying messages ({title})")

    for text in texts:
        result = model.predict(text)

        if isinstance(result, dict):
            if 'label' in result:
                label = 1 if result['label'].lower() in ['hate', 'hateful', 'offensive'] else 0
            else:
                label = 1 if str(result).lower() in ['hate', 'hateful', 'offensive'] else 0
        else:
            label = 1 if str(result).lower() in ['hate', 'hateful', 'offensive'] else 0

        labels.append(label)
        pbar.update(1)

    pbar.close()

    # write classification results
    output_csv = os.path.join("../Database", f"classification_results_{title}.csv")
    result_df = pd.DataFrame({
        "published_at": timestamps,
        "upvotes": upvotes,
        "label": labels,
        "text": texts
    })
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(result_df)} rows to {output_csv}")


if __name__ == "__main__":
    main()
