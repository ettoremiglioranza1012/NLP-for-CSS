import os
import torch
import pandas as pd
from tqdm import tqdm
from data_classification import ModelLoader


def main():
    # load the fine-tuned RoBERTa model and tokenizer
    loader = ModelLoader(path="./RobWgs")  # adjust if your model dir is elsewhere
    model, tokenizer = loader.load_model()
    model.eval()
    device = loader.device

    # read your scraped comments
    input_csv = os.path.join("Database", "comments_category1.csv")
    df = pd.read_csv(input_csv)

    # extract fields
    timestamps = df["published_at"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    upvotes = df["upvotes"].fillna(0).astype(int).tolist()

    # classify in batches with progress bar
    batch_size = 32
    labels = []
    total_texts = len(texts)
    pbar = tqdm(total=total_texts, desc="Classifying messages")

    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        labels.extend(preds)
        pbar.update(len(batch_texts))

    pbar.close()

    # write results.csv with published_at, upvotes, label, and text (in that order)
    result_df = pd.DataFrame({
        "published_at": timestamps,
        "upvotes": upvotes,
        "label": labels,
        "text": texts  # text as last field
    })
    result_df.to_csv("Database/classification_results.csv", index=False)
    print(f"✅ Saved {len(result_df)} rows to classification_results.csv")


if __name__ == "__main__":
    main()
