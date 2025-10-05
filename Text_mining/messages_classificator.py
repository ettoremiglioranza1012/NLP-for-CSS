# messages_classificator.py
# Enhanced classifier leveraging CUDA when available, safe CPU fallback
# - Rounds scores to 3 decimals
# - Preserves original post/comment ID if present
# - DOES NOT filter out records where text == "[removed]"

import os
import gc
import warnings

# ==== MODE SWITCH ============================================================
# Set SPILLOVER = 1 to read "comments_spillover.csv" and write
# "classification_results_spillover.csv". If 0, use the regular files.
SPILLOVER = 1
# ============================================================================

# ==== CONFIG ====
use_gpu = True  # enable GPU when available
# You can also set CLASSIFIER_BATCH_SIZE env var; defaults below will be used otherwise.
# ===============

# Windows stability toggles
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not use_gpu:
    os.environ["FORCE_CPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Using CPU mode for stability")
else:
    print("Auto-detecting best device (GPU preferred)")

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
import torch

from classification_model import ModelLoader


def safe_float_convert(value, default=0.0):
    """Clamp numeric value to [0,1] with safe conversion."""
    try:
        result = float(value)
        return max(0.0, min(1.0, result))
    except (ValueError, TypeError, OverflowError):
        return default


def normalize_model_output(result):
    """Normalize various model outputs into a list of dicts with expected keys."""
    if result is None:
        return [{"label": "non-hateful", "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
                 "threat": 0.0, "insult": 0.0, "identity_attack": 0.0}]
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return result
    return [{"label": str(result), "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
             "threat": 0.0, "insult": 0.0, "identity_attack": 0.0}]


def process_batch_safely(model, texts, max_retries=3):
    """Predict with retries and GPU cache cleanup on failure."""
    for attempt in range(max_retries):
        try:
            results = model.predict(texts, return_probability=True)
            return normalize_model_output(results)
        except Exception as e:
            print(f"Batch processing failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print("Using fallback results for failed batch")
                return [{"label": "non-hateful", "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
                         "threat": 0.0, "insult": 0.0, "identity_attack": 0.0}] * len(texts)


def main():
    print("=== Enhanced Detoxify Classification ===")

    # === Select input/output files based on SPILLOVER switch ===
    in_name = "comments_spillover.csv" if SPILLOVER == 1 else "comments_by_category.csv"
    out_name = "classification_results_spillover.csv" if SPILLOVER == 1 else "classification_results_by_category.csv"

    input_csv = os.path.join("../Database", in_name)
    output_csv = os.path.join("../Database", out_name)
    print(f"Mode: {'SPILLOVER' if SPILLOVER == 1 else 'STANDARD'}")
    print(f"Input CSV:  {input_csv}")
    print(f"Output CSV: {output_csv}")

    # === Load data ===
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        return

    try:
        print(f"📁 Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        print(f"✅ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # === Basic column checks ===
    required_cols = {"published_at", "text", "upvotes", "category"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ Error: Missing required columns: {sorted(missing_cols)}")
        return

    # NOTE: We intentionally DO NOT filter out rows where text == "[removed]".
    # If you want to monitor removal rates downstream, keep these rows.

    # Detect an ID column (keep post/comment reference)
    id_col_candidates = ["post_id", "id", "comment_id", "commentId", "comment_id_str"]
    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    if id_col is None:
        print("⚠️  Warning: no post/comment ID column found. (expected one of: "
              f"{', '.join(id_col_candidates)})")
        ids = [None] * len(df)
    else:
        print(f"🔗 Using ID column: {id_col}")
        ids = df[id_col].astype(str).tolist()

    # Prepare input lists
    timestamps = df["published_at"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()
    upvotes = df["upvotes"].fillna(0).astype(int).tolist()
    categories = df["category"].astype(str).tolist()

    print(f"🎯 Processing {len(texts):,} texts...")

    # === Choose device ===
    device_type = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    # === Load model ===
    try:
        print("🤖 Loading classification model...")
        loader = ModelLoader(device=device_type)
        model, _ = loader.load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # === Batch size ===
    if device_type == "cuda":
        batch_size = int(os.environ.get("CLASSIFIER_BATCH_SIZE", "256"))
        print(f"🚀 Using GPU batch size: {batch_size}")
    else:
        batch_size = int(os.environ.get("CLASSIFIER_BATCH_SIZE", "16"))
        print(f"💻 Using CPU batch size: {batch_size}")

    # === Accumulators ===
    labels = []
    toxicities = []
    severe_toxicities = []
    obscene_scores = []
    threat_scores = []
    insult_scores = []
    identity_attack_scores = []

    # === Predict in batches ===
    try:
        with tqdm(total=len(texts), desc="🔍 Classifying texts", unit="texts") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = process_batch_safely(model, batch_texts)

                for result in batch_results:
                    toxicity = safe_float_convert(result.get("toxicity", 0.0))
                    label = 1 if toxicity >= 0.5 else 0

                    toxicities.append(toxicity)
                    labels.append(label)
                    severe_toxicities.append(safe_float_convert(result.get("severe_toxicity", 0.0)))
                    obscene_scores.append(safe_float_convert(result.get("obscene", 0.0)))
                    threat_scores.append(safe_float_convert(result.get("threat", 0.0)))
                    insult_scores.append(safe_float_convert(result.get("insult", 0.0)))
                    identity_attack_scores.append(safe_float_convert(result.get("identity_attack", 0.0)))

                pbar.update(len(batch_texts))

                if i % (batch_size * 5) == 0 and i > 0:
                    gc.collect()
                    if device_type == "cuda":
                        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return

    # === Length sanity check ===
    n = len(texts)
    if len(labels) != n:
        print(f"⚠️ Warning: Results length mismatch. Expected {n}, got {len(labels)}")
        while len(labels) < n:
            labels.append(0)
            toxicities.append(0.0)
            severe_toxicities.append(0.0)
            obscene_scores.append(0.0)
            threat_scores.append(0.0)
            insult_scores.append(0.0)
            identity_attack_scores.append(0.0)

    # === Build output DataFrame ===
    try:
        output_df = pd.DataFrame({
            (id_col if id_col else "post_id"): ids,
            "published_at": timestamps,
            "upvotes": upvotes,
            "category": categories,
            "toxicity": toxicities,
            "label": labels,
            "severe_toxicity": severe_toxicities,
            "obscene": obscene_scores,
            "threat": threat_scores,
            "insult": insult_scores,
            "identity_attack": identity_attack_scores,
            "text": texts
        })

        # Round numeric scores to 3 decimals
        score_cols = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
        output_df[score_cols] = output_df[score_cols].round(3)

        # Save results
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        output_df.to_csv(output_csv, index=False)

        print(f"\n✅ Successfully saved {len(output_df):,} rows to {output_csv}")

        toxic_count = int(sum(labels))
        toxic_percentage = 100 * toxic_count / len(labels) if len(labels) > 0 else 0

        # Console summary
        print(f"\n📊 Classification Summary:")
        print(f"   📝 Total messages: {len(labels):,}")
        print(f"   ☠️  Toxic messages: {toxic_count:,} ({toxic_percentage:.1f}%)")
        print(f"   📈 Average scores:")
        print(f"      • Toxicity: {output_df['toxicity'].mean():.3f}")
        print(f"      • Severe toxicity: {output_df['severe_toxicity'].mean():.3f}")
        print(f"      • Obscene: {output_df['obscene'].mean():.3f}")
        print(f"      • Threat: {output_df['threat'].mean():.3f}")
        print(f"      • Insult: {output_df['insult'].mean():.3f}")
        print(f"      • Identity attack: {output_df['identity_attack'].mean():.3f}")
        print(f"   🖥️  Device used: {device_type.upper()}")

        if 'category' in output_df.columns:
            print(f"\n📂 Toxicity by category:")
            category_stats = output_df.groupby('category').agg({
                'label': ['count', 'sum'],
                'toxicity': 'mean'
            }).round(3)
            for category in category_stats.index:
                total = int(category_stats.loc[category, ('label', 'count')])
                toxic = int(category_stats.loc[category, ('label', 'sum')])
                avg_tox = float(category_stats.loc[category, ('toxicity', 'mean')])
                toxic_pct = (toxic / total) * 100 if total > 0 else 0
                print(f"      • {category}: {toxic}/{total} ({toxic_pct:.1f}%) - avg: {avg_tox:.3f}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        import traceback
        traceback.print_exc()

    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    print("Memory cleaned up")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Process completed")
