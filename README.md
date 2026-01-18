# Reddit Toxicity Analysis on the Israeli-Palestinian Conflict

This repository contains the code, data processing scripts, and analysis related to a study of online toxicity on Reddit, focusing on discussions surrounding the Israeli-Palestinian conflict. The study investigates how conflict narratives influence toxic reactions in online discourse and whether these reactions spill over into apolitical contexts, such as sports and culture.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data Collection](#data-collection)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Content Classification](#content-classification)  
5. [Narrative Framing and Analysis](#narrative-framing-and-analysis)  
6. [Spillover Analysis](#spillover-analysis)  

---

## Project Overview

This project examines the emergence and spread of toxic language on Reddit during periods of heightened conflict in Gaza and over a longitudinal period of 2015–2025. Key objectives include:

- Understanding how **narrative framing** (military, humanitarian, political) influences toxic reactions.  
- Investigating the **spillover of toxicity** into apolitical online discussions, particularly in sports and cultural subreddits.  

The analysis combines data collection from Reddit, natural language processing, and machine learning-based toxicity classification.

---

## Data Collection

Data was collected using Reddit's official API. Posts and their comment trees were retrieved based on predefined queries, covering:

- **Conflict period**: March 18–25, 2025, coinciding with intensified military operations in Gaza.  
- **Longitudinal period**: January 2015 – August 2025, to observe the evolution of conflict-related hostility in non-political discussions.

Comments were filtered to retain only those with positive engagement (upvotes), and metadata was stored, including:

- Post ID  
- Title  
- Publication time  
- Positive votes  
- Assigned narrative category  

### Narrative Categories

1. **Israeli Military Actions** – Posts referring to air strikes, ground incursions, or tactical operations.  
2. **Humanitarian Narrative** – Posts focusing on civilian casualties, hospital attacks, or displacement crises.  
3. **Political Representation** – Posts discussing official reactions, diplomacy, and institutional statements.  
4. **Apolitical/Spillover** – Posts in sports and cultural subreddits mentioning Israeli or Jewish identities.

Boolean queries were carefully constructed to capture relevant discussions while excluding low-signal content (e.g., routine match threads).

---

## Data Preprocessing

- Recursive extraction of full comment trees  
- Removal of low-quality comments  
- Metadata enrichment for each comment  
- Categorization according to narrative and thematic context  

---

## Content Classification

The **Detoxify** library (RoBERTa-based transformer) was used to classify comments across six toxicity dimensions:

- Toxic  
- Severely toxic  
- Obscene  
- Threatening  
- Insulting  
- Identity attack  

The **unbiased version** of Detoxify was applied to minimize systematic misclassification of identity-related terms. Continuous probability outputs were used to capture nuanced expressions of hostility, rather than binary labels.

Systematic observations:

- Profanities and vulgar markers increase toxicity scores, even when not directed at any group.  
- Geopolitical vocabulary may inflate toxicity scores due to co-occurrence with hostile discourse in training data.  
- Sarcasm, irony, and rhetorical framing are often missed by transformer-based models, highlighting the challenge of detecting implicit hostility.

---

## Narrative Framing and Analysis

- Principal Component Analysis (PCA) identified three latent dimensions of hostility:  
  1. **Verbal abuse and profanity**  
  2. **Explicit threats**  
  3. **Identity-based attacks**  

- Analysis revealed different toxicity patterns depending on narrative framing:  
  - Humanitarian narratives → verbal abuse and profanity  
  - Political narratives → explicit threats  
  - Military and humanitarian narratives → identity-based hostility  

Visualizations (Figures 4.4–4.6) show normalized distributions of these dimensions across narrative frames.

---

## Spillover Analysis

- Toxicity in apolitical contexts (sports and cultural subreddits) was assessed annually (2015–2025).  
- Sports-related discussions consistently exhibited higher toxicity than cultural discussions.  
- Increases in toxicity correlated with periods of escalated conflict, suggesting the diffusion of conflict-driven hostility into otherwise neutral spaces.  
- Figures 4.7–4.8 visualize annual averages and a three-year moving average of composite toxicity scores.
