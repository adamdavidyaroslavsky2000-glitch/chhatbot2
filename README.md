# AI Chatbot System

מערכת צ'אטבוט מתקדמת עם מודלים מאומנים בשלבים מרובים. המערכת כוללת:

- **מודלים מאומנים**: בסיס + כללים דינמיים + לחץ חברתי + סינון טוקנים + שיחה
- **ממשק מודרני**: צ'אטבוט מינימליסטי כמו ChatGPT
- **API מתקדם**: FastAPI עם תמיכה מלאה בעברית
- **פריסה קלה**: הפעלה בפקודה אחת

## 🚀 הפעלה מהירה

```bash
# יצירת virtual environment
python3 -m venv chatbot_env
source chatbot_env/bin/activate

# התקנת חבילות
pip install -r backend/requirements.txt

# הפעלת הצ'אטבוט
python start_chatbot.py
```

הצ'אטבוט יהיה זמין ב: http://localhost:8000

## 🎨 תכונות הממשק

- **עיצוב מודרני**: דומה ל-ChatGPT ו-Gemini
- **סיידבר**: היסטוריית שיחות עם אפשרות לחזור לשיחות קודמות
- **רספונסיבי**: עובד מושלם על מובייל וטאבלט
- **אנימציות חלקות**: מעברים חלקים ואנימציות מקצועיות
- **תמיכה בעברית**: ממשק מלא בעברית ואנגלית
- **שמירת היסטוריה**: השיחות נשמרות בדפדפן

## 🏗️ ארכיטקטורת המערכת

### שלבי האימון
1. **Base Model** - מודל בסיס עם SentencePiece
2. **Dynamic Rules** - התאמה לכללים דינמיים  
3. **Social Pressure** - התאמה ללחץ חברתי
4. **Token Filter** - סינון טוקנים חשובים
5. **Conversation** - התאמה לשיחה טבעית

### רכיבי המערכת
- **Frontend**: HTML/CSS/JS מודרני עם עיצוב ChatGPT
- **Backend**: FastAPI עם מודלים מאומנים
- **Models**: TinyTransformerLM עם 600 טוקנים
- **Tokenizer**: SentencePiece מותאם אישית

## Contents

* **`token_filter_data/`** – 25 seed JSONL files (`filter_data_01.jsonl`
  … `filter_data_25.jsonl`). Each line contains a user sentence,
  lists of important and less‑important tokens, and a short
  explanation. These seeds can be expanded to much larger datasets
  using the provided script.
* **`scripts/expand_filter_data_to_30k.py`** – Expands each seed file
  to any target size (e.g. 30 000 lines) by duplicating entries and
  mutating the explanations for lexical variety.  Run this locally on
  your machine after extraction.
* **`train/datasets_token_filter.py`** – Dataset loader that turns
  the JSONL files into prompt/target pairs for supervised training.
* **`train/train_token_filter_adapter.py`** – Training script to fine
  tune your base model (or rules/usage adapter) on the expanded token
  filter datasets.
* **`inference/generate_with_filter.py`** – Inference script that
  applies simple heuristics to boost important tokens and penalise
  fillers at generation time.  Use this as a drop‑in replacement for
  your existing generator if you want a quick improvement without
  further training.
* **`data/anti_gibberish_sft.jsonl`** – A small anti‑gibberish
  supervised fine tuning (SFT) dataset.  Use this to perform a
  short adapter tuning step after rule/usage training to align the
  model with clear, concise English outputs.

## Quick Start

1. **Expand the seed datasets** (optional but recommended):

   ```bash
   python3 scripts/expand_filter_data_to_30k.py \
       --glob 'token_filter_data/filter_data_*.jsonl' \
       --target 30000
   ```

   This will rewrite each seed file with at least 30 000 entries.  Feel
   free to adjust the `--target` value.

2. **Fine tune the token filter adapter** on top of your existing
   model (e.g. after rule/usage adapters):

   ```bash
   python -m train.train_token_filter_adapter \
       --ckpt_in checkpoints/rules_adapter.pt \
       --ckpt_out checkpoints/token_filter_adapter.pt \
       --spm_model tokenizer/spm.model \
       --filter_glob 'token_filter_data/filter_data_*.jsonl' \
       --steps 6000 --batch_size 4 --device mps
   ```

   Adjust `--steps` and `--batch_size` according to your hardware.

3. **Optionally perform a short anti‑gibberish SFT** to
   stabilise English generation:

   ```bash
   python -m train.train_token_filter_adapter \
       --ckpt_in checkpoints/token_filter_adapter.pt \
       --ckpt_out checkpoints/token_filter_adapter_refined.pt \
       --spm_model tokenizer/spm.model \
       --filter_glob 'data/anti_gibberish_sft.jsonl' \
       --steps 2000 --batch_size 2 --device mps --limit_per_file 100
   ```

   This uses the same training script but feeds the anti‑gibberish SFT
   file as a dataset.  A few thousand steps suffice.

4. **Generate text** using the filtered generator:

   ```bash
   python -m inference.generate_with_filter \
       --ckpt checkpoints/token_filter_adapter_refined.pt \
       --spm_model tokenizer/spm.model \
       --prompt "Explain dropout for beginners in 5 bullets." \
       --topic machine_learning --format bullets \
       --controls_json '{"temperature":0.6, "top_p":0.9, "min_tokens":60, "max_tokens":180}' \
       --device mps
   ```

   You should see clean, on‑topic bullets without spurious meta or
   gibberish.

## Notes

* The token filter dataset is built from your existing training and
  validation corpora.  Feel free to add more sentences to those
  files before generating seeds to capture a wider variety of
  vocabulary and topics.
* The filter heuristic in `generate_with_filter.py` is simple.  For
  even better results you can parse the output of the trained token
  filter adapter to dynamically adjust token scores.  The provided
  script is a good starting point.
* All files are in English and avoid the use of any external API.
  Training and generation run entirely offline.
