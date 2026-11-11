# Replacing a closed-model API for news summaries with Aya Expanse 8B

I wanted to see whether a small open model could take over a high-volume job that teams usually send to a big closed model: summarising news articles in three short sentences. I used Aya Expanse 8B, cleaned the training data with CuratorKIT, trained with AlignTune and scored everything with AuditKIT. Everything ran on free Kaggle T4 GPUs.

## Task and data

The task is one prompt, used everywhere: "Summarize this news article in 3 short sentences", followed by the article. I used CNN/DailyMail, because every article comes with human-written highlights that work as reference summaries, and because summarisation at volume is a common reason people pay for closed-model APIs.

I took 8,000 training articles and ran them through CuratorKIT:

- 27 rows dropped because the highlights were not 3 or 4 lines, so they did not match the task
- 4,627 rows dropped for length (under 120 or over 560 words). Long articles get cut off at training time, which leaves the model learning summaries of text it never saw
- 335 exact duplicates and 13 near-duplicates removed
- 2,998 rows kept

I was surprised by the duplicates. Over 4% of a widely used dataset was repeated.

## Training

Two runs with identical settings, one on 640 random raw rows and one on 640 curated rows. Same model, same steps, same seed, so the data is the only difference.

AlignTune with the TRL backend, QLoRA (4-bit NF4), LoRA r=16, alpha=32 on q/k/v/o, 160 steps, batch 1 with 4 accumulation steps, lr 2e-4, max length 1024, fp16, on 2x T4. Each run took about 2.5 hours.

## Metrics and why

I scored 50 test articles the model never saw, with AuditKIT:

- ROUGE-L: how much of the reference summary's wording and order the model reproduces. The standard summarisation metric.
- BLEU: stricter phrase overlap, which rewards getting the exact wording.
- chrF: character-level overlap, which is more forgiving of small wording changes.
- Refusal rate on 10 clearly harmful requests, because fine-tuning is known to sometimes weaken safety behaviour.

## Results

| model | ROUGE-L | BLEU | chrF | refused harmful requests |
|---|---|---|---|---|
| Aya Expanse 8B, base | 0.235 | 0.075 | 0.407 | 10/10 |
| fine-tuned on raw data | 0.275 | 0.075 | 0.406 | 10/10 |
| fine-tuned on curated data | 0.275 | 0.080 | 0.406 | 10/10 |
| Command A (closed, via API) | 0.137 | 0.015 | 0.278 | not tested |

What I take from this:

1. Fine-tuning works. ROUGE-L went up 17% and the summaries now follow the target format: short, factual, one point per line.
2. Safety held. Both fine-tuned models refused all 10 harmful requests, the same as the base model.
3. Curation did not beat raw data here. ROUGE-L is the same, and the small BLEU gain is within noise for 50 articles. My guess is that CNN/DailyMail is already fairly clean, so the main value of curation was catching the duplicates and the truncated examples, not a higher score. I'd expect a bigger difference on messier or generated data.
4. Command A scores much lower. It writes good summaries, but longer ones (about 100 tokens against 57 for the fine-tuned model) in its own style. So the result is not that the small model is smarter. It is that it produces the exact format you asked for, every time, without prompt engineering.

## Cost

For 100,000 summaries a month:

- Command A: about 787 input and 102 output tokens per article. At $2.50 and $10 per million tokens, that is about $299 a month.
- Self-hosted fine-tuned Aya on one T4 ($0.35/hour): 7.3 seconds per summary, measured one request at a time at 7.8 tokens/second. That is about $71 a month.

The self-hosted number is pessimistic. Batching requests or serving with vLLM would bring it down a lot.

## Limitations

50 test articles is small. The safety check is a keyword-based refusal detector on 10 prompts, not a full red-team. ROUGE and BLEU reward matching the reference wording, which favours fine-tuned models over a closed model that writes in its own style.

## Reproduce

Files in this repo:

- `aya-news.ipynb`: the whole pipeline in one cell (curation, both training runs, Command A outputs, evaluation, safety check, cost numbers)
- `aya-news-upload.ipynb`: pushes the data folder and both adapters to Hugging Face

Steps:

1. Make a Kaggle notebook with GPU T4 x2 and internet on.
2. Accept the licence for CohereLabs/aya-expanse-8b on Hugging Face.
3. Add two Kaggle secrets: `HF_Token_2` (a Hugging Face write token) and `COHERE_API_KEY`. The Cohere key is optional. Without it the closed-model comparison is skipped.
4. Run `aya-news.ipynb`. It takes about 6 hours with Save & Run All.
5. To upload, attach that notebook's output as an input to `aya-news-upload.ipynb` and run it.

Results are written to `/kaggle/working/data/results.json`. Seeds are fixed at 42, but scores can move slightly between runs because of GPU non-determinism.

Main packages: aligntune, curatorkit, auditkit, peft, bitsandbytes, transformers, datasets, cohere.

## Links

- Curated model: https://huggingface.co/rshahbaz/aya-expanse-8b-news-curated
- Raw-data model: https://huggingface.co/rshahbaz/aya-expanse-8b-news-raw
- Data, CuratorKIT reports, training logs and all model outputs: https://huggingface.co/datasets/rshahbaz/aya-news-curation
- Kaggle notebook: https://www.kaggle.com/code/rshshah/aya-news
