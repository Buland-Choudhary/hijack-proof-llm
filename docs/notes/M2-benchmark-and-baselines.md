# M2 design note: benchmark and baselines

## What this milestone built, in two minutes
- **TaskForge**: a seeded generator of small tasks over documents (12 document types, 18 question
  types). Every answer is computed from the data behind the document, so correctness is checked
  by code, not by a judge model.
- **Attack library**: 11 ways to hide an instruction in a document, plus multi-injection. Every
  payload is harmless (print a code, state a fake answer, add an example.com link, refuse,
  answer in French) and detected by code.
- **Frozen evaluation**: held-out templates, 4 attack styles never used for training, separate
  payload wording for training and evaluation, 95% bootstrap intervals, checksums, and a test that
  regenerates the files to catch accidental changes.
- **Baselines**: five untrained models, a prompting defense chosen on dev, and a detector.

## Why it is designed this way
- *Why not an LLM judge?* A judge adds its own errors and cost, and it can't be re-run exactly.
  Hijacks here are objective: a random code appears in the answer or it doesn't.
- *Why a dev set that differs from the test set?* The dev set only contains attack styles and
  wordings that training may use. If I tuned on held-out styles, "works on unseen attacks" would
  no longer be a fair claim. I found this leak in my first version of the dev set and rebuilt it.
- *Why choose the baseline defense on dev?* Choosing it on the test set would make the baseline
  look better than it should and make my trained model's win look smaller or larger for the wrong
  reason. The rule was written before the results: lowest attack success with at most a 3-point
  loss in clean accuracy.
- *Why report "loose" and "strict" attack success?* Loose counts any appearance of the payload,
  including a model that quotes the injected text while answering correctly. Strict requires the
  response to start with the payload. They differ by only about 1 point, so quoting is rare.

## Findings and what they mean
- Bigger models in the same family were hijacked more (45% to 58-59%), because better
  instruction following cuts both ways. A defense has to teach *which* instructions to follow.
- Unseen attack styles work less on untrained models (39% vs 49% for Qwen3-0.6B). So the fair test
  of a trained defense is the held-out column compared with its own lower baseline.
- The sandwich reminder gives a 10-12 point drop at no accuracy cost, so a trained model must beat
  roughly 34-36% to be worth the training.
- The detector is strong on detection but costly in false alarms (9.3%) and it doesn't answer the
  question. It is a useful comparison point, not a replacement.

## Pitfalls hit
- Hugging Face and the original AlpacaFarm prompts are blocked here, so the SecAlign-style suite uses
  held-out Alpaca-cleaned prompts instead (decision D15); its numbers are not comparable with papers.
- The llama.cpp server was killed for memory (default 8 GB RAM prompt cache) and failed once on a
  cold disk read; both are fixed and recorded in the code.
- A cloud machine reset killed a first batch of runs. Evaluations now save every chunk and resume.
- The detector first trained in pure bf16, which can lose small updates; it now keeps fp32 weights
  with bf16 compute.

## Questions you should be able to answer
1. *What is prompt injection and why can't a bigger model just ignore it?*
2. *How do you know a defense generalizes to attacks it hasn't seen?* (held-out styles and wordings,
   dev/test separation, checksummed frozen set)
3. *Why are the confidence intervals bootstrap-based and what do they tell you?* (resample items;
   the 95% interval for 900 attacked items is about +/-3 points)
4. *Why compare against a prompting defense and a detector, not just the untrained model?*
5. *What would make these numbers misleading?* (synthetic tasks, small models, my own IFEval setup,
   a canary-string payload being easier to detect than a real attack)
