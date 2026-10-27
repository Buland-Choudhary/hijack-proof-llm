# M3 note: is the eval set contaminated by the training data?

`scripts/overlap_report.py` splits every eval item (document plus user instruction) into word 8-grams
and measures how many also occur in the training text (the Qwen SFT rows plus the trusted-directive rows;
`results/sft/overlap_8gram.json`).

- Final test (`core`, 1,350 items): mean overlap 13.5%. Clean documents overlap *most* (14.9%), attacked
  ones with held-out styles least (10.9%). That pattern points to shared template boilerplate (field labels,
  "Answer concisely", table headers), not leaked documents: the test documents come from templates that never
  appear in training, and the names, amounts and dates differ.
- 5.0% of core items share at least half their 8-grams with the training text (max 72.9%). These are the short
  documents, where a few boilerplate phrases are most of the text. The dev split is higher (23.5% mean) because
  it uses training templates by design.
- `overdefense` (1.5%) and `secalign` (3.2%) are nearly disjoint from training.

Conclusion: no sign of copied test documents; overlap is the generators' shared structure. Documented as a
limitation: the benchmark is synthetic, so a model can learn the generators' surface form.
