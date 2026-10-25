# Vendored code: IFEval

`instructions.py`, `instructions_registry.py`, `instructions_util.py`, `evaluation_lib.py` and
`input_data.jsonl` are copied from Google Research's IFEval
(https://github.com/google-research/google-research/tree/master/instruction_following_eval),
licensed under the Apache License 2.0 (Copyright 2023 The Google Research Authors).

Modifications: imports now point at this package, and `instructions_util._get_sentence_tokenizer`
uses NLTK's `punkt_tab` (current NLTK no longer loads the pickled tokenizer). The files were also auto-formatted
with ruff (whitespace only). The checking logic is unchanged. Paper: Zhou et al., *Instruction-Following Evaluation for Large Language Models* (2023).
