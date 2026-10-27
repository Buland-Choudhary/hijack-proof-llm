# Data card

| Set | Where | Purpose | Notes |
|---|---|---|---|
| TaskForge train documents | generated on demand (`hijack_proof.data.taskforge`), rows in `data/generated/` (not committed) | SFT and DPO | train templates only |
| Alpaca-cleaned | downloaded to `data/raw/` (not committed) | keep general usefulness | with and without injected attacks |
| Trusted-directive rows | `data/generated/sft_userfollow.jsonl` (not committed) | teach obeying the user's own directive | deterministic compliant targets |
| Frozen eval sets | [`evalsets/v2/`](../evalsets/v2/MANIFEST.json) | final test, dev, userfollow, over-defense, SecAlign-style | sha256 manifest; never trained on |
| IFEval | vendored official checkers and prompts (Apache-2.0, `src/hijack_proof/eval/ifeval/NOTICE.md`) | usefulness check | |

- **Generation:** all TaskForge text, names and numbers are synthetic. Alpaca-cleaned is third-party data under
  its own license.
- **Held-out:** four attack families and all evaluation phrasings never appear in training data. Test documents
  come from templates not used in training. An 8-gram overlap report is in
  [`results/sft/overlap_8gram.json`](../results/sft/overlap_8gram.json).
- **Payloads:** benign only (canary strings, wrong answers, `example.com` links, refusals, French).
- **Personal data:** none. The repository safety hook blocks secrets, personal emails and phone numbers.
