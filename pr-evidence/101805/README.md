# PR #101805 visual evidence

These screenshots preserve the production reproduction and recovery sequence
for NousResearch/hermes-agent PR #101805.

- `checkpoint-error.png`: an already-open Desktop session continued to raise
  `CompressionCheckpointUnavailable` after `checkpoint_required` changed to
  `false`.
  SHA-256: `eaf87d4f43946539de4ec21e9ba944f38870dda59f23a664ef1518ff7c6b83da`
- `retry-success.png`: after a full Desktop restart rebuilt the agent from the
  corrected config, the exact failed prompt completed successfully.
  SHA-256: `b26753b094acac31ade6ff0ea84ab5c324fe89fd56e468f089ce4d1da20fdba4`

The restart receipt demonstrates the production workaround and stale-agent
diagnosis. The regression tests in PR #101805 directly demonstrate the
no-restart code fix.
