The paired trials replay synthetic requests against the full DeepSeek-V4.1-Flash-1017 checkpoint. Both serving profiles use TP4, EP4, six-token DSPARK target verification, full CUDA graphs for decode, 2,048-token eager prefill chunks, and a 275 W limit on each of the four SM120 cards. The test runs the unpacked serving image with the SGLang source overlay recorded in the measurement setup. The control keeps the checkpoint-native FP8 projection. The joined arm uses the candidate’s calibrated INT7 projection for target layer 0.

Each primary fixed-period trace sends the same four distinct 2,048-input, 256-output requests at 20-second intervals over 80.5 seconds, including trailing idle time. Fixed-gap trials wait one second after each stream closes. A separate burst uses 24 distinct 256-input, 16-output requests with 250 ms after each stream. An independent observer samples the broker, retains the four workers’ hardware entry timestamps and verifies that their completed-slice counters stay still during decode. Each trace has a separate untimed warm-up. The handbook reproduces the historical Responses prompts with an 8,192-token budget at 32K and 512-token budgets at 131K and 200K. Generated output lengths are recorded for each request.

Fleet bounds count completed hardware tiles at either side of each time boundary. The calibrated direct kernel uses 160 × 256 tiles with a reduction dimension of 65,536. Inference proof workers count scanned tickets from the actual projection. One compact decode ticket covers 2 × 16 × 2,048 MACs. A wide prefill ticket covers 2 × 64 × 2,048. One full TP4 projection row therefore contributes at most 41,943,040 MACs. The protocol scales the success target by that tile work for both jobs. Actual accepted shares also depend on difficulty, uniqueness, proof submission and verifier acceptance. The dedicated test peers exercised model-origin proofs on public inputs locally. Payout was outside the measurement.

The direct job’s base target is 2^208. The joined qualification peer uses 2^224 so the short single-stream trials can produce witnesses. A second joined run used the direct fleet’s exact 2^208 target. The adapter opens at most one winning ticket per projection invocation. At the easy target several winners can therefore produce one ACK. Dataset digests, phase counters and stage-specific times support attribution. Public network acceptance and equivalence to the original checkpoint's model quality require separate evidence.

The harness records the exact runtime pins. [Measurement setup](../measurement-setup.json)

To recompute the equal-target comparison from this directory, use the archived trace and fleet records. The comparison writes its output to the supplied receipt path.

```sh
python3 ../../full-canary/graph_compare.py compare-traces \
  --control switch-fixed-period.json --candidate joined-equal-target-fixed-period.json \
  --control-fleet switch-fixed-period-fleet.json \
  --candidate-fleet joined-equal-target-fixed-period-fleet.json \
  --receipt /tmp/pearl-275-equal-target.json
```
