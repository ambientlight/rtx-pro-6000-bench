# F2Pool verification gateway

Run the gateway in the qualified checkpoint environment. It authenticates one worker at `pearl.f2pool.com:5500` and supplies the same certificate-v3 job to four tensor-parallel clients on loopback. This test assumes an isolated inference endpoint serving public inputs, since a Pearl proof includes opened activation and weight chunks.

```sh
python gateway.py --model "$MODEL" --verifier "$VERIFIER" \
    --token-file "$RUN/token" --receipt "$RUN/gateway.json" --arm-file "$RUN/armed"
```

The default account comes from the benchmark host's owner-only F2Pool preflight file. Both the gateway and rank clients need the same fresh private 64-hex-character token. Set `PEARL_REPLAY_GATEWAY_TOKEN_FILE` to its path within the serving process and `PEARL_REPLAY_POOL_PORT=31990`. A new run requires a fresh receipt path and an absent arm file. The gateway supplies actual jobs while the model warms up. The workload controller creates the arm file once startup checks finish and removes it after the run. Both changes invalidate the previous local job alias.

The receipt is an atomically replaced JSON snapshot. Read `connected`, `authorized_clients`, `armed`, `run_id` and `job` for readiness. Before each submission, the gateway matches weight openings to one TP shard, checks the reported phase against its allowed row counts, and independently verifies the proof against the pool's real header and assigned target with the pinned CPU verifier. Phase attribution also depends on the isolated authenticated serving process.

Only a matching successful upstream response increments `accepted`. The first such response, send and reply times, connection generation, job and proof hashes, target, checkpoint shard and phase remain in `first_accepted`. `rejected` counts pool refusals and `pool_stale` counts those identified as stale. `stale` counts candidates dropped on job or arm changes before forwarding. `unknown` counts submitted candidates with no confirmed result. A late matching response on the same connection can resolve an unknown. The receipt stores no raw proofs, pool credentials or model inputs.

Run `python test_gateway.py` for in-memory protocol checks.
