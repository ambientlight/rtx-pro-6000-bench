"""Synthetic September handbook fixture from dc74842:scripts/dsv41_long_tps.py."""

TOPICS = (
    "admission control",
    "request cancellation",
    "chunked prefill",
    "prefix caching",
    "memory accounting",
    "speculative decoding",
    "fair scheduling",
    "stream backpressure",
    "tool argument validation",
    "incremental parsing",
    "deadline propagation",
    "retry budgets",
    "distributed tracing",
    "token accounting",
    "tenant isolation",
    "model rollouts",
    "crash recovery",
    "overload management",
    "capacity planning",
    "storage tiering",
    "data retention",
    "reproducible benchmarking",
    "regression testing",
    "incident response",
)

TASK = (
    "Write a comprehensive engineering handbook for a hypothetical multi-tenant inference service. "
    "The reference notes below are synthetic design inputs, not measurements of a real system. "
    "Write a substantial chapter for EVERY topic in the numbered list, in order. Each chapter "
    "should contain 400-600 words of original technical prose explaining design decisions, "
    "tradeoffs, a concrete failure scenario, and tests. Do not replace chapters with an outline "
    "or a summary. Do not ask to continue in another response. Avoid counting sequences and "
    "verbatim repetition. No tools are available; output the handbook itself in English.\n"
    + "\n".join(f"{i}. {topic}" for i, topic in enumerate(TOPICS, 1))
)

def make_prompt(nonce, count):
    lines = [f"Independent benchmark identity: {nonce}\n", TASK, "\nSynthetic reference notes:\n"]
    for i in range(count):
        topic = TOPICS[i % len(TOPICS)]
        variant = (
            f"Tenant {i % 137} has a {17 + i % 211} second deadline and a burst of {1 + i % 31} requests. "
            f"Track service epoch {1000 + i}, pending bytes {4096 + (i * 8191) % 65521}, "
            f"and cancellation budget {2 + i % 11}. "
        )
        concern = (
            "Separate queue delay from processing time and propagate cancellation to every owner.",
            "Bound resident state, account for in-flight work, and release resources after the terminal event.",
            "Preserve event ordering under partial writes; document retries and reject ambiguous replay.",
            "Compare uncached latency with steady-state throughput using versioned, reproducible fixtures.",
            "Consider rolling upgrades, worker loss, slow consumers, and fairness between large and small jobs.",
        )[i % 5]
        lines.append(f"Record {i}: {topic}. {variant}{concern}\n")
    lines.append("\nEnd of notes. Now write the full handbook as requested, starting with chapter 1.\n" + TASK)
    return "".join(lines)
