"""Correct reference scheduler with a deliberately brittle fixed chunk size."""


def schedule(requests, limits):
    decode = [request for request in requests if request["remaining_prefill_tokens"] == 0]
    if decode:
        return [
            {"request_id": request["id"], "tokens": 1}
            for request in decode[: limits["max_batch_size"]]
        ]

    result = []
    budget = limits["max_batch_tokens"]
    for request in sorted(requests, key=lambda item: (item["arrived_at_ms"], item["id"])):
        tokens = min(128, request["remaining_prefill_tokens"], budget)
        if tokens <= 0 or len(result) >= limits["max_batch_size"]:
            break
        result.append({"request_id": request["id"], "tokens": tokens})
        budget -= tokens
    return result
