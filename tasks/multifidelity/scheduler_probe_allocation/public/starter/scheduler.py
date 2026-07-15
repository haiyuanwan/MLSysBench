"""Static expert-template policy that may be retained or generalized."""


def schedule(requests, limits):
    decode = sorted(
        (request for request in requests if request["remaining_prefill_tokens"] == 0),
        key=lambda request: (-request.get("priority", 0), -request["waiting_ms"], request["id"]),
    )
    if decode:
        return [
            {"request_id": request["id"], "tokens": 1}
            for request in decode[: limits["max_batch_size"]]
        ]
    selected = sorted(
        requests,
        key=lambda request: (-request.get("priority", 0), request["remaining_prefill_tokens"], request["id"]),
    )
    result = []
    budget = limits["max_batch_tokens"]
    for request in selected:
        tokens = min(64, request["remaining_prefill_tokens"], budget)
        if tokens <= 0 or len(result) >= limits["max_batch_size"]:
            break
        result.append({"request_id": request["id"], "tokens": tokens})
        budget -= tokens
    return result
