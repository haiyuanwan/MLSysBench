"""Strict FIFO starter policy. Replace it with an online policy."""


def schedule(requests, limits):
    oldest = min(requests, key=lambda request: (request["arrived_at_ms"], request["id"]))
    tokens = 1
    if oldest["remaining_prefill_tokens"] > 0:
        tokens = min(oldest["remaining_prefill_tokens"], limits["max_batch_tokens"])
    return [{"request_id": oldest["id"], "tokens": tokens}]
