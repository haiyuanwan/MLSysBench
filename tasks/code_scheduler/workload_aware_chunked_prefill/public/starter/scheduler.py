"""Starter scheduler with intentional head-of-line blocking.

Only this file is editable. Keep the public ``schedule(requests, limits)`` API.
"""


def schedule(requests, limits):
    """Return work for the next batch.

    The starter uses strict FIFO and lets the oldest prefill consume the entire
    token budget. It is correct, but long prompts can block short requests and
    active decode work.
    """

    if not requests:
        return []
    oldest = min(requests, key=lambda request: (request["arrived_at_ms"], request["id"]))
    if oldest["remaining_prefill_tokens"] > 0:
        tokens = min(oldest["remaining_prefill_tokens"], limits["max_batch_tokens"])
    else:
        tokens = 1
    return [{"request_id": oldest["id"], "tokens": tokens}]
