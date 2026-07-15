# Nonstationary Fair Scheduler — Protocol Fixture

Implement an online policy using only the request fields exposed at each
decision. The public trace is one arrival profile; final replay covers burst,
constant-rate and shifted irregular arrivals and reports worst-profile goodput,
request SLO pass rate, and Jain fairness across tenants.

The trace is synthetic and only shaped by patterns reported in serving-trace
literature. Replace it with licensed, revision-pinned trace slices before using
this task in a paper leaderboard.
