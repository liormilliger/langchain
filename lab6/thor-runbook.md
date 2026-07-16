# Thor Platform Runbook — Field Operations (labs 6)

## 1. CrashLoopBackOff on thor-core

When thor-core pods enter CrashLoopBackOff, do not restart the pods manually.
Manual restarts mask the underlying cause and desynchronize the HelmRelease state.
Instead, inspect the last termination reason with kubectl describe pod, and check
whether the post-upgrade passport hook Job completed. The most common cause in the
field is a missing or malformed secret, typically thor-mq-credentials in the thor
namespace. If the secret is missing, recreate it from Secrets Manager and resume
the HelmRelease with flux resume helmrelease thor-platform. Service should recover
within two reconciliation cycles, roughly three minutes.

## 2. WireGuard connectivity loss

If a field unit shows zero handshakes, verify in this order: the concentrator
security group allows UDP 13233 from the unit's egress IP, the unit's system clock
is within 90 seconds of true time, and the wg-quick service survived the last
reboot. A stale PID file after an unclean shutdown prevents wg-quick from starting;
remove /var/run/wireguard/wg0.pid and restart the service. Never regenerate unit
keys in the field — a key mismatch requires re-registration of the peer on the
concentrator and takes the unit offline for the duration.

## 3. ActiveMQ queue depth growth

Steadily growing queue depth almost always means consumers are down or too slow,
not that producers are too fast. Check consumer pod health first. If consumers are
healthy but slow, inspect prefetch settings before scaling out. Purging queues in
production requires a change ticket and sign-off from the on-call lead, because
purged telemetry cannot be recovered — units do not retransmit acknowledged batches.

## 4. Postgres connection exhaustion

Connection pool exhaustion presents as intermittent application timeouts while the
database itself shows low CPU. Check max_connections on the server and the pgbouncer
pool size. The thor-core deployment defaults to 40 connections per replica; scaling
replicas without adjusting pgbouncer multiplies this. Reducing per-replica pool size
is almost always safer than raising max_connections.