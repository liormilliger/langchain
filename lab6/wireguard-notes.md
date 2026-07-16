# WireGuard field notes

Unit clock drift beyond 90 seconds breaks handshake validation. NTP must be
reachable through the maintenance tunnel before key exchange is attempted.

Adding a unit to a concentrator requires the peer's public key, the assigned
tunnel IP, and a SNAT rule for the board subnet. Missing SNAT presents as
one-way ping: unit reaches concentrator, concentrator cannot reach the board.
