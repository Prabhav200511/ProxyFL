# Communication-overhead review

Reviewed 2026-08-28. This assessment does not change the FL algorithm, security
protocol, routing behavior, or reported measurements.

## What the current implementation measures

- `network.send_msg` sends encoded application messages over localhost TCP.
  `bytes_tx` counts successful `sendall` calls as encoded message length plus a
  four-byte application frame prefix. It is not a packet-capture measurement of
  bytes physically transmitted or proof of application-level delivery.
- The complete secure envelope is included in those bytes. Security byte cost
  and useful FL payload byte cost are not separately attributed. Security
  timings and energy are different quantities and cannot be added to bytes.
- `model_payload_bytes_rx` is a receiver-side accepted-model metric, not a
  complete count of FL traffic on all links. It cannot simply be subtracted
  from global transmitted bytes to obtain security overhead.
- `vanet_channel` models capacity from link distance. Its airtime is
  observational: it does not simulate MAC contention, retries, fragmentation,
  packet loss, or routing. The quantity named `vanet_goodput_bps` currently
  uses complete framed-message bits, not just useful FL payload bits, so it
  should not be described as application-payload goodput.
- No AODV route table, route discovery, route repair, RREQ, RREP, RERR, HELLO,
  or RREP-ACK events exist. Current results cannot establish AODV routing
  overhead or normalized routing load. Report these as **not modeled**, not
  as evidence of zero routing overhead in a real VANET.

## Valid accounting if routing is implemented

The proposed sum is sound only when all terms use the same units and counting
boundary, and the components do not overlap:

`total communication bytes = FL bytes + security bytes + routing bytes`

If FL bytes include the useful model payload, the sum is total communication
volume, rather than overhead above the useful payload. To claim an overall
on-air measure, also define where transport/network/MAC/PHY headers, framing,
ACKs, and retransmissions are counted. Do not add security bytes again when
the FL-message term already includes the secure envelope.

Count each routing-control transmission at each hop, including rebroadcasts
and retries at the chosen measurement layer. One broadcast transmission is
not multiplied by the number of neighbors that receive it. Do not sum TX and
RX counters to estimate bytes transmitted: that double-counts reception.
RREQ/RREP/RERR are not the complete list in every AODV configuration: include
RREP-ACK and HELLO (a special RREP) when they are used, without counting a
HELLO twice. Include startup discovery as well as later repairs.
[RFC 3561, sections 5 and 6](https://www.rfc-editor.org/rfc/rfc3561.html)

Report normalized routing load separately, with an explicit definition such
as:

`NRL = routing-control packet transmissions across hops / data packets delivered to their final destinations`

This packet ratio is not a byte count and must not be added to the overhead
sum. Define the data-packet layer and observation window; an FL update is not
necessarily one network packet. Use N/A when no data packets are delivered.
Routing efficiency must distinguish control traffic from delivered data and
state its counting conventions.
[RFC 2501, section 6.1](https://www.rfc-editor.org/rfc/rfc2501.html)

Stationary vehicles may reduce route repairs, but startup discovery, route
expiry, optional HELLO traffic, channel contention, and workload still matter.
Neither latency decreasing monotonically nor throughput increasing with every
FL round follows from vehicles being stationary. In the current fixed-distance
capacity model, unchanged links have unchanged capacity.

## Next decision

Actual AODV/NRL results require a routing simulation or packet-level simulator
integration with recorded control transmissions and data deliveries. Adding
fixed synthetic RREQ/RREP/RERR byte allowances to the current direct-link
simulation would not validate an AODV implementation. This is a separate
modeling extension, not a folder-cleanup change.
