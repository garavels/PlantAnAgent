# E-beta mandatory-delivery model repair

Status: **prospective continuation of the user-authorized uncertainty/model repair**

The first three fixtures at `c5b049b` are all verified negative results: E-beta
stopped after 72, 144 and 25 decisions respectively. All process exits were zero,
all evidence verified, and no cap, fallback or protected reset occurred. Seed
9,100,002 also failed its first daily CH4 return. Full results and archive hashes
are retained in `results/v2_beta_uncertainty_c5b049b.json`.

The original sunset failure was passed, but the retained delivery records expose
a second model mismatch. At decision 71 on seed 9,100,001, the model expected to
ship 41.933381 kg; the simulator shipped 240 kg. At decision 95 on seed 9,100,002,
the model expected zero shipment and the simulator shipped 240 kg. The physical
settlement is `min(available CH4, request)`; there is no operator withholding
action in the four-target interface. Maximizing aggregate delivery under upper
bounds does not reproduce that mandatory settlement, particularly when positive
terminal inventory requirements compete with shipments.

Before another controller run, policy `public-p10-early-reserves-v2` adds exact
mandatory settlement to its model: at each deadline, either the full request is
delivered or post-delivery inventory is zero. One binary per actual deadline
encodes this disjunction; existing inventory balance and request upper bounds
then imply `delivered = min(available, requested)`. A partial delivery cannot
coexist with retained methane. Zero-availability and exact-request ties remain
valid. This changes the revised E-beta model only; legacy mean/M24/alpha defaults
and simulator settlement remain untouched. Original hard returns, 1% buffers,
two-hour lead, p10 reconstruction, constant-power envelope and solver tolerances
are unchanged. Infeasible models still fail explicitly without fallback.

Tests must cover full and partial settlement independently of objective weights,
reject withholding a shipment to protect a reserve, retain legacy formulation,
and reconstruct the complete solver witness. After committing this repair and
its tests, a new distinct fixture schema may repeat the same three disclosed
fixture seeds in their fixed order, once per clean revision, with all limits from
`BETA_MPC_UNCERTAINTY_REPAIR.md` unchanged. The earlier attempt is not replaced or
retried at its revision. No registered training or study evaluation is authorized.
