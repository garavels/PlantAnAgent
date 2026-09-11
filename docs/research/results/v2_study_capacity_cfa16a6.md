# Capacity after the E-beta terminal-bound repair

The separately frozen fixture at clean
`cfa16a670d686c940e91b67d9d0fd25050393835` completed all four process attempts.
**Complete study capacity remains unadmitted.** The E-beta rounding failure was
passed, but a later public boundary was infeasible. The original
[`17331fe` result](v2_study_capacity_17331fe.md) remains unchanged and visible.

All actors used the same fixture seed 9,100,001, full 168-hour task, 15-minute
decisions and one-minute physics. PPO remained an untrained fixed-action proxy.
No process timeout, log/RSS cap, fallback, optimizer update or protected reset
occurred. All child exits were zero because failures were retained and their
prefix evidence independently reconstructed.

| Arm | Decisions / 672 | Minutes / 10,080 | Child seconds | Peak RSS GiB | Physical stored bytes | Decoded bytes | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H | 672 | 10,080 | 64.148 | 1.478 | 11,406,113 | 109,386,649 | Complete, cycle-infeasible |
| M24 | 36 | 540 | 54.803 | 0.532 | 1,606,197 | 14,616,421 | Unsupported degraded/offline DAC |
| E-beta | 54 | 810 | 30.597 | 0.521 | 1,678,155 | 15,803,974 | Primary service model infeasible |
| Fixed PPO proxy | 672 | 10,080 | 65.506 | 1.429 | 11,635,665 | 110,003,801 | Complete, cycle-infeasible; forced stops |

H and the fixed proxy again provide complete storage/time observations only.
Neither meets the cycle gate; the proxy also fails hard-safety eligibility because
of electrolyser forced stops. Neither is a policy-performance success. M24's
declared equipment-model limitation persists. Neither MPC prefix establishes a
complete-week capacity bound or a controller-performance comparison.

## E-beta failure mechanism

The original attempt stopped after 52 decisions when a candidate battery return
was two floating-point ULPs below its minimum. The versioned equivalent-bound
repair passed 219 targeted tests and the new fixture advanced through that
boundary. It stopped after **54 decisions, 13.5 elapsed hours**, with HiGHS status
2 (infeasible) in the primary service stage. No economic stage or fallback then ran.

The saved current public boundary permits a solver-free explanation:

- Battery energy: **5012.459604808251 kWh**.
- Required cycle minimum: **5013.628497150464 kWh**.
- Remaining horizon: **42 intervals**, 10.5 hours to the cycle boundary.
- Remaining issued solar energy over that horizon: **0 kWh**.
- Even allocating all forecast solar to the battery with no losses or consumption
  leaves a **1.16889234221253 kWh** deficit.

The preceding plan placed its terminal battery exactly at the minimum. Realized
minute operation did not reach that projected state before the remaining solar
forecast became zero. This establishes infeasibility of the retained model from
that boundary; it does not establish that the task was infeasible at episode
opening or that a different earlier policy would fail. It is not another rounding
defect, and relaxing the task tolerance would change the research question.

The next control work is to assess preservation of resource feasibility under
forecast and model error before the last production window closes. That changes
controller strategy and needs its own prospective scope, separate from the
completed numerical repair. No such strategy or replacement comparator was
selected by this capacity fixture. Primary-comparator readiness must be resolved
before spending the registered training budget.

## Evidence

Raw report, archives, logs and fixed checkpoint remain outside Git at
`/private/tmp/plantanagent-study-capacity-cfa16a670d68/`.
The [compact report](v2_study_capacity_cfa16a6.json) binds every measurement,
process, archive and inventory. The [public-boundary energy diagnosis](v2_study_capacity_cfa16a6_boundary.json)
binds the exact saved observation, forecast projection and rebuilt model; it used
zero environment resets and zero solver calls.

Report raw SHA-256:
`819e4b499560679c7726abbe805a69511ed10b724bd015d374419802675807ef`.

Report domain SHA-256:
`bb83ac37a12146d5834ad2e275a6d5188fe6425667ed6a315e4d9972e012c039`.

Compact report domain SHA-256:
`edd9105119af56870672018bdea39134cf53a664c30b936980bee0c850aa2670`.

Public-boundary diagnosis domain SHA-256:
`f219e66c576240fa99e78845688259ad182fd2225d9f7a7f9e2574603322f5ea`.

No stage budget is frozen from these prefixes. The complete clean repository
release gate and later explicit training authorization remain outstanding.
The three-family/five-optimizer-seed design and protected seed domains are intact.
