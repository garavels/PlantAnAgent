# Product

## Register

product

## Users

Operators and demo viewers of an autonomous, intermittently powered industrial
plant. They sit in a control room context: watching live plant state, judging
whether the plant is behaving, and occasionally driving it (play / pause /
reset, dispatch shares). In the near future an agent takes over the driving and
the human supervises. Success on screen is instant situational reading: what is
generating, what is stored, what is flowing, what is off.

## Product Purpose

Plant An Agent is the foundation for an autonomous industrial plant: a small
working digital twin (solar -> power allocation -> battery + electrolyser ->
H2 storage) plus a control room that watches it. V2 adds a 3D digital-twin
view so the plant reads as a physical place, not a dashboard. The 3D view is a
visualization layer only; the simulator stays completely independent of it.

## Brand Personality

Serious technical product. Quiet, precise, engineered. A control room, not an
admin panel — and not a video game. Three words: calm, industrial, exact.

## Anti-references

- Generic admin-dashboard aesthetics; excessive nested cards
- Cyberpunk neon, gratuitous gradients, glassmorphism
- Game HUD aesthetics (radar rings, health bars, loot-style callouts)
- Huge rounded consumer UI elements; visual clutter
- Realistic CAD-level plant models; video-game scene dressing

## Design Principles

- The plant is the protagonist. Chrome recedes; the scene carries the screen.
- State, not decoration. Every visual change maps to a PlantState field; the
  UI adds no physics and no motion that doesn't encode information.
- Hue is rationed. Mostly monochrome surfaces; one desaturated tint per energy
  carrier (sand solar, sage stored, mist feed, chalk hydrogen).
- One coherent world. The 3D twin extends the warm olive-charcoal identity of
  the console — it is the same room, seen closer.
- Restraint over richness. Subtle lighting, smooth but minimal animation,
  clean typography, strong spatial hierarchy.

## Accessibility & Inclusion

WCAG AA text contrast on overlays and controls; keyboard-reachable controls;
respect `prefers-reduced-motion` (freeze ambient and flow animation).
