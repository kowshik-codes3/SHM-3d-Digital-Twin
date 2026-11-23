---
name: Structural Telemetry HUD
colors:
  surface: '#0f131d'
  surface-dim: '#0f131d'
  surface-bright: '#353944'
  surface-container-lowest: '#0a0e18'
  surface-container-low: '#171b26'
  surface-container: '#1c1f2a'
  surface-container-high: '#262a35'
  surface-container-highest: '#313540'
  on-surface: '#dfe2f1'
  on-surface-variant: '#bbc9cd'
  inverse-surface: '#dfe2f1'
  inverse-on-surface: '#2c303b'
  outline: '#859397'
  outline-variant: '#3c494c'
  surface-tint: '#2fd9f4'
  primary: '#8aebff'
  on-primary: '#00363e'
  primary-container: '#22d3ee'
  on-primary-container: '#005763'
  inverse-primary: '#006877'
  secondary: '#adc6ff'
  on-secondary: '#002e6a'
  secondary-container: '#0566d9'
  on-secondary-container: '#e6ecff'
  tertiary: '#68f5b8'
  on-tertiary: '#003824'
  tertiary-container: '#46d89d'
  on-tertiary-container: '#005a3d'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#a2eeff'
  primary-fixed-dim: '#2fd9f4'
  on-primary-fixed: '#001f25'
  on-primary-fixed-variant: '#004e5a'
  secondary-fixed: '#d8e2ff'
  secondary-fixed-dim: '#adc6ff'
  on-secondary-fixed: '#001a42'
  on-secondary-fixed-variant: '#004395'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#0f131d'
  on-background: '#dfe2f1'
  surface-variant: '#313540'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 40px
    fontWeight: '600'
    lineHeight: 48px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 30px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0.005em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.01em
  telemetry-lg:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: -0.02em
  telemetry-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: -0.01em
  telemetry-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-mobile: 0.75rem
  margin: 1.5rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system embodies the calculated authority of mission-critical civil infrastructure monitoring crossed with the optical precision of aerospace telemetry heads-up displays (HUD). It balances heavy industrial safety accountability with cutting-edge spatial computing and machine intelligence. 

The aesthetic is anchored in Dark Technical Minimalist Glassmorphism: dark, low-emissivity canvases overlaid with hyper-legible translucent panels, hairline borders, and targeted luminescent accents. Visual weight communicates operational gravity; glowing data points act as functional beacons rather than decoration. The interface feels like an advanced structural flight-deck—uncluttered, instantaneous, mathematically grounded, and calm under operational stress.

## Colors

The palette establishes an ultra-low luminance foundation (`#0b0f19`) to maximize the signal-to-noise ratio of critical structural metrics. Surfaces elevate via dark slate containers (`#111827` and `#131b2e`) utilizing 8% white alphas (`rgba(255, 255, 255, 0.08)`) for physical separation without heavy opaque fills.

### Functional Roles
- **Accent / Telemetry Glow (`#22d3ee`):** Denotes active sensor interrogations, cursor crosshairs, focal inspection targets, and digital twin node selections.
- **Nominal / Baseline (`#3b82f6`):** Standard operational state for structural members, expected load vectors, and verified sensor telemetry.
- **Live / Stream Active (`#10b981`):** Active data ingress, high-frequency sampling health, and healthy network edge relays.
- **Warning (`#f59e0b`):** Modal deformation divergence, threshold alerts, high strain anomalies.
- **Critical Failure (`#ef4444`):** Limit state violations, seismic exceedance, imminent structural failure. Paired with a concentrated high-saturation red bloom (`0 0 16px rgba(239, 68, 68, 0.45)`).

## Typography

The typographic strategy implements strict dual-type discipline:
- **Inter** handles narrative, system navigation, incident log descriptions, and contextual dialogs. It maintains optical balance, high neutral clarity, and predictable legibility across varied densities.
- **JetBrains Mono** governs the data layer: sensor node identifiers (`SN-8820-C`), spatial coordinates (`X: +124.08mm`), Mean Squared Error (`MSE: 0.0024`), Hz readings, modal frequencies, and epoch timestamps. 

All technical metrics, status codes, and telemetry fields are rendered in tabular numerals (`tnum`) to eliminate spatial jitter during continuous real-time data streaming.

## Layout & Spacing

Layouts conform to a dense, precision-docked telemetry cockpit model:
- **Desktop (Viewport ≥ 1280px):** 12-column dynamic grid framing the primary 3D WebGL digital twin viewport. Sidebars and metric overlays anchor with fixed widths (360px to 420px) to prevent spatial shifts during data polling, with `1rem` gutters between floating metric cards.
- **Tablet (768px – 1279px):** 8-column layout. Metric readouts dock along the bottom horizontal rail; digital twin remains fixed at a 60% viewport height split.
- **Mobile (< 768px):** Single-column stacked stream. Secondary panels collapse into swipeable HUD sheets.

Internal component spacing follows strict 4px/8px modular units. Compact density allows maximum data visibility without layout scrolling during structural emergency events.

## Elevation & Depth

Visual hierarchy uses semi-transparent glassmorphic panels and dark optical layering rather than standard diffuse drop shadows:

- **Surface Base (Level 0):** Primary background (`#0b0f19`), pure non-reflective slate void.
- **Surface Layer 1 (Level 1 HUD Panels):** `#111827` at 75% opacity, backdrop filter blur of `16px`, bordered by `1px solid rgba(255, 255, 255, 0.08)`.
- **Surface Layer 2 (Level 2 Flyouts & Active Cards):** `#131b2e` at 85% opacity, backdrop filter blur of `24px`, bordered by `1px solid rgba(255, 255, 255, 0.12)`.
- **Luminescent Overlays:** Elements under warning or fault conditions project directed, ambient neon glows onto adjacent dark surfaces:
  - Cyan Telemetry Focus: `box-shadow: 0 0 12px rgba(34, 211, 238, 0.25);`
  - Critical Limit Anomaly: `box-shadow: 0 0 16px rgba(239, 68, 68, 0.45);`

## Shapes

The design uses a clean, engineered profile (`0.25rem` / 4px base border-radius). This low-curvature structure reinforces precision engineering, telemetry instruments, and industrial machinery. 

Buttons, metric enclosures, and diagnostic tabs feature crisp, clipped corners, with larger containers (`rounded-lg` at 8px) housing the glassmorphic viewport trays. Pill shapes and circular elements are reserved exclusively for sensor node indicators and status beacons.

## Components

### Buttons & Interactive Controls
- **Primary Telemetry Trigger:** Cyan-tinted ghost border (`rgba(34, 211, 238, 0.3)`) over `#131b2e`, text in `#22d3ee`, with subtle internal cyan gradient. Hover state triggers a `0 0 12px rgba(34, 211, 238, 0.3)` glow.
- **Destructive / E-Stop Action:** Dark red surface with `#ef4444` border and high-contrast light text; active states exhibit an emergency halo.
- **Control Sizing:** Compact 32px standard height with 12px inline padding to optimize operational screen density.

### Badges & Telemetry Nodes
- **Sensor Nodes (3D Overlay & HUD):** Outer concentric pulsing ring (8px diameter) surrounding a solid core (4px). Color strictly maps to status: Cyan (selected), Blue (nominal), Amber (warning), Red (critical limit).
- **Metric Badges:** Monospaced label containers with `0.25rem` radius, dark gray fill (`rgba(17, 24, 39, 0.8)`), and a leading 2px status indicator border on the left edge.

### Precision Gauges & Line Charts
- **Residual & Strain Charts:** Dark background with hairline micro-grids (`rgba(255, 255, 255, 0.04)`). Chart traces have a 1.5px stroke width accompanied by a faint vertical area gradient (`rgba(34, 211, 238, 0.15)` to transparent).
- **Radial Load Gauges:** High-contrast circular arcs with mono tick marks, using step indicators along the outer circumference to display peak-hold values.

### Cards & Viewport HUD Trays
- Translucent dark bases with 8% white hairline borders. Card headers feature monospaced category tags (`label-caps`) paired with right-aligned operational status dots (`#10b981`).

### Timeline Replay Scrubber & Fault Toolbars
- **Timeline Scrubber:** Millisecond-accurate scrubber utilizing JetBrains Mono timestamps. Anomalous time segments are highlighted on the track in amber and red bands.
- **Fault Injection Control:** Segmented multi-toggle switches with etched divider lines and cyan indicators denoting artificial load variables.