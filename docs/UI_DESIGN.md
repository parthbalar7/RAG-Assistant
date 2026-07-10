# RAG Assistant — UI Design System

Single source of truth for the 2026 redesign. Every styling decision derives from here.
Principles: **calm, minimal, paper-like, fast.** No neon, no gradients, no glassmorphism,
no decorative motion. Hierarchy comes from type scale, spacing, and hairline borders —
not color, glow, or shadow.

## 1. Color tokens

Defined on `:root` (light) and `[data-theme="dark"]`. Components use ONLY these tokens.

| Token | Light (paper) | Dark (ink) | Use |
|---|---|---|---|
| `--bg` | `#F7F6F2` | `#161719` | App background |
| `--surface` | `#FFFFFF` | `#1D1F23` | Cards, panels, modals, composer |
| `--surface-2` | `#EFEEE8` | `#25272C` | Hover fills, wells, code chips, input bg |
| `--text-1` | `#1C1B18` | `#E8E6E1` | Primary text |
| `--text-2` | `#5D5B54` | `#A3A19A` | Secondary text, labels |
| `--text-3` | `#98958B` | `#6C6A64` | Tertiary, placeholders, timestamps |
| `--border` | `#E4E2DA` | `#2E3036` | Hairline borders everywhere |
| `--border-strong` | `#CFCCC2` | `#3C3F46` | Focused inputs, active outlines |
| `--accent` | `#4E5F7E` | `#93A7C9` | Primary actions, active states, links |
| `--accent-tint` | `#4E5F7E14` | `#93A7C91A` | Accent-tinted fills (selected item, user bubble) |
| `--ok` | `#5F7F63` | `#8FAE93` | Success |
| `--warn` | `#8F7A45` | `#C2AC77` | Warnings |
| `--danger` | `#9C5A50` | `#C08B84` | Errors, destructive actions |
| `--overlay` | `rgba(28,27,24,.35)` | `rgba(0,0,0,.5)` | Modal scrim |

Rules: exactly ONE accent hue (muted slate-blue). Status colors are desaturated —
never saturated red/green. White-on-accent text for primary buttons. Never use pure
black or pure white in dark theme.

### Legacy aliases (compat layer — do not use in new code)
Old variable names must keep resolving so any missed inline style degrades gracefully:
`--neon-cyan`→accent, `--neon-purple`→accent, `--neon-green`→ok, `--neon-pink`/`--neon-red`→danger,
`--warm`→warn, `--bg-base`→bg, `--bg-raised`/`--bg-surface`/`--glass`/`--glass-strong`→surface,
`--bg-subtle`/`--bg-overlay`→surface-2, `--text-primary/secondary/tertiary`→text-1/2/3,
`--text-ghost`→text-3, `--border-subtle`→border, `--border-neon`/`--glass-border`→border,
`--accent-soft`→accent-tint, `--accent-bright`/`--accent-dim`/`--accent-glow`→accent,
`--gradient-main`/`--gradient-subtle`/`--gradient-glass`→none (flat `var(--surface)`),
`--shadow-neon`/`--shadow-purple`/`--glass-glow`/`--border-glow`→`none`, `--code-bg`→surface-2,
`--success`→ok, `--info`→accent, `--font-display`→font-body.

## 2. Typography

- `--font-body`: `ui-sans-serif, -apple-system, "Segoe UI Variable", "Segoe UI", Roboto, Inter, sans-serif`
- `--font-mono`: `ui-monospace, "Cascadia Code", Consolas, "SF Mono", monospace`
- Scale: 11px (micro labels/badges) · 12px (secondary UI) · 13px (default UI) ·
  14.5px/1.65 (chat body) · 15px/600 (panel titles) · 18px/650 (modal titles).
- No display font, no letter-spacing tricks except `0.02em` uppercase 11px section labels.

## 3. Shape, depth, spacing

- Radii: `--radius-sm: 6px`, `--radius-md: 8px`, `--radius-lg: 12px` (modals, composer).
- Depth: hairline `1px solid var(--border)` is the primary separator. Shadows ONLY on
  floating layers: `--shadow-pop: 0 8px 28px rgba(0,0,0,.10)` light / `.45` dark
  (modals, dropdowns, toasts). Nothing else casts shadows.
- Spacing grid: 4px base; component padding 12–16px; section gaps 20–24px.

## 4. Motion

- One transition token: `--transition: 120ms ease` applied to `background-color, border-color,
  color, opacity` only. **No transforms, no keyframe animations, no hover lifts, no pulses,
  no springs, no particles.** Two exceptions (functional): a single minimal loading spinner
  (`spin 0.8s linear infinite` on a 14px 2px-ring) and the streaming caret (opacity blink 1s).
- Delete: ParticlesBackground, all `@keyframes` except the two above, `--transition-spring`.

## 5. Component recipes

- **Buttons**: primary = accent bg, white text, radius-md, 8px×14px, hover darken 6%.
  Secondary = surface bg + border, text-1. Ghost = transparent, text-2, hover surface-2 bg.
  Destructive = ghost with danger text. Height 32px standard, 28px compact.
- **Inputs/textarea**: surface-2 bg, border, radius-md, focus → border-strong + no glow ring.
- **Toggles**: 32×18px pill, surface-2 track/border, 14px knob; ON = accent track, white knob.
  No spring — knob position transitions 120ms.
- **Modals**: centered, surface, radius-lg, border + shadow-pop, max-width 480px, scrim overlay.
  Title 18px/650, body 13px text-2, actions right-aligned.
- **Toasts**: bottom-right stack, surface card, border + shadow-pop, 3px left border in
  status color, 13px text-1, no slide animation (fade only).
- **Chat**: centered column max-width 760px. User message = right-aligned block, accent-tint bg,
  radius-lg, 10px×14px. Assistant = NO bubble — plain text on bg with a 20px muted avatar dot
  and 12px name row; metadata (latency, route, memories) as 11px text-3 inline chips
  (surface-2 bg, radius-sm, no borders glow). Sources/badges = collapsed rows with 12px
  text-2, chevron, hairline top border.
- **Composer**: surface card, border, radius-lg, focus-within → border-strong. Toolbar icons
  ghost 28px; primary send button accent circle 32px.
- **Sidebar (left)**: bg (not surface), 260px, session items = 13px text-2 rows radius-md,
  hover surface-2, active = accent-tint bg + text-1. Section labels 11px uppercase text-3.
- **Right panel**: 320px surface with hairline left border; tabs = icon rail (36px buttons,
  active = accent-tint bg + accent icon), panel content 13px.
- **Topbar**: 48px, bg, hairline bottom border: app name 13px/600, status as plain text-3
  dots ("1,204 chunks · memory on"), theme toggle ghost icon button.
- **Settings**: grouped lists — 12px uppercase group label, rows of label + toggle with
  hairline separators. No cards-within-cards.
- **Scrollbars**: 8px, transparent track, `--border-strong` thumb, radius 4px.
- **Code blocks**: surface-2 bg, border, radius-md, 12.5px mono, language label 11px text-3
  top-right, copy button ghost.
- **Empty states**: centered, 15px text-2 title + 13px text-3 hint, single secondary button.
  No illustrations, no canvas.

## 6. Knowledge graph (clean + fancy)

Canvas bg = `--bg`. Edges: 1px, `--text-3` at 30% opacity (40% dark). Nodes: filled circles
with 1.5px `--surface` ring; size by degree (4–11px). Community palette (6 desaturated hues
that work on both themes): `#7E93B8 #8FAE93 #C2AC77 #B08A9E #7FA6A8 #A08D7B`; fall back to
accent when no community. Labels: 11px font-body `--text-2`, only for top-degree nodes and
the hovered/selected node (others appear on hover). Selected node: 2px accent ring + its
edges at full accent, rest dimmed to 15%. Tooltip = standard popover (surface, border,
shadow-pop). Controls (rebuild, communities, fit) = compact ghost buttons in a top-right
surface chip row. No glow, no motion trails; drag/zoom interactions keep working.

## 7. Layout

Keep the three-zone shell but calmer: left sidebar (bg) · chat (centered 760px column on bg)
· right panel (surface, icon-rail tabs). Both side zones collapsible. All panels flat —
no floating glass cards. Density: comfortable, not cramped; chat gets the space.
