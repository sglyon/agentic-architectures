# Neversink Theme Documentation - Compressed

## Core Concepts

**Theme**: Slidev presentation theme with custom layouts, components, and styling system
**Base**: Built on Vue.js, Tailwind CSS, and UnoCSS

## Color System

### Schemes (3 tiers)
- **B&W**: black, white, dark, light, navy, navy-light
- **Light colors**: [color]-light (e.g., red-light, blue-light) - lighter backgrounds
- **Regular colors**: Standard colors (red, blue, green, etc.) - darker backgrounds

### CSS Variables Set by Schemes
```css
--neversink-bg-color, --neversink-text-color, --neversink-border-color
--neversink-highlight-color, --neversink-fg-color, --neversink-bg-code-color
```

### Application
- Class format: `neversink-{color}-scheme` + `ns-c-bind-scheme`
- Example: `<div class="neversink-red-scheme ns-c-bind-scheme">`

## Layouts (12 types)

### Structure Components
- **Frontmatter**: YAML metadata defining layout type and parameters
- **Slots**: Named content areas using `:: slot-name ::` syntax

### Layout Types & Key Props

1. **cover/intro**: Title slides
   - Props: `color`
   - Slots: default, notes

2. **default**: Standard content slide
   - Props: `color`
   - Slots: default only

3. **two-cols-title**: Two-column with title
   - Props: `color`, `columns` (12-unit grid), `align` (3-part: title-left-right), `titlepos` (t/b/n)
   - Slots: title, left, right, default

4. **top-title**: Title band at top
   - Props: `color`, `align` (l/c/r)
   - Slots: title, content

5. **top-title-two-cols**: Title band + two columns
   - Props: `color`, `columns`, `align` (title-left-right)
   - Slots: title, left, right

6. **side-title**: Vertical title band
   - Props: `color`, `titlewidth`, `align`, `side` (l/r)
   - Slots: title, content, default

7. **quote**: Centered quotation
   - Props: `color`, `quotesize`, `authorsize`, `author`
   - Slots: default

8. **section**: Section divider
   - Props: `color`
   - Slots: default

9. **full**: Full-screen content
   - Props: `color`
   - Slots: default

10. **credits**: Scrolling credits
    - Props: `color`, `speed`, `loop`
    - Slots: default

### Column System (12-unit grid)
- Shorthands: `is-1` through `is-11`, `is-half`, `is-one-third`, etc.
- Alignment notation: `[h][v]` where h=l/c/r (horizontal), v=t/m/b (vertical)

## Components (13 types)

### Info/Decoration
- **Admonition**: Colored info box with icon (`title`, `color`, `width`, `icon`)
- **AdmonitionType**: Preset admonitions (`type`: info/important/tip/warning/caution)
- **StickyNote**: Post-it style note (`title`, `color`, `width`, `textAlign`)
- **SpeechBubble**: Speech bubble (`position`, `shape`, `color`, `maxWidth`)

### Drawing/Graphics
- **ArrowDraw**: Hand-drawn arrow (`color`, `width`)
- **ArrowHeads**: Multiple arrows to center (`color`, `width`)
- **Line**: Straight line (`x1`, `y1`, `x2`, `y2`, `width`, `color`)
- **VDragLine**: Draggable line (same props as Line)
- **Box**: Rectangle/circle (`shape`, `size`, `color`)
- **Thumb**: Thumbs up/down (`dir`, `color`, `width`)

### Interactive
- **QRCode**: QR code generator (`value`, `size`, `render-as`)
- **Kawaii**: Cute characters (`mood`, `size`, `color`)
- **Email**: Email formatter (`v` for address)

### Positioning
- Most components support `v-drag` directive for manual positioning

## Styling Classes

### Utility Classes (prefix: `ns-c-`)
- **Color shortcuts**: `ns-c-[color]-scheme` (e.g., `ns-c-sk-scheme` for sky)
- **Spacing**: `ns-c-tight`, `ns-c-supertight` (bullet spacing)
- **Alignment**: `ns-c-center-item` (auto margins)
- **Effects**: `ns-c-fader` (v-clicks fade), `ns-c-quote` (quote styling)
- **References**: `ns-c-cite`, `ns-c-cite-bl` (citations)
- **Links**: `ns-c-iconlink`, `ns-c-plainlink` (remove underlines)
- **Images**: `ns-c-imgtile` (grid images)

## Markdown Features

### Special Syntax
- **Highlighting**: `==text==` for highlighted text
- **HTML/CSS**: Requires blank lines before/after blocks
- **Slots**: Require blank line after declaration

## Branding

### Slide Info
- **Slug**: Set via `neversink_slug` in frontmatter
- **Hide info**: `slide_info: false`
- **Override**: Custom `slide-bottom.vue` or `global-bottom.vue`

## Quick Start

```yaml
---
theme: neversink
layout: cover
color: blue-light
---

# Title Here

Content here

:: slot-name ::

Slot content
```

## Key Patterns

1. **Color application**: Always pair scheme class with bind class
2. **Column sizing**: Use 12-unit grid system with `is-*` shortcuts
3. **Alignment**: Format as `[title]-[left][vert]-[right][vert]`
4. **Components**: Use v-drag for positioning, set fixed width
5. **Markdown**: Always add blank lines around HTML/slots
