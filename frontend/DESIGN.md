# Design System Specification: DRIFT Platform
 
## 1. Overview & Creative North Star
**Creative North Star: "The Fluid Sanctuary"**
 
This design system moves away from the rigid, cold structures of traditional FinTech or data platforms. Instead, it embraces a high-end editorial feel that mimics the organic movement of deep-sea currents. We are not simply building a "dashboard"; we are curating a digital sanctuary that feels professional yet profoundly calm.
 
To break the "template" look, designers must embrace **Intentional Asymmetry**. Large `display-lg` typography should be used to anchor layouts, often offset from the main content column to create breathing room. Overlapping elements—such as a `surface-container` card slightly bleeding over a `primary-container` hero section—create a sense of bespoke, layered depth that feels expensive and intentional.
 
---
 
## 2. Colors & Surface Philosophy
The palette is a sophisticated deep-ocean spectrum, moving from the lightless depths of the abyss to the shimmering bioluminescence of the primary accents.
 
### The "No-Line" Rule
**Borders are a failure of hierarchy.** Within this system, 1px solid borders are strictly prohibited for sectioning. Boundaries must be defined through:
- **Tonal Shifts:** Placing a `surface-container-low` element against a `background` base.
- **Negative Space:** Using generous padding to define groupings.
- **Glassmorphism:** Using `backdrop-filter: blur(20px)` on semi-transparent `surface` colors to create soft, organic separation.
 
### Surface Hierarchy & Nesting
Treat the UI as a physical stack of semi-translucent materials. 
- **Base Layer:** `background` (#0d1417)
- **Secondary Logic:** `surface-container-low` (#161d20) for large structural areas.
- **Interactive Layers:** `surface-container` (#1a2124) or `surface-container-high` (#242b2e) for cards and modals.
 
### The "Glass & Gradient" Rule
To inject "soul" into the interface, avoid flat primary blocks. 
- **Signature Gradients:** Use a subtle linear gradient for primary CTAs, transitioning from `primary` (#6bd4f2) at the top-left to `primary-container` (#004c5c) at the bottom-right.
- **Atmospheric Glow:** Floating elements should use a `surface-variant` with 60% opacity and a heavy backdrop blur to pull the colors of the "ocean" through the component.
 
---
 
## 3. Typography
The typography has been pivoted to a humanistic, rounded aesthetic to reduce cognitive load and increase "soft" authority.
 
*   **Display & Headlines (Plus Jakarta Sans):** Chosen for its modern, geometric-yet-soft curves. Use `display-lg` (3.5rem) with tight letter-spacing (-0.02em) for hero moments to create a bold, editorial impact.
*   **Body & Utility (Manrope):** A highly legible, functional typeface that maintains a friendly, open character. 
 
**The Typographic Tension:**
The system relies on the contrast between the expansive `display` scales and the functional, compact `body-md`. This "Big/Small" tension is what makes the layout feel custom rather than standardized.
 
---
 
## 4. Elevation & Depth
We do not use shadows to represent "height" in a vacuum; we use them to represent "submersion" and "buoyancy."
 
*   **The Layering Principle:** Depth is achieved by stacking. A `surface-container-highest` card sitting on a `surface-container-low` section creates a natural "lift" through luminance alone.
*   **Ambient Shadows:** If a shadow is required for a floating modal, use: `box-shadow: 0 24px 48px -12px rgba(0, 0, 0, 0.5)`. The shadow must never be pure black; it should feel like an extension of the `background` color.
*   **The "Ghost Border" Fallback:** For accessibility in complex data views, use the `outline-variant` (#40484c) at **15% opacity**. This provides a guide for the eye without creating a hard visual "cage."
 
---
 
## 5. Components
 
### Buttons
- **Primary:** Gradient fill (`primary` to `primary-container`), `on-primary` text, and `full` roundedness. No border.
- **Secondary:** `surface-container-highest` fill with `primary` text. Provides a soft, integrated look.
- **Tertiary:** No background. `primary` text with an underline that appears only on hover.
 
### Cards & Lists
- **The Rule of Zero Dividers:** Horizontal lines are replaced by `1.5rem` (xl) vertical spacing. 
- **Hover States:** Instead of a border or shadow increase, a card should shift from `surface-container` to `surface-container-high` on hover, creating a "pulsing" effect like light hitting the water.
 
### Input Fields
- **Styling:** Use `surface-container-lowest` as the field background. 
- **Focus State:** Do not use a heavy outline. Use a 2px `primary` glow on the bottom edge only, or a subtle `primary` tint to the entire background of the field.
 
### Data Visualization (The "Current" Component)
- In DRIFT, data should feel moving. Use the `tertiary` (#9ad0d3) and `secondary` (#aecbd8) tokens for sparklines, ensuring lines are rounded (`stroke-linecap: round`) to match the typography.
 
---
 
## 6. Do's and Don'ts
 
### Do:
- **Do** use `display-lg` typography for more than just titles—use it for key metrics and "moment" statements.
- **Do** lean into the `full` roundedness (9999px) for pill-shaped badges and buttons to reinforce the "soft" mandate.
- **Do** use `surface-bright` (#333a3e) very sparingly, only for elements that need to break the deep-ocean immersion.
 
### Don't:
- **Don't** use 100% opaque `outline` colors. It shatters the "Fluid Sanctuary" atmosphere.
- **Don't** use sharp corners (0px-4px). If an element is not at least `0.5rem` (DEFAULT) rounded, it does not belong in this system.
- **Don't** use pure white (#FFFFFF) for text. Always use `on-surface` (#dde3e8) to maintain a soft, low-blue-light experience.