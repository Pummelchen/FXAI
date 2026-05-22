# How To Add Theme V2

Theme V2 should be added by extending the existing architecture, not by copying view code and hardcoding values.

## Step 1. Create the theme type

Add a new `AppTheme` implementation under:

- `Sources/FXAIGUICore/Theme/`

At minimum provide:

- `themeID`
- `displayName`
- `colors`
- `gradients`
- `shadows`
- `glows`
- `cornerRadii`
- `typography`
- `spacing`
- `layoutMetrics`
- `materials`
- `chartStyle`
- `components`
- `renderingPolicy`

## Step 2. Register it

Add the theme to `ThemeBootstrap` through `ThemeRegistry`.

## Step 3. Keep component contracts semantic

Do not change views to consume raw theme values directly if a component style should own them.

Prefer:

- `theme.components.kpiCard`
- `theme.components.tooltip`
- `theme.components.footerStrip`

Instead of scattering token access across every view.

## Step 4. Reuse the same layout engine

Only change layout policy if the new theme truly requires a different composition.

If Theme V2 still uses the same dashboard semantics, keep:

- `DashboardLayoutInput`
- `DashboardLayoutOutput`
- `DashboardAdaptiveRules`

And only adjust:

- scale clamps
- spacing
- wide-band max width
- decorative reduction thresholds
- chart minimums

## Step 5. Add previews and tests

For every new theme:

- add preview scenes
- add registry tests
- add token tests
- add responsive layout assertions if the scale rules differ materially

## Step 6. Verify rendering-tier assumptions

If Theme V2 changes material richness, glow, blur, or chart fidelity, update `ThemeRenderingPolicy` rather than forcing every view to choose a tier ad hoc.
