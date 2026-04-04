# Finance App Theme Migration

This note describes how to integrate the FXAI GUI theme architecture into an existing finance macOS app without rewriting business logic.

## 1. Make the app shell theme-aware

- Create a shared `ThemeEnvironment`
- Register themes in `ThemeBootstrap`
- Inject the environment at the app root with `.environmentObject(...)`
- Keep domain and data services independent from theme types

## 2. Migrate views in layers

Recommended order:

1. App shell and navigation surfaces
2. Core finance dashboard surfaces
3. Shared cards, badges, charts, and footer strips
4. Secondary reports and operator tools
5. Legacy views last

## 3. Stop hardcoding visual values in views

Move raw values out of view bodies and into:

- `ThemeColors`
- `ThemeGradients`
- `ThemeShadows`
- `ThemeGlows`
- `ThemeCornerRadii`
- `ThemeTypography`
- `ThemeSpacing`
- `ThemeMaterials`
- `ThemeChartStyle`
- `ThemeComponentStyles`
- `ThemeRenderingPolicy`

## 4. Migrate layout decisions into the dashboard engine

Views should not manually infer compact vs wide behavior. Use:

- `DashboardLayoutInput`
- `DashboardLayoutOutput`
- `DashboardAdaptiveRules`
- `DashboardLayoutEngine`

This keeps layout decisions testable and future-theme-safe.

## 5. Preserve business logic boundaries

Do not pass theme objects into domain models, data pipelines, or research logic.

Keep business logic independent. Only UI and rendering layers should know about:

- `AppTheme`
- `ThemeEnvironment`
- rendering tiers
- component styles

## 6. Escalate rendering only where needed

Default to SwiftUI for composition and structure, then escalate specific components through rendering policy:

- SwiftUI for app shell, text, and simple structure
- CoreGraphics or Core Animation for premium shape, shadow, glass, and chart paths
- Metal only where bloom, compositing, or large-surface fidelity materially improves the result

## 7. Use calibration tools during migration

The GUI includes:

- SVG overlay comparison
- layout guides
- frame outlines
- theme inspector
- layout class display
- reduced-effects mode

Use those before changing token values blindly.
