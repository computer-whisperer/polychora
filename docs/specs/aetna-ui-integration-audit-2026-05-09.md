# Aetna UI integration audit

Date: 2026-05-09

## Verdict

Aetna is ready for an incremental Polychora integration, but not for a wholesale
egui replacement in one patch.

The low-risk first slice is to embed `aetna-vulkano` as an alternate overlay
path and port one self-contained surface: the hotbar, WAILA panel, or main-menu
root. The pause menu, settings, inventory, block GUI, and developer console
should stay on egui until the Aetna event bridge and render-pass embedding are
proven in-game.

## What looks ready

- `aetna-vulkano` builds cleanly on the current local Aetna workspace.
- `aetna-core` has broad unit coverage for layout, hit testing, focus, scroll,
  popovers, text inputs, icons, images, themes, animation, and linting.
- Dependency shape matches Polychora well: both are on `vulkano 0.35` and
  `winit 0.30`.
- Aetna's Vulkan backend is designed for custom hosts. The host owns the window,
  swapchain, command buffer, and frame cadence; Aetna owns UI state, layout,
  pipelines, text/icon/image paint, and input routing.
- Polychora's present render pass is compatible in shape with Aetna's simple
  Vulkan pass: one swapchain-format color attachment, one sample, clear/store,
  no depth attachment.

## Integration constraints

- Polychora currently converts egui primitives into its own HUD vertex batches
  and draws them inside the existing present pass. Aetna instead owns GPU
  pipelines and a `Runner`, so the first integration should call
  `Runner::prepare` before the pass and `Runner::draw` inside the existing pass
  after Polychora's scene/HUD draws.
- `Runner::render` is less suitable for Polychora's current frame graph because
  it owns render-pass lifetimes and clears the target. Use `Runner::draw` for
  the incremental path.
- Aetna input is event-routed and stateful. Polychora's egui UI currently
  mutates `App` fields directly during `run_egui_frame`. The port needs a small
  bridge that forwards pointer/key/text/wheel events to Aetna only while an
  Aetna surface is active, then applies returned `UiEvent`s to `App` state.
- Aetna needs device features merged at device creation:
  `aetna_vulkano::required_device_features()` currently requires
  `sample_rate_shading`. Polychora must OR this into its existing Vulkan feature
  request before creating an Aetna runner.
- Aetna bundles default fonts. That improves polish, but it will increase the
  binary unless `aetna-core` is used with narrowed font features.

## Suggested landing sequence

1. Add an optional `aetna-ui` feature and path dependencies on local Aetna
   crates while the project is still unpublished or under active development.
2. Add `RenderContext` ownership for an optional `aetna_vulkano::Runner`.
   Initialize it after swapchain/render context creation and update it on
   resize.
3. Add an `AetnaPaintData` or direct `El` build path for one passive surface,
   preferably WAILA or hotbar. Avoid text input in the first slice.
4. Route pointer hover/click and keyboard activation only for the active Aetna
   surface. Keep egui input handling in place for all existing menus.
5. Once one overlay draws and handles input, port the main-menu root. That gives
   the biggest polish win with low gameplay risk.
6. Port settings/inventory/block GUIs after text input, scroll, tabs, sliders,
   and controlled widgets are proven inside Polychora's real event loop.

## Checks run

```text
cargo check -p aetna-vulkano --target-dir /tmp/aetna-target
cargo test -p aetna-core --lib --target-dir /tmp/aetna-target
cargo check -p polychora
```

Results:

- `aetna-vulkano`: passed
- `aetna-core --lib`: 646 passed, 0 failed
- `polychora`: passed
