use common::{MatN, ModelInstance};
use higher_dimension_playground::render::{
    FrameParams, RenderBackend, RenderContext, RenderOptions, TetraFrameInput,
};
use polychora::content_registry::{ContentRegistry, MaterialResolver};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::device::{Device, Queue};
use vulkano::instance::Instance;

pub const ICON_SIZE: u32 = 64;
const SHEET_COLUMNS: u32 = 10;
const ICON_FOCAL_LENGTH_XY: f32 = 4.0;
const ICON_FOCAL_LENGTH_ZW: f32 = 6.0;

/// Build a 5x5 homogeneous rotation matrix in the XZ plane (angle in radians).
fn rot_xz(a: f32) -> MatN<5> {
    let (s, c) = a.sin_cos();
    let mut m = MatN::<5>::identity();
    m[[0, 0]] = c;
    m[[0, 2]] = -s;
    m[[2, 0]] = s;
    m[[2, 2]] = c;
    m
}

/// Build a 5x5 homogeneous rotation matrix in the YZ plane.
fn rot_yz(a: f32) -> MatN<5> {
    let (s, c) = a.sin_cos();
    let mut m = MatN::<5>::identity();
    m[[1, 1]] = c;
    m[[1, 2]] = -s;
    m[[2, 1]] = s;
    m[[2, 2]] = c;
    m
}

/// Build a 5x5 homogeneous rotation matrix in the XW plane.
fn rot_xw(a: f32) -> MatN<5> {
    let (s, c) = a.sin_cos();
    let mut m = MatN::<5>::identity();
    m[[0, 0]] = c;
    m[[0, 3]] = -s;
    m[[3, 0]] = s;
    m[[3, 3]] = c;
    m
}

/// Build a 5x5 homogeneous translation matrix.
fn translate(dx: f32, dy: f32, dz: f32, dw: f32) -> MatN<5> {
    let mut m = MatN::<5>::identity();
    m[[0, 4]] = dx;
    m[[1, 4]] = dy;
    m[[2, 4]] = dz;
    m[[3, 4]] = dw;
    m
}

fn icon_view_matrix() -> MatN<5> {
    let center = translate(-0.5, -0.5, -0.5, -0.5);
    let r1 = rot_xz(0.50);
    let r2 = rot_yz(-0.40);
    // Keep a subtle hidden-dimension tilt, but reduce it so icons stay crisp.
    let r3 = rot_xw(0.22);
    let push_back = translate(0.0, 0.15, 2.8, 2.8);
    push_back * r3 * r2 * r1 * center
}

fn icon_render_options() -> RenderOptions {
    RenderOptions {
        do_frame_clear: true,
        do_raster: true,
        do_raytrace: false,
        render_backend: RenderBackend::TetraRaster,
        do_edges: false,
        do_tetrahedron_edges: false,
        do_navigation_hud: false,
        zw_angle_color_shift_enabled: false,
        // Keep padding flags clear so rasterizer stays in non-overlay mode.
        zw_angle_color_shift_strength: 0.0,
        prepare_render_screenshot: true,
        ..Default::default()
    }
}

/// Render a spawn egg icon (rounded rect + highlight dot) into an RGBA buffer.
fn render_spawn_egg_icon_rgba(base_color: [u8; 3]) -> Vec<u8> {
    let size = ICON_SIZE as usize;
    let mut pixels = vec![0u8; size * size * 4];

    // Brighten dark colors so the egg is visible
    let [r, g, b] = base_color;
    let (r, g, b) = if (r as u16 + g as u16 + b as u16) < 80 {
        (
            r.saturating_add(60),
            g.saturating_add(60),
            b.saturating_add(60),
        )
    } else {
        (r, g, b)
    };

    let center_x = size as f32 / 2.0;
    let center_y = size as f32 / 2.0;
    let half = size as f32 / 2.0;
    let rounding = half * 0.4;

    // Dot center and radius (matching the egui version)
    let dot_cx = center_x;
    let dot_cy = center_y - half * 0.15;
    let dot_r = half * 0.12;
    let dot_color = (
        r.saturating_add(80),
        g.saturating_add(80),
        b.saturating_add(80),
    );

    for py in 0..size {
        for px in 0..size {
            let fx = px as f32 + 0.5;
            let fy = py as f32 + 0.5;

            // SDF rounded rect: distance from center, reduced by rounding
            let dx = (fx - center_x).abs() - (half - rounding);
            let dy = (fy - center_y).abs() - (half - rounding);
            let dist_rect = dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0) - rounding;

            if dist_rect < 0.5 {
                let rect_alpha = (0.5 - dist_rect).clamp(0.0, 1.0);

                // Check if inside highlight dot
                let dist_dot = ((fx - dot_cx).powi(2) + (fy - dot_cy).powi(2)).sqrt() - dot_r;

                let (pr, pg, pb) = if dist_dot < 0.5 {
                    let dot_alpha = (0.5 - dist_dot).clamp(0.0, 1.0);
                    (
                        (r as f32 * (1.0 - dot_alpha) + dot_color.0 as f32 * dot_alpha) as u8,
                        (g as f32 * (1.0 - dot_alpha) + dot_color.1 as f32 * dot_alpha) as u8,
                        (b as f32 * (1.0 - dot_alpha) + dot_color.2 as f32 * dot_alpha) as u8,
                    )
                } else {
                    (r, g, b)
                };

                let offset = (py * size + px) * 4;
                pixels[offset] = pr;
                pixels[offset + 1] = pg;
                pixels[offset + 2] = pb;
                pixels[offset + 3] = (rect_alpha * 255.0) as u8;
            }
        }
    }

    pixels
}

/// A sprite sheet containing all material and spawn egg icons packed into a
/// single texture.  Keyed exclusively by `(namespace, texture_id)` — callers
/// convert block/entity identity to a `TextureRef` first, then look up the UV.
pub struct MaterialIconSheet {
    /// RGBA pixel data for the entire sprite sheet
    pub pixels: Vec<u8>,
    /// Width of the sprite sheet in pixels
    pub width: u32,
    /// Height of the sprite sheet in pixels
    pub height: u32,
    /// Map from (texture_namespace, texture_id) to UV rectangle
    /// [u_min, v_min, u_max, v_max].
    uv_rects: HashMap<(u32, u32), [f32; 4]>,
    /// Per-texture images for Aetna. Aetna owns GPU upload/cache for these,
    /// so callers do not need to route through the legacy egui sprite sheet.
    aetna_images: HashMap<(u32, u32), aetna_core::image::Image>,
}

impl MaterialIconSheet {
    /// Get the UV rectangle for a texture by (namespace, texture_id), or None
    /// if not found.
    pub fn uv_rect(&self, namespace: u32, texture_id: u32) -> Option<[f32; 4]> {
        self.uv_rects.get(&(namespace, texture_id)).copied()
    }

    /// Get an Aetna image for a texture by (namespace, texture_id), or None
    /// if not found.
    pub fn aetna_image(&self, namespace: u32, texture_id: u32) -> Option<aetna_core::image::Image> {
        self.aetna_images.get(&(namespace, texture_id)).cloned()
    }
}

/// Generate a sprite sheet containing all material icons packed into a grid
/// by rendering the tetrahedron pipeline offscreen on GPU, plus CPU-rendered
/// spawn egg icons.
///
/// `pending_texture_uploads` provides the WASM plugin's 3D textures so migrated
/// blocks (those using texture pool tokens) render correctly in the offscreen
/// context.
pub fn generate_material_icon_sheet_gpu(
    device: Arc<Device>,
    queue: Arc<Queue>,
    instance: Arc<Instance>,
    content_registry: &ContentRegistry,
    material_resolver: &MaterialResolver,
    pending_texture_uploads: &[polychora::plugin_loader::PendingTextureUpload],
) -> Option<MaterialIconSheet> {
    let num_blocks = content_registry.block_count() as u32;
    let num_eggs: u32 = content_registry.spawnable_entities().count() as u32;
    let total_slots = num_blocks + num_eggs;
    let rows = total_slots.div_ceil(SHEET_COLUMNS);
    let sheet_w = SHEET_COLUMNS * ICON_SIZE;
    let sheet_h = rows * ICON_SIZE;

    let mut pixels = vec![0u8; (sheet_w * sheet_h * 4) as usize];
    let mut uv_rects = HashMap::new();
    let mut aetna_images = HashMap::new();
    let icon_pixel_len = (ICON_SIZE * ICON_SIZE * 4) as usize;

    let mut offscreen = RenderContext::new(
        device.clone(),
        queue.clone(),
        instance,
        None,
        [ICON_SIZE, ICON_SIZE, 1],
    );

    // Upload WASM plugin textures so migrated blocks render with their 3D
    // textures instead of falling back to magenta.
    for upload in pending_texture_uploads {
        offscreen.upload_texture_3d(
            &upload.data,
            upload.width,
            upload.height,
            upload.depth,
            upload.format,
        );
    }

    let view_matrix: ndarray::Array2<f32> = icon_view_matrix().into();

    // Phase 1: GPU-rendered block icons
    for (idx, entry) in content_registry.all_blocks_ordered().enumerate() {
        let col = (idx as u32) % SHEET_COLUMNS;
        let row = (idx as u32) / SHEET_COLUMNS;
        let token = material_resolver.resolve_block(entry.namespace, entry.block_type);

        let model_instance = [ModelInstance {
            model_transform: MatN::<5>::identity(),
            cell_material_ids: [token as u32; 8],
        }];

        offscreen.render_tetra_frame(
            device.clone(),
            queue.clone(),
            FrameParams {
                view_matrix: view_matrix.clone(),
                time_ticks_ms: 0,
                focal_length_xy: ICON_FOCAL_LENGTH_XY,
                focal_length_zw: ICON_FOCAL_LENGTH_ZW,
                render_options: icon_render_options(),
            },
            TetraFrameInput {
                model_instances: &model_instance,
            },
        );
        let Some((icon_w, icon_h, raw)) = offscreen.capture_rendered_frame_rgba8(true) else {
            eprintln!(
                "Failed to capture offscreen material icon render for material {}",
                token
            );
            return None;
        };
        if icon_w != ICON_SIZE || icon_h != ICON_SIZE || raw.len() != icon_pixel_len {
            eprintln!(
                "Unexpected offscreen icon size for material {}: {}x{} ({} bytes)",
                token,
                icon_w,
                icon_h,
                raw.len()
            );
            return None;
        }

        copy_icon_to_sheet(&mut pixels, &raw, col, row, sheet_w);

        let uv = compute_uv(col, row, sheet_w, sheet_h);
        let aetna_image = aetna_core::image::Image::from_rgba8(ICON_SIZE, ICON_SIZE, raw);
        uv_rects
            .entry((entry.texture.namespace, entry.texture.texture_id))
            .or_insert(uv);
        aetna_images
            .entry((entry.texture.namespace, entry.texture.texture_id))
            .or_insert_with(|| aetna_image.clone());

        // Phase 3 inline: namespace-0 aliases for migrated blocks.
        // Blocks whose texture uses a plugin namespace (e.g. 0x706f6c79)
        // also need a (0, texture_id) entry so entity model_textures
        // (which use tex() → namespace 0) can resolve to an icon.
        if entry.texture.namespace != 0 {
            uv_rects.entry((0, entry.texture.texture_id)).or_insert(uv);
            aetna_images
                .entry((0, entry.texture.texture_id))
                .or_insert_with(|| aetna_image.clone());
        }
    }

    // Phase 2: CPU-rendered spawn egg icons
    let mut egg_idx = num_blocks;
    for entity in content_registry.spawnable_entities() {
        if entity.spawn_egg_texture_id == 0 {
            egg_idx += 1;
            continue;
        }

        let col = egg_idx % SHEET_COLUMNS;
        let row = egg_idx / SHEET_COLUMNS;

        let raw = render_spawn_egg_icon_rgba(entity.base_color);

        copy_icon_to_sheet(&mut pixels, &raw, col, row, sheet_w);

        let uv = compute_uv(col, row, sheet_w, sheet_h);
        let aetna_image = aetna_core::image::Image::from_rgba8(ICON_SIZE, ICON_SIZE, raw);
        uv_rects
            .entry((0, entity.spawn_egg_texture_id))
            .or_insert(uv);
        aetna_images
            .entry((0, entity.spawn_egg_texture_id))
            .or_insert(aetna_image);

        egg_idx += 1;
    }

    Some(MaterialIconSheet {
        pixels,
        width: sheet_w,
        height: sheet_h,
        uv_rects,
        aetna_images,
    })
}

fn copy_icon_to_sheet(pixels: &mut [u8], raw: &[u8], col: u32, row: u32, sheet_w: u32) {
    let dst_x = col * ICON_SIZE;
    let dst_y = row * ICON_SIZE;
    for py in 0..ICON_SIZE {
        let src_offset = (py * ICON_SIZE * 4) as usize;
        let dst_offset = ((dst_y + py) * sheet_w + dst_x) as usize * 4;
        pixels[dst_offset..dst_offset + (ICON_SIZE * 4) as usize]
            .copy_from_slice(&raw[src_offset..src_offset + (ICON_SIZE * 4) as usize]);
    }
}

fn compute_uv(col: u32, row: u32, sheet_w: u32, sheet_h: u32) -> [f32; 4] {
    let dst_x = col * ICON_SIZE;
    let dst_y = row * ICON_SIZE;
    let u_min = dst_x as f32 / sheet_w as f32;
    let v_min = dst_y as f32 / sheet_h as f32;
    let u_max = (dst_x + ICON_SIZE) as f32 / sheet_w as f32;
    let v_max = (dst_y + ICON_SIZE) as f32 / sheet_h as f32;
    [u_min, v_min, u_max, v_max]
}
