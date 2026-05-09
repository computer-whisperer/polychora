use super::*;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum RenderBackend {
    /// Legacy behavior: derive backend from existing booleans.
    #[default]
    Auto,
    /// Existing tetrahedron tile raster path.
    TetraRaster,
    /// Existing tetrahedron raytrace path.
    TetraRaytrace,
    /// New voxel traversal engine path (currently placeholder).
    VoxelTraversal,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum VteDisplayMode {
    #[default]
    Integral,
    Slice,
    ThickSlice,
    DebugCompare,
    DebugIntegral,
}

impl VteDisplayMode {
    pub(super) fn as_u32(self) -> u32 {
        match self {
            Self::Integral => 0,
            Self::Slice => 1,
            Self::ThickSlice => 2,
            Self::DebugCompare => 3,
            Self::DebugIntegral => 4,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Integral => "integral",
            Self::Slice => "slice",
            Self::ThickSlice => "thick_slice",
            Self::DebugCompare => "debug_compare",
            Self::DebugIntegral => "debug_integral",
        }
    }
}

impl RenderBackend {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::TetraRaster => "tetra_raster",
            Self::TetraRaytrace => "tetra_raytrace",
            Self::VoxelTraversal => "voxel_traversal",
        }
    }
}

pub const OVERLAY_EDGE_TAG_MASK: u32 = 0x4000_0000;
pub const OVERLAY_EDGE_TAG_TARGET: u32 = 1;
pub const OVERLAY_EDGE_TAG_PLACE: u32 = 2;
pub const OVERLAY_EDGE_TAG_DIAG_BASE: u32 = 16;
pub const OVERLAY_EDGE_DIAG_TAG_COUNT: u32 = 8;
pub const OVERLAY_EDGE_TAG_REGION_BRANCH: u32 = OVERLAY_EDGE_TAG_DIAG_BASE;
pub const OVERLAY_EDGE_TAG_REGION_EMPTY: u32 = OVERLAY_EDGE_TAG_DIAG_BASE + 1;
pub const OVERLAY_EDGE_TAG_REGION_UNIFORM: u32 = OVERLAY_EDGE_TAG_DIAG_BASE + 2;
pub const OVERLAY_EDGE_TAG_REGION_CHUNK_ARRAY: u32 = OVERLAY_EDGE_TAG_DIAG_BASE + 3;
pub const OVERLAY_EDGE_TAG_REGION_PROCEDURAL: u32 = OVERLAY_EDGE_TAG_DIAG_BASE + 4;

#[derive(Clone, Debug)]
pub struct CustomOverlayLine {
    pub start_ndc: [f32; 2],
    pub end_ndc: [f32; 2],
    pub color: [f32; 4],
    /// Optional style id consumed by present line shader (`>0.5` enables adaptive mode).
    pub style: f32,
}

#[derive(Clone, Debug)]
pub struct HudPlayerTag {
    pub text: String,
    pub anchor_ndc: [f32; 2],
    pub scale: f32,
    pub bg_alpha: f32,
    pub text_color: [f32; 4],
    pub border_color: [f32; 4],
    pub connector_color: [f32; 4],
    pub target_rect_ndc: Option<[f32; 4]>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HudReadoutMode {
    Full,
    CompactVectors,
}

#[derive(Clone, Debug)]
pub struct EguiPaintVertex {
    pub position_px: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EguiTextureSlot {
    EguiAtlas,
    MaterialIcons,
}

#[derive(Clone, Debug)]
pub struct EguiPaintMesh {
    pub clip_rect_px: [f32; 4],
    pub vertices: Vec<EguiPaintVertex>,
    pub texture_slot: EguiTextureSlot,
}

#[derive(Clone, Debug)]
pub struct EguiTextureUpdate {
    pub size: [u32; 2],
    pub pos: Option<[u32; 2]>,
    pub pixels: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct EguiPaintData {
    pub texture_updates: Vec<EguiTextureUpdate>,
    pub meshes: Vec<EguiPaintMesh>,
}

pub fn generate_tesseract_tetrahedrons() -> Vec<ModelTetrahedron> {
    super::geometry::generate_tesseract_tetrahedrons()
}

pub fn generate_tesseract_edges() -> Vec<common::ModelEdge> {
    super::geometry::generate_tesseract_edges()
}

pub struct RenderOptions {
    pub do_frame_clear: bool,
    pub do_raster: bool,
    pub do_raytrace: bool,
    pub render_backend: RenderBackend,
    pub vte_max_trace_steps: u32,
    pub vte_max_trace_distance: f32,
    pub vte_display_mode: VteDisplayMode,
    pub vte_slice_layer: Option<u32>,
    pub vte_thick_half_width: u32,
    pub vte_reference_compare: bool,
    pub vte_reference_mismatch_only: bool,
    pub vte_compare_slice_only: bool,
    pub vte_integral_sky_emissive_tweak: bool,
    pub vte_integral_sky_scale: f32,
    pub vte_integral_hit_emissive_boost: f32,
    pub vte_integral_log_merge_tweak: bool,
    pub vte_integral_log_merge_k: f32,
    pub zw_angle_color_shift_enabled: bool,
    pub zw_angle_color_shift_strength: f32,
    pub vte_highlight_hit_min: Option<[f32; 4]>,
    pub vte_highlight_hit_max: [f32; 4],
    pub vte_highlight_face_axis: u32,
    pub vte_highlight_face_sign: i32,
    pub do_edges: bool,
    pub do_tetrahedron_edges: bool,
    pub do_navigation_hud: bool,
    pub custom_overlay_lines: Vec<CustomOverlayLine>,
    pub custom_overlay_edge_instances: Vec<common::ModelInstance>,
    pub take_framebuffer_screenshot: bool,
    pub prepare_render_screenshot: bool,
    pub hud_readout_mode: HudReadoutMode,
    pub hud_rotation_label: Option<String>,
    pub hud_target_hit_voxel: Option<[i32; 4]>,
    pub hud_target_hit_face: Option<[i32; 4]>,
    pub hud_player_tags: Vec<HudPlayerTag>,
    pub egui_paint: Option<EguiPaintData>,
    pub aetna_ui: Option<aetna_core::El>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            do_frame_clear: false,
            do_raster: true,
            do_raytrace: false,
            render_backend: RenderBackend::Auto,
            vte_max_trace_steps: 320,
            vte_max_trace_distance: 160.0,
            vte_display_mode: VteDisplayMode::Integral,
            vte_slice_layer: None,
            vte_thick_half_width: 2,
            vte_reference_compare: false,
            vte_reference_mismatch_only: false,
            vte_compare_slice_only: false,
            vte_integral_sky_emissive_tweak: true,
            vte_integral_sky_scale: 0.40,
            vte_integral_hit_emissive_boost: 0.025,
            vte_integral_log_merge_tweak: true,
            vte_integral_log_merge_k: 8.0,
            zw_angle_color_shift_enabled: false,
            zw_angle_color_shift_strength: 0.35,
            vte_highlight_hit_min: None,
            vte_highlight_hit_max: [0.0; 4],
            vte_highlight_face_axis: 0,
            vte_highlight_face_sign: 0,
            do_edges: false,
            do_tetrahedron_edges: false,
            do_navigation_hud: false,
            custom_overlay_lines: Vec::new(),
            custom_overlay_edge_instances: Vec::new(),
            take_framebuffer_screenshot: false,
            prepare_render_screenshot: false,
            hud_readout_mode: HudReadoutMode::Full,
            hud_rotation_label: None,
            hud_target_hit_voxel: None,
            hud_target_hit_face: None,
            hud_player_tags: Vec::new(),
            egui_paint: None,
            aetna_ui: None,
        }
    }
}

pub struct FrameParams {
    pub view_matrix: ndarray::Array2<f32>,
    pub time_ticks_ms: u32,
    pub focal_length_xy: f32,
    pub focal_length_zw: f32,
    pub render_options: RenderOptions,
}

pub struct TetraFrameInput<'a> {
    pub model_instances: &'a [common::ModelInstance],
}
