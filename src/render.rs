mod buffers;
#[cfg(test)]
mod bvh_topology_tests;
mod capture;
mod context_init;
mod geometry;
mod hud;
mod overlay;
mod pipelines;
mod profiler;
mod texture_pool;
mod types;
mod vte;

use self::buffers::{LiveBuffers, OneTimeBuffers, SizedBuffers};
pub use self::buffers::{VoxelBufferCapacities, VoxelGpuBuffers};
use self::geometry::{mat5_mul_vec5, project_view_point_to_ndc, transform_model_point};
use self::hud::{
    build_font_atlas, load_hud_font, map_to_panel, ndc_to_pixels, pixels_to_ndc, push_cross,
    push_filled_rect_quads, push_line, push_minecraft_crosshair, push_rect, push_text_lines,
    push_text_quads, HudResources, HudVertex, LineVertex, OverlayLine, HUD_VERTEX_CAPACITY,
};
use self::pipelines::{ComputePipelineContext, PresentPipelineContext};
use self::profiler::{GpuProfiler, PROFILER_MAX_TIMESTAMPS};
use self::texture_pool::TexturePool;
pub use self::types::*;
pub use self::vte::{
    GpuVoxelChunkBvhNode, GpuVoxelChunkHeader, GpuVoxelLeafHeader, VoxelChunkBvhNodeRangeWrite,
    VoxelChunkHeaderRangeWrite, VoxelFrameDirtyRanges, VoxelFrameInput, VoxelLeafHeaderRangeWrite,
    VoxelMutationBatch, VoxelU32RangeWrite, VteDebugCounters, VTE_LEAF_CHUNK_ENTRY_EMPTY,
    VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG, VTE_LEAF_KIND_UNIFORM, VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
    VTE_MAX_DENSE_CHUNKS, VTE_REGION_BVH_INVALID_NODE, VTE_REGION_BVH_NODE_CAPACITY,
    VTE_REGION_BVH_NODE_FLAG_LEAF, VTE_REGION_LEAF_CAPACITY, VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY,
};
use ab_glyph::FontArc;
use bytemuck::Zeroable;
use common::ModelTetrahedron;
use exr::prelude::WritableImage;
use glam::{Vec2, Vec4};
use image::{ImageBuffer, Rgba};
use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
    CopyImageToBufferInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::{
    DescriptorSetAllocator, StandardDescriptorSetAllocator,
    StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::{
    DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::sys::ImageCreateInfo;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageType, ImageUsage};
use vulkano::instance::Instance;
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::sync::PipelineStage;
use vulkano::{sync, Validated, VulkanError};
use winit::dpi::PhysicalSize;
use winit::window::Window;

const VTE_LOD_TINT_ENV: &str = "R4D_VTE_LOD_TINT";
const VTE_ENTITY_LINEAR_ONLY_ENV: &str = "R4D_VTE_ENTITY_LINEAR_ONLY";
const VTE_ENTITY_BVH_COMPARE_ENV: &str = "R4D_VTE_ENTITY_BVH_COMPARE";
const VTE_ENTITY_DIAG_ENV: &str = "R4D_VTE_ENTITY_DIAG";
const VTE_ENTITY_DIAG_VERBOSE_ENV: &str = "R4D_VTE_ENTITY_DIAG_VERBOSE";
const VTE_ENTITY_DIAG_BVH_READBACK_ENV: &str = "R4D_VTE_ENTITY_DIAG_BVH_READBACK";
const VTE_ENTITY_DIAG_BVH_TOPOLOGY_ENV: &str = "R4D_VTE_ENTITY_DIAG_BVH_TOPOLOGY";
const VTE_ENTITY_DIAG_BVH_INTERVAL_ENV: &str = "R4D_VTE_ENTITY_DIAG_BVH_INTERVAL";
const VTE_STAGE_A_BREAKDOWN_ENV: &str = "R4D_VTE_STAGEA_BREAKDOWN";
const VTE_STAGE_A_BREAKDOWN_INTERVAL_ENV: &str = "R4D_VTE_STAGEA_BREAKDOWN_INTERVAL";
const VTE_WORLD_BVH_RAY_DIAG_ENV: &str = "R4D_VTE_WORLD_BVH_RAY_DIAG";
const VTE_WORLD_BVH_RAY_DIAG_SAMPLES_ENV: &str = "R4D_VTE_WORLD_BVH_RAY_DIAG_SAMPLES";
const VTE_WORLD_BVH_RAY_DIAG_INTERVAL_ENV: &str = "R4D_VTE_WORLD_BVH_RAY_DIAG_INTERVAL";
const VTE_ENTITY_DIAG_DEFAULT_INTERVAL: usize = 120;
const VTE_STAGE_A_BREAKDOWN_DEFAULT_INTERVAL: usize = 120;
const VTE_WORLD_BVH_RAY_DIAG_DEFAULT_SAMPLES: usize = 64;
const VTE_WORLD_BVH_RAY_DIAG_DEFAULT_INTERVAL: usize = 60;
const VTE_ENTITY_DIAG_TRANSFORM_ABS_WARN: f32 = 16_384.0;
// Force an occasional full rebuild after repeated refits to keep traversal quality healthy.
const VTE_ENTITY_BVH_REFIT_REBUILD_INTERVAL: usize = 120;
// Keep in sync with OVERLAY_RASTER_SCALE in `slang-shaders/src/rasterizer.slang`.
const VTE_OVERLAY_RASTER_SCALE: u32 = 3;
const WORKING_FLAG_VTE_COLLAPSED: u32 = 1u32 << 0u32;
const WORKING_FLAG_ZW_ANGLE_COLOR_SHIFT: u32 = 1u32 << 1u32;
const WORKING_ZW_SHIFT_STRENGTH_SHIFT: u32 = 8u32;
const VTE_CPU_MISS_REASON_NONE: u32 = 0;
const VTE_CPU_MISS_REASON_TOUCHED_VISIBLE_CHUNK: u32 = 1;
const VTE_CPU_MISS_REASON_VOXEL_BUDGET: u32 = 2;
const VTE_CPU_MISS_REASON_CHUNK_BUDGET: u32 = 3;
const VTE_CPU_MISS_REASON_MAX_DISTANCE: u32 = 4;

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

pub(super) const fn vte_diagnostics_feature_enabled() -> bool {
    cfg!(feature = "vte-diagnostics")
}

fn vte_lod_tint_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_LOD_TINT_ENV))
}

fn vte_entity_linear_only_enabled() -> bool {
    if !vte_diagnostics_feature_enabled() {
        return false;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_ENTITY_LINEAR_ONLY_ENV))
}

fn vte_entity_bvh_compare_enabled() -> bool {
    if !vte_diagnostics_feature_enabled() {
        return false;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_ENTITY_BVH_COMPARE_ENV))
}

fn vte_stage_a_breakdown_enabled() -> bool {
    if !vte_diagnostics_feature_enabled() {
        return false;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_STAGE_A_BREAKDOWN_ENV))
}

fn vte_world_bvh_ray_diag_env_enabled() -> bool {
    if !vte_diagnostics_feature_enabled() {
        return false;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_WORLD_BVH_RAY_DIAG_ENV))
}

fn env_usize(name: &str, default_value: usize) -> usize {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|v| *v > 0)
            .unwrap_or(default_value),
        Err(_) => default_value,
    }
}

fn clamp_dirty_range(
    range: Option<std::ops::Range<usize>>,
    len: usize,
) -> Option<std::ops::Range<usize>> {
    let range = range?;
    let start = range.start.min(len);
    let end = range.end.min(len);
    if start >= end {
        return None;
    }
    Some(start..end)
}

fn clamp_write_span(
    start: u32,
    values_len: usize,
    limit: usize,
) -> Option<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    if values_len == 0 || limit == 0 {
        return None;
    }
    let dst_start = (start as usize).min(limit);
    if dst_start >= limit {
        return None;
    }
    let dst_end = dst_start.saturating_add(values_len).min(limit);
    if dst_end <= dst_start {
        return None;
    }
    let src_len = dst_end - dst_start;
    Some((dst_start..dst_end, 0..src_len))
}

/// Collection of all shader modules loaded from Slang-compiled SPIR-V
struct ShaderModules {
    // Present shaders
    line_vs: Arc<ShaderModule>,
    line_fs: Arc<ShaderModule>,
    buffer_vs: Arc<ShaderModule>,
    buffer_fs: Arc<ShaderModule>,
    // HUD shaders
    hud_vs: Arc<ShaderModule>,
    hud_fs: Arc<ShaderModule>,
    // Compute shaders
    tetrahedron_cs: Arc<ShaderModule>,
    edge_cs: Arc<ShaderModule>,
    tetrahedron_pixel_cs: Arc<ShaderModule>,
    raytrace_preprocess: Arc<ShaderModule>,
    entity_instance_aabb_preprocess: Arc<ShaderModule>,
    raytrace_pixel: Arc<ShaderModule>,
    raytrace_clear: Arc<ShaderModule>,
    // Voxel traversal engine (VTE) compute shaders
    voxel_trace_stage_a_integral_fused: Arc<ShaderModule>,
    voxel_trace_stage_a_layered: Arc<ShaderModule>,
    voxel_display_stage_b: Arc<ShaderModule>,
    // Tile binning
    bin_tets_cs: Arc<ShaderModule>,
    // BVH compute shaders
    bvh_scene_bounds: Arc<ShaderModule>,
    bvh_morton_codes: Arc<ShaderModule>,
    bvh_bitonic_sort_local: Arc<ShaderModule>,
    bvh_bitonic_sort: Arc<ShaderModule>,
    bvh_bitonic_sort_local_merge: Arc<ShaderModule>,
    bvh_init_leaves: Arc<ShaderModule>,
    bvh_build_tree: Arc<ShaderModule>,
    bvh_link_parents: Arc<ShaderModule>,
    bvh_propagate_aabbs: Arc<ShaderModule>,
}

const LINE_VERTEX_CAPACITY: usize = 100_000;
const HUD_BREADCRUMB_CAPACITY: usize = 128;
const HUD_BREADCRUMB_MIN_STEP: f32 = 0.2;
const FRAMES_IN_FLIGHT: usize = 2;

struct FrameInFlight {
    live_buffers: LiveBuffers,
    line_vertexes_buffer: Subbuffer<[LineVertex]>,
    hud_vertex_buffer: Option<Subbuffer<[HudVertex]>>,
    hud_descriptor_set: Option<Arc<DescriptorSet>>,
    egui_descriptor_set: Option<Arc<DescriptorSet>>,
    material_icons_descriptor_set: Option<Arc<DescriptorSet>>,
    sized_descriptor_set: Arc<DescriptorSet>,
    cpu_clipped_tet_count_buffer: Subbuffer<[u32]>,
    query_pool: Arc<QueryPool>,
    fence: Option<Box<dyn GpuFuture>>,
    vte_compare_enabled: bool,
    vte_world_bvh_ray_diag_enabled: bool,
    last_voxel_metadata_generation: Option<u64>,
    vte_world_bvh_ray_diag_expected: Vec<VteWorldBvhRayDiagExpectedRecord>,
    vte_entity_diag_copy_scheduled: bool,
    vte_entity_diag_non_voxel_tet_count: usize,
}

struct AetnaOverlay {
    runner: aetna_vulkano::Runner,
}

struct EguiResources {
    atlas_view: Arc<ImageView>,
    atlas_sampler: Arc<Sampler>,
    texture_size: [u32; 2],
    texture_pixels: Vec<u8>,
    retired_atlas_views: Vec<(Arc<ImageView>, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HudTextureSlot {
    Hud,
    EguiAtlas,
    MaterialIcons,
}

#[derive(Clone, Copy)]
struct HudDrawBatch {
    first_vertex: u32,
    vertex_count: u32,
    scissor: Scissor,
    texture_slot: HudTextureSlot,
}

fn create_rgba8_srgb_texture_view(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Arc<ImageView> {
    let staging_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        pixels.iter().copied(),
    )
    .unwrap();

    let atlas_image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
            extent: [width.max(1), height.max(1), 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();

    let mut upload_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    upload_builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            staging_buffer,
            atlas_image.clone(),
        ))
        .unwrap();
    let upload_cmd = upload_builder.build().unwrap();
    let upload_future = sync::now(queue.device().clone())
        .then_execute(queue.clone(), upload_cmd)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    upload_future.wait(None).unwrap();

    ImageView::new_default(atlas_image).unwrap()
}

fn create_hud_descriptor_set(
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    hud_vertex_buffer: Subbuffer<[HudVertex]>,
    atlas_view: Arc<ImageView>,
    atlas_sampler: Arc<Sampler>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator,
        descriptor_set_layout,
        [
            WriteDescriptorSet::buffer(0, hud_vertex_buffer),
            WriteDescriptorSet::image_view_sampler(1, atlas_view, atlas_sampler),
        ],
        [],
    )
    .unwrap()
}

fn model_instance_is_finite(instance: &common::ModelInstance) -> bool {
    for row in 0..5 {
        for col in 0..5 {
            if !instance.model_transform[[row, col]].is_finite() {
                return false;
            }
        }
    }
    true
}

/// Filter out `ModelInstance`s with non-finite transforms.  Returns the
/// original slice (borrowed) when every element is finite, avoiding
/// allocation on the common path.  `dropped` is incremented for each
/// discarded instance.
fn filter_finite_instances<'a>(
    instances: &'a [common::ModelInstance],
    dropped: &mut usize,
) -> Cow<'a, [common::ModelInstance]> {
    if instances.iter().all(model_instance_is_finite) {
        Cow::Borrowed(instances)
    } else {
        let mut filtered = Vec::with_capacity(instances.len());
        for instance in instances.iter().copied() {
            if model_instance_is_finite(&instance) {
                filtered.push(instance);
            } else {
                *dropped += 1;
            }
        }
        Cow::Owned(filtered)
    }
}

fn model_instance_transform_extrema(instances: &[common::ModelInstance]) -> (f32, f32, usize) {
    let mut translation_abs_max = 0.0f32;
    let mut basis_abs_max = 0.0f32;
    let mut outlier_count = 0usize;

    for instance in instances {
        let mut instance_translation_abs_max = 0.0f32;
        let mut instance_basis_abs_max = 0.0f32;
        for axis in 0..4 {
            instance_translation_abs_max =
                instance_translation_abs_max.max(instance.model_transform[[axis, 4]].abs());
            for basis_axis in 0..4 {
                instance_basis_abs_max =
                    instance_basis_abs_max.max(instance.model_transform[[axis, basis_axis]].abs());
            }
        }
        translation_abs_max = translation_abs_max.max(instance_translation_abs_max);
        basis_abs_max = basis_abs_max.max(instance_basis_abs_max);
        if instance_translation_abs_max > VTE_ENTITY_DIAG_TRANSFORM_ABS_WARN
            || instance_basis_abs_max > VTE_ENTITY_DIAG_TRANSFORM_ABS_WARN
        {
            outlier_count += 1;
        }
    }

    (translation_abs_max, basis_abs_max, outlier_count)
}

struct BvhTopologySummary {
    total_nodes: usize,
    internal_nodes: usize,
    internal_ready: usize,
    invalid_child_edges: usize,
    self_child_edges: usize,
    nodes_without_parent_excluding_root: usize,
    nodes_with_multiple_parents: usize,
    unreachable_internal_nodes: usize,
    unreachable_leaf_nodes: usize,
    leaf_invalid_tetra_indices: usize,
    leaf_duplicate_tetra_indices: usize,
    leaf_missing_tetra_indices: usize,
}

fn summarize_bvh_topology(
    bvh_nodes: &[common::BVHNode],
    num_tetrahedrons: usize,
) -> Option<BvhTopologySummary> {
    if num_tetrahedrons == 0 {
        return None;
    }
    let total_nodes = num_tetrahedrons.checked_mul(2)?.checked_sub(1)?;
    if total_nodes == 0 || total_nodes > bvh_nodes.len() {
        return None;
    }
    let internal_nodes = num_tetrahedrons.saturating_sub(1);
    let mut parent_ref_counts = vec![0u32; total_nodes];
    let mut invalid_child_edges = 0usize;
    let mut self_child_edges = 0usize;
    let mut internal_ready = 0usize;

    for (idx, node) in bvh_nodes.iter().enumerate().take(internal_nodes) {
        if node.atomic_visit_count >= 2 {
            internal_ready += 1;
        }
        for child in [node.left_child, node.right_child] {
            if child == u32::MAX {
                invalid_child_edges += 1;
                continue;
            }
            let child_idx = child as usize;
            if child_idx >= total_nodes {
                invalid_child_edges += 1;
                continue;
            }
            if child_idx == idx {
                self_child_edges += 1;
            }
            parent_ref_counts[child_idx] = parent_ref_counts[child_idx].saturating_add(1);
        }
    }

    let mut nodes_without_parent_excluding_root = 0usize;
    let mut nodes_with_multiple_parents = 0usize;
    for (idx, &count) in parent_ref_counts.iter().enumerate() {
        if idx == 0 {
            continue;
        }
        if count == 0 {
            nodes_without_parent_excluding_root += 1;
        } else if count > 1 {
            nodes_with_multiple_parents += 1;
        }
    }

    let mut visited = vec![false; total_nodes];
    let mut stack = Vec::with_capacity(total_nodes.min(256));
    stack.push(0usize);
    while let Some(idx) = stack.pop() {
        if idx >= total_nodes || visited[idx] {
            continue;
        }
        visited[idx] = true;
        let node = &bvh_nodes[idx];
        if node.is_leaf == 0 {
            for child in [node.left_child, node.right_child] {
                if child != u32::MAX {
                    let child_idx = child as usize;
                    if child_idx < total_nodes {
                        stack.push(child_idx);
                    }
                }
            }
        }
    }

    let unreachable_internal_nodes = visited[..internal_nodes]
        .iter()
        .filter(|&&seen| !seen)
        .count();
    let unreachable_leaf_nodes = visited[internal_nodes..total_nodes]
        .iter()
        .filter(|&&seen| !seen)
        .count();
    let mut leaf_seen = vec![0u8; num_tetrahedrons];
    let mut leaf_invalid_tetra_indices = 0usize;
    let mut leaf_duplicate_tetra_indices = 0usize;
    for node in &bvh_nodes[internal_nodes..total_nodes] {
        let tet_idx = node.tetrahedron_index as usize;
        if tet_idx >= num_tetrahedrons {
            leaf_invalid_tetra_indices += 1;
            continue;
        }
        if leaf_seen[tet_idx] != 0 {
            leaf_duplicate_tetra_indices += 1;
        } else {
            leaf_seen[tet_idx] = 1;
        }
    }
    let leaf_missing_tetra_indices = leaf_seen.iter().filter(|&&count| count == 0).count();

    Some(BvhTopologySummary {
        total_nodes,
        internal_nodes,
        internal_ready,
        invalid_child_edges,
        self_child_edges,
        nodes_without_parent_excluding_root,
        nodes_with_multiple_parents,
        unreachable_internal_nodes,
        unreachable_leaf_nodes,
        leaf_invalid_tetra_indices,
        leaf_duplicate_tetra_indices,
        leaf_missing_tetra_indices,
    })
}

#[derive(Clone, Copy, Default)]
struct VteWorldBvhRayDiagExpectedRecord {
    slot: u32,
    pixel_x: u32,
    pixel_y: u32,
    layer: u32,
    hit: bool,
    hit_material: u32,
    hit_chunk: [i32; 4],
    hit_t: f32,
    miss_reason: u32,
    chunk_steps: u32,
    remaining_voxels: u32,
}

#[derive(Clone, Copy, Default)]
struct VteCpuRayTraceResult {
    hit: bool,
    hit_material: u32,
    hit_chunk: [i32; 4],
    hit_t: f32,
    miss_reason: u32,
    chunk_steps: u32,
    remaining_voxels: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VteCpuChunkPayload {
    Empty,
    Uniform(u32),
    Dense(u32),
}

fn vte_cpu_entry_chunk_coord(position: f32, direction: f32) -> i32 {
    let chunk_size = 8.0f32;
    let q = position / chunk_size;
    let q_floor = q.floor();
    let mut chunk = q_floor as i32;
    if direction < 0.0 {
        let boundary = q_floor * chunk_size;
        let eps = 1e-7 * chunk_size;
        if (position - boundary).abs() <= eps {
            chunk -= 1;
        }
    }
    chunk
}

fn vte_cpu_step_sign(d: f32) -> i32 {
    if d > 0.0 {
        1
    } else if d < 0.0 {
        -1
    } else {
        0
    }
}

fn vte_cpu_safe_inv_abs(d: f32) -> f32 {
    const EPS: f32 = 1e-6;
    if d.abs() > EPS {
        1.0 / d.abs()
    } else {
        1e30
    }
}

fn vte_cpu_intersect_aabb(
    ray_origin: [f32; 4],
    ray_dir: [f32; 4],
    bmin: [f32; 4],
    bmax: [f32; 4],
) -> Option<(f32, f32)> {
    const EPS: f32 = 1e-6;
    let mut t_min = 0.0f32;
    let mut t_max = 1e30f32;
    for axis in 0..4 {
        if ray_dir[axis].abs() > EPS {
            let t0 = (bmin[axis] - ray_origin[axis]) / ray_dir[axis];
            let t1 = (bmax[axis] - ray_origin[axis]) / ray_dir[axis];
            t_min = t_min.max(t0.min(t1));
            t_max = t_max.min(t0.max(t1));
        } else if ray_origin[axis] < bmin[axis] || ray_origin[axis] > bmax[axis] {
            return None;
        }
    }
    if t_max >= t_min {
        Some((t_min, t_max))
    } else {
        None
    }
}

fn vte_cpu_lookup_leaf_chunk_entry(
    leaf: &GpuVoxelLeafHeader,
    chunk_coord: [i32; 4],
    leaf_chunk_entries: &[u32],
    leaf_chunk_entry_count: u32,
) -> u32 {
    let local_x = chunk_coord[0] - leaf.min_chunk_coord[0];
    let local_y = chunk_coord[1] - leaf.min_chunk_coord[1];
    let local_z = chunk_coord[2] - leaf.min_chunk_coord[2];
    let local_w = chunk_coord[3] - leaf.min_chunk_coord[3];
    if local_x < 0 || local_y < 0 || local_z < 0 || local_w < 0 {
        return vte::VTE_LEAF_CHUNK_ENTRY_EMPTY;
    }
    let dim_x = (leaf.max_chunk_coord[0] - leaf.min_chunk_coord[0] + 1).max(0) as usize;
    let dim_y = (leaf.max_chunk_coord[1] - leaf.min_chunk_coord[1] + 1).max(0) as usize;
    let dim_z = (leaf.max_chunk_coord[2] - leaf.min_chunk_coord[2] + 1).max(0) as usize;
    let dim_w = (leaf.max_chunk_coord[3] - leaf.min_chunk_coord[3] + 1).max(0) as usize;
    if local_x as usize >= dim_x
        || local_y as usize >= dim_y
        || local_z as usize >= dim_z
        || local_w as usize >= dim_w
    {
        return vte::VTE_LEAF_CHUNK_ENTRY_EMPTY;
    }
    let linear = local_x as usize
        + dim_x * (local_y as usize + dim_y * (local_z as usize + dim_z * local_w as usize));
    let entry_index = leaf.chunk_entry_offset as usize + linear;
    if entry_index >= leaf_chunk_entry_count as usize || entry_index >= leaf_chunk_entries.len() {
        return vte::VTE_LEAF_CHUNK_ENTRY_EMPTY;
    }
    leaf_chunk_entries[entry_index]
}

fn vte_cpu_lookup_chunk_payload_linear(
    chunk_coord: [i32; 4],
    leaf_headers: &[GpuVoxelLeafHeader],
    leaf_count: u32,
    leaf_chunk_entries: &[u32],
    leaf_chunk_entry_count: u32,
    chunk_count: u32,
) -> VteCpuChunkPayload {
    for leaf in leaf_headers.iter().take(leaf_count as usize) {
        if chunk_coord[0] < leaf.min_chunk_coord[0]
            || chunk_coord[0] > leaf.max_chunk_coord[0]
            || chunk_coord[1] < leaf.min_chunk_coord[1]
            || chunk_coord[1] > leaf.max_chunk_coord[1]
            || chunk_coord[2] < leaf.min_chunk_coord[2]
            || chunk_coord[2] > leaf.max_chunk_coord[2]
            || chunk_coord[3] < leaf.min_chunk_coord[3]
            || chunk_coord[3] > leaf.max_chunk_coord[3]
        {
            continue;
        }
        if leaf.leaf_kind == vte::VTE_LEAF_KIND_UNIFORM {
            return if leaf.uniform_material != 0 {
                VteCpuChunkPayload::Uniform(leaf.uniform_material)
            } else {
                VteCpuChunkPayload::Empty
            };
        }
        if leaf.leaf_kind != vte::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY {
            continue;
        }
        let encoded = vte_cpu_lookup_leaf_chunk_entry(
            leaf,
            chunk_coord,
            leaf_chunk_entries,
            leaf_chunk_entry_count,
        );
        if (encoded & vte::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG) != 0 {
            let material = encoded & (!vte::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG);
            return if material != 0 {
                VteCpuChunkPayload::Uniform(material)
            } else {
                VteCpuChunkPayload::Empty
            };
        }
        if encoded == vte::VTE_LEAF_CHUNK_ENTRY_EMPTY {
            return VteCpuChunkPayload::Empty;
        }
        let chunk_index = encoded.saturating_sub(1);
        if chunk_index < chunk_count {
            return VteCpuChunkPayload::Dense(chunk_index);
        }
        return VteCpuChunkPayload::Empty;
    }
    VteCpuChunkPayload::Empty
}

fn vte_cpu_sample_voxel_in_chunk(
    voxel_coord: [i32; 4],
    chunk_min: [i32; 4],
    header: &GpuVoxelChunkHeader,
    occupancy_words: &[u32],
    occupancy_word_count: u32,
    material_words: &[u32],
    material_word_count: u32,
) -> Option<u32> {
    let local = [
        voxel_coord[0] - chunk_min[0],
        voxel_coord[1] - chunk_min[1],
        voxel_coord[2] - chunk_min[2],
        voxel_coord[3] - chunk_min[3],
    ];
    let local_idx =
        (local[3] as u32) * 512 + (local[2] as u32) * 64 + (local[1] as u32) * 8 + local[0] as u32;
    if (header.flags & GpuVoxelChunkHeader::FLAG_FULL) == 0 {
        let occupancy_idx = header.occupancy_word_offset + (local_idx >> 5);
        let occupancy_idx = occupancy_idx as usize;
        if occupancy_idx >= occupancy_word_count as usize || occupancy_idx >= occupancy_words.len()
        {
            return None;
        }
        let occupancy_word = occupancy_words[occupancy_idx];
        if (occupancy_word & (1u32 << (local_idx & 31))) == 0 {
            return None;
        }
    }
    let material_idx = header.material_word_offset + (local_idx >> 1);
    let material_idx = material_idx as usize;
    if material_idx >= material_word_count as usize || material_idx >= material_words.len() {
        return None;
    }
    let material_word = material_words[material_idx];
    let material = (material_word >> ((local_idx & 1) * 16)) & 0xFFFF;
    if material != 0 {
        Some(material)
    } else {
        None
    }
}

fn vte_cpu_macro_cell_may_contain_solid(
    voxel_coord: [i32; 4],
    chunk_min: [i32; 4],
    header: &GpuVoxelChunkHeader,
    macro_words: &[u32],
    macro_word_count: u32,
) -> bool {
    if macro_word_count == 0 {
        return true;
    }
    let local = [
        voxel_coord[0] - chunk_min[0],
        voxel_coord[1] - chunk_min[1],
        voxel_coord[2] - chunk_min[2],
        voxel_coord[3] - chunk_min[3],
    ];
    let mx = (local[0] >> 1) as u32;
    let my = (local[1] >> 1) as u32;
    let mz = (local[2] >> 1) as u32;
    let mw = (local[3] >> 1) as u32;
    let macro_idx = mw * 64 + mz * 16 + my * 4 + mx;
    let macro_word_idx = header.macro_word_offset + (macro_idx >> 5);
    let macro_word_idx = macro_word_idx as usize;
    if macro_word_idx >= macro_word_count as usize || macro_word_idx >= macro_words.len() {
        return true;
    }
    (macro_words[macro_word_idx] & (1u32 << (macro_idx & 31))) != 0
}

fn vte_cpu_advance_dda4(
    coord: &mut [i32; 4],
    step: [i32; 4],
    t_max: &mut [f32; 4],
    t_delta: [f32; 4],
) -> f32 {
    let mut axis = 0usize;
    let mut best = t_max[0];
    for (i, value) in t_max.iter().enumerate().skip(1) {
        if *value < best {
            axis = i;
            best = *value;
        }
    }
    t_max[axis] += t_delta[axis];
    coord[axis] += step[axis];
    best
}

fn vte_cpu_trace_voxels_in_chunk(
    ray_origin: [f32; 4],
    ray_dir: [f32; 4],
    mut t_enter: f32,
    mut t_exit: f32,
    chunk_coord: [i32; 4],
    header: &GpuVoxelChunkHeader,
    occupancy_words: &[u32],
    occupancy_word_count: u32,
    material_words: &[u32],
    material_word_count: u32,
    macro_words: &[u32],
    macro_word_count: u32,
    remaining_voxels: &mut u32,
) -> Option<(u32, f32)> {
    const EPS: f32 = 1e-6;
    if *remaining_voxels == 0 {
        return None;
    }

    let chunk_min = [
        chunk_coord[0] * 8,
        chunk_coord[1] * 8,
        chunk_coord[2] * 8,
        chunk_coord[3] * 8,
    ];
    let chunk_max = [
        chunk_min[0] + 8,
        chunk_min[1] + 8,
        chunk_min[2] + 8,
        chunk_min[3] + 8,
    ];

    let solid_bmin = [
        (chunk_min[0] + header.solid_local_min[0]) as f32,
        (chunk_min[1] + header.solid_local_min[1]) as f32,
        (chunk_min[2] + header.solid_local_min[2]) as f32,
        (chunk_min[3] + header.solid_local_min[3]) as f32,
    ];
    let solid_bmax = [
        (chunk_min[0] + header.solid_local_max[0] + 1) as f32,
        (chunk_min[1] + header.solid_local_max[1] + 1) as f32,
        (chunk_min[2] + header.solid_local_max[2] + 1) as f32,
        (chunk_min[3] + header.solid_local_max[3] + 1) as f32,
    ];
    let (solid_enter, solid_exit) =
        vte_cpu_intersect_aabb(ray_origin, ray_dir, solid_bmin, solid_bmax)?;
    t_enter = t_enter.max(solid_enter);
    t_exit = t_exit.min(solid_exit);
    let mut interval_eps = EPS.max(t_enter.abs().max(t_exit.abs()) * 1e-7);
    if t_exit <= t_enter + interval_eps {
        return None;
    }

    let t_start = t_enter.max(0.0);
    let start_bias = interval_eps.max(t_start.abs() * 1e-7);
    let t_trace_start = t_start + start_bias;
    if t_trace_start >= t_exit {
        return None;
    }
    let local_t_exit = t_exit - t_trace_start;
    if local_t_exit <= interval_eps {
        return None;
    }

    let start_pos = [
        ray_origin[0] + ray_dir[0] * t_trace_start,
        ray_origin[1] + ray_dir[1] * t_trace_start,
        ray_origin[2] + ray_dir[2] * t_trace_start,
        ray_origin[3] + ray_dir[3] * t_trace_start,
    ];
    let mut voxel_coord = [
        start_pos[0].floor() as i32,
        start_pos[1].floor() as i32,
        start_pos[2].floor() as i32,
        start_pos[3].floor() as i32,
    ];
    for axis in 0..4 {
        if voxel_coord[axis] < chunk_min[axis] {
            voxel_coord[axis] = chunk_min[axis];
        }
        if voxel_coord[axis] >= chunk_max[axis] {
            voxel_coord[axis] = chunk_max[axis] - 1;
        }
    }

    let step = [
        vte_cpu_step_sign(ray_dir[0]),
        vte_cpu_step_sign(ray_dir[1]),
        vte_cpu_step_sign(ray_dir[2]),
        vte_cpu_step_sign(ray_dir[3]),
    ];
    let t_delta = [
        vte_cpu_safe_inv_abs(ray_dir[0]),
        vte_cpu_safe_inv_abs(ray_dir[1]),
        vte_cpu_safe_inv_abs(ray_dir[2]),
        vte_cpu_safe_inv_abs(ray_dir[3]),
    ];
    let mut t_max_local = [1e30f32; 4];
    if step[0] > 0 {
        t_max_local[0] = (voxel_coord[0] as f32 + 1.0 - start_pos[0]) * t_delta[0];
    } else if step[0] < 0 {
        t_max_local[0] = (start_pos[0] - voxel_coord[0] as f32) * t_delta[0];
    }
    if step[1] > 0 {
        t_max_local[1] = (voxel_coord[1] as f32 + 1.0 - start_pos[1]) * t_delta[1];
    } else if step[1] < 0 {
        t_max_local[1] = (start_pos[1] - voxel_coord[1] as f32) * t_delta[1];
    }
    if step[2] > 0 {
        t_max_local[2] = (voxel_coord[2] as f32 + 1.0 - start_pos[2]) * t_delta[2];
    } else if step[2] < 0 {
        t_max_local[2] = (start_pos[2] - voxel_coord[2] as f32) * t_delta[2];
    }
    if step[3] > 0 {
        t_max_local[3] = (voxel_coord[3] as f32 + 1.0 - start_pos[3]) * t_delta[3];
    } else if step[3] < 0 {
        t_max_local[3] = (start_pos[3] - voxel_coord[3] as f32) * t_delta[3];
    }

    let can_macro_skip =
        (header.flags & GpuVoxelChunkHeader::FLAG_FULL) == 0 && macro_word_count > 0;
    let mut cached_macro_coord = [i32::MIN; 4];
    let mut cached_macro_has_solid = true;
    let mut local_t = 0.0f32;

    loop {
        if *remaining_voxels == 0 {
            return None;
        }
        if local_t > local_t_exit + interval_eps {
            return None;
        }

        let next_local_t = t_max_local.iter().copied().fold(f32::INFINITY, f32::min);
        let segment_end = next_local_t.min(local_t_exit);
        let has_interior = (segment_end - local_t) > interval_eps;
        if has_interior {
            *remaining_voxels = remaining_voxels.saturating_sub(1);
            let mut macro_has_solid = true;
            if can_macro_skip {
                let macro_coord = [
                    (voxel_coord[0] - chunk_min[0]) >> 1,
                    (voxel_coord[1] - chunk_min[1]) >> 1,
                    (voxel_coord[2] - chunk_min[2]) >> 1,
                    (voxel_coord[3] - chunk_min[3]) >> 1,
                ];
                if macro_coord != cached_macro_coord {
                    cached_macro_coord = macro_coord;
                    cached_macro_has_solid = vte_cpu_macro_cell_may_contain_solid(
                        voxel_coord,
                        chunk_min,
                        header,
                        macro_words,
                        macro_word_count,
                    );
                }
                macro_has_solid = cached_macro_has_solid;
            }
            if macro_has_solid {
                if let Some(material) = vte_cpu_sample_voxel_in_chunk(
                    voxel_coord,
                    chunk_min,
                    header,
                    occupancy_words,
                    occupancy_word_count,
                    material_words,
                    material_word_count,
                ) {
                    return Some((material, t_trace_start + local_t));
                }
            }
        }

        let _ = vte_cpu_advance_dda4(&mut voxel_coord, step, &mut t_max_local, t_delta);
        local_t = local_t.max(next_local_t);
        for axis in 0..4 {
            if voxel_coord[axis] < chunk_min[axis] || voxel_coord[axis] >= chunk_max[axis] {
                return None;
            }
        }
        interval_eps = EPS.max(t_enter.abs().max(t_exit.abs()) * 1e-7);
    }
}

// NOTE: This CPU reference trace assumes scale_exp=0 throughout (hardcoded
// CHUNK_SIZE=8 for chunk DDA and voxel size=1.0). It will produce incorrect
// results for non-zero-scale leaves, causing false diagnostic mismatches when
// `vte_reference_compare` is enabled on multi-scale worlds.
fn vte_cpu_trace_ray_linear(
    ray_origin: [f32; 4],
    ray_dir: [f32; 4],
    meta: &vte::GpuVoxelFrameMeta,
    region_bvh_nodes: &[GpuVoxelChunkBvhNode],
    leaf_headers: &[GpuVoxelLeafHeader],
    leaf_chunk_entries: &[u32],
    chunk_headers: &[GpuVoxelChunkHeader],
    occupancy_words: &[u32],
    material_words: &[u32],
    macro_words: &[u32],
) -> VteCpuRayTraceResult {
    const CHUNK_EPS: f32 = 1e-6;
    if meta.leaf_count == 0 || meta.region_bvh_node_count == 0 {
        return VteCpuRayTraceResult::default();
    }
    if meta.region_bvh_root_index >= meta.region_bvh_node_count {
        return VteCpuRayTraceResult::default();
    }
    let root_index = meta.region_bvh_root_index as usize;
    let Some(root_node) = region_bvh_nodes.get(root_index) else {
        return VteCpuRayTraceResult::default();
    };
    let root_bmin = root_node.world_min;
    let root_bmax = root_node.world_max;
    let Some((root_enter, root_exit)) =
        vte_cpu_intersect_aabb(ray_origin, ray_dir, root_bmin, root_bmax)
    else {
        return VteCpuRayTraceResult::default();
    };
    let traversal_min_t = root_enter.max(0.0);
    let max_distance = meta.max_trace_distance.max(1.0);
    let traversal_max_t = root_exit.min(max_distance);
    if traversal_max_t <= traversal_min_t + CHUNK_EPS {
        return VteCpuRayTraceResult::default();
    }

    let clipped_by_max_distance = root_exit > max_distance + CHUNK_EPS;
    let mut touched_visible_chunk = false;
    let max_chunk_steps = meta.max_trace_steps.clamp(1, 4096);
    let mut remaining_voxels = meta.max_trace_steps.saturating_mul(8).clamp(1, 32768);
    let mut chunk_steps = 0u32;
    let mut current_t = traversal_min_t;
    let mut chunk_coord = [0i32; 4];
    let mut chunk_state_valid = false;
    let chunk_step = [
        vte_cpu_step_sign(ray_dir[0]),
        vte_cpu_step_sign(ray_dir[1]),
        vte_cpu_step_sign(ray_dir[2]),
        vte_cpu_step_sign(ray_dir[3]),
    ];
    let t_delta_chunk = [
        vte_cpu_safe_inv_abs(ray_dir[0]) * 8.0,
        vte_cpu_safe_inv_abs(ray_dir[1]) * 8.0,
        vte_cpu_safe_inv_abs(ray_dir[2]) * 8.0,
        vte_cpu_safe_inv_abs(ray_dir[3]) * 8.0,
    ];
    let mut t_max_chunk = [1e30f32; 4];
    let mut last_chunk = [0i32; 4];

    while remaining_voxels > 0 && chunk_steps < max_chunk_steps {
        chunk_steps += 1;
        if current_t > traversal_max_t {
            break;
        }
        if !chunk_state_valid {
            let probe_eps = CHUNK_EPS.max(current_t.abs() * 1e-7);
            let probe_t = traversal_max_t.min(current_t + probe_eps);
            let probe_pos = [
                ray_origin[0] + ray_dir[0] * probe_t,
                ray_origin[1] + ray_dir[1] * probe_t,
                ray_origin[2] + ray_dir[2] * probe_t,
                ray_origin[3] + ray_dir[3] * probe_t,
            ];
            chunk_coord = [
                vte_cpu_entry_chunk_coord(probe_pos[0], ray_dir[0]),
                vte_cpu_entry_chunk_coord(probe_pos[1], ray_dir[1]),
                vte_cpu_entry_chunk_coord(probe_pos[2], ray_dir[2]),
                vte_cpu_entry_chunk_coord(probe_pos[3], ray_dir[3]),
            ];
            t_max_chunk = [1e30; 4];
            if chunk_step[0] > 0 {
                t_max_chunk[0] = (((chunk_coord[0] + 1) * 8) as f32 - ray_origin[0]) / ray_dir[0];
            } else if chunk_step[0] < 0 {
                t_max_chunk[0] = ((chunk_coord[0] * 8) as f32 - ray_origin[0]) / ray_dir[0];
            }
            if chunk_step[1] > 0 {
                t_max_chunk[1] = (((chunk_coord[1] + 1) * 8) as f32 - ray_origin[1]) / ray_dir[1];
            } else if chunk_step[1] < 0 {
                t_max_chunk[1] = ((chunk_coord[1] * 8) as f32 - ray_origin[1]) / ray_dir[1];
            }
            if chunk_step[2] > 0 {
                t_max_chunk[2] = (((chunk_coord[2] + 1) * 8) as f32 - ray_origin[2]) / ray_dir[2];
            } else if chunk_step[2] < 0 {
                t_max_chunk[2] = ((chunk_coord[2] * 8) as f32 - ray_origin[2]) / ray_dir[2];
            }
            if chunk_step[3] > 0 {
                t_max_chunk[3] = (((chunk_coord[3] + 1) * 8) as f32 - ray_origin[3]) / ray_dir[3];
            } else if chunk_step[3] < 0 {
                t_max_chunk[3] = ((chunk_coord[3] * 8) as f32 - ray_origin[3]) / ray_dir[3];
            }
            chunk_state_valid = true;
        }
        last_chunk = chunk_coord;
        let mut chunk_exit_t =
            traversal_max_t.min(t_max_chunk.iter().copied().fold(f32::INFINITY, f32::min));
        chunk_exit_t = chunk_exit_t.max(current_t);

        let payload = vte_cpu_lookup_chunk_payload_linear(
            chunk_coord,
            leaf_headers,
            meta.leaf_count,
            leaf_chunk_entries,
            meta.leaf_chunk_entry_count,
            meta.chunk_count,
        );
        if payload != VteCpuChunkPayload::Empty {
            touched_visible_chunk = true;
        }
        match payload {
            VteCpuChunkPayload::Uniform(material) => {
                if material != 0 {
                    let hit_t = current_t.max(0.0);
                    let probe_eps = CHUNK_EPS.max(hit_t.abs() * 1e-7);
                    let probe_t = chunk_exit_t.min(hit_t + probe_eps);
                    let probe_pos = [
                        ray_origin[0] + ray_dir[0] * probe_t,
                        ray_origin[1] + ray_dir[1] * probe_t,
                        ray_origin[2] + ray_dir[2] * probe_t,
                        ray_origin[3] + ray_dir[3] * probe_t,
                    ];
                    let hit_chunk = [
                        vte_cpu_entry_chunk_coord(probe_pos[0], ray_dir[0]),
                        vte_cpu_entry_chunk_coord(probe_pos[1], ray_dir[1]),
                        vte_cpu_entry_chunk_coord(probe_pos[2], ray_dir[2]),
                        vte_cpu_entry_chunk_coord(probe_pos[3], ray_dir[3]),
                    ];
                    return VteCpuRayTraceResult {
                        hit: true,
                        hit_material: material,
                        hit_chunk,
                        hit_t,
                        miss_reason: VTE_CPU_MISS_REASON_NONE,
                        chunk_steps,
                        remaining_voxels,
                    };
                }
            }
            VteCpuChunkPayload::Dense(chunk_index) => {
                if let Some(header) = chunk_headers.get(chunk_index as usize) {
                    if chunk_exit_t > current_t + CHUNK_EPS {
                        if let Some((material, hit_t)) = vte_cpu_trace_voxels_in_chunk(
                            ray_origin,
                            ray_dir,
                            current_t,
                            chunk_exit_t,
                            chunk_coord,
                            header,
                            occupancy_words,
                            meta.occupancy_word_count,
                            material_words,
                            meta.material_word_count,
                            macro_words,
                            meta.macro_word_count,
                            &mut remaining_voxels,
                        ) {
                            return VteCpuRayTraceResult {
                                hit: true,
                                hit_material: material,
                                hit_chunk: chunk_coord,
                                hit_t,
                                miss_reason: VTE_CPU_MISS_REASON_NONE,
                                chunk_steps,
                                remaining_voxels,
                            };
                        }
                    }
                }
            }
            VteCpuChunkPayload::Empty => {}
        }

        let next_chunk_t = vte_cpu_advance_dda4(
            &mut chunk_coord,
            chunk_step,
            &mut t_max_chunk,
            t_delta_chunk,
        );
        let advance_eps = CHUNK_EPS.max(next_chunk_t.abs() * 1e-7);
        current_t = (current_t + CHUNK_EPS).max(next_chunk_t + advance_eps);
    }

    let miss_reason = if remaining_voxels == 0 {
        VTE_CPU_MISS_REASON_VOXEL_BUDGET
    } else if chunk_steps >= max_chunk_steps {
        VTE_CPU_MISS_REASON_CHUNK_BUDGET
    } else if clipped_by_max_distance {
        VTE_CPU_MISS_REASON_MAX_DISTANCE
    } else if touched_visible_chunk {
        VTE_CPU_MISS_REASON_TOUCHED_VISIBLE_CHUNK
    } else {
        VTE_CPU_MISS_REASON_NONE
    };
    VteCpuRayTraceResult {
        hit: false,
        hit_material: 0,
        hit_chunk: last_chunk,
        hit_t: -1.0,
        miss_reason,
        chunk_steps,
        remaining_voxels,
    }
}

fn vte_world_ray_direction_for_pixel_layer(
    pixel_x: u32,
    pixel_y: u32,
    layer: u32,
    render_dimensions: [u32; 3],
    present_dimensions: [u32; 2],
    focal_length_xy: f32,
    focal_length_zw: f32,
    world_dir_x: [f32; 4],
    world_dir_y: [f32; 4],
    world_dir_z: [f32; 4],
    world_dir_w: [f32; 4],
) -> [f32; 4] {
    let width = render_dimensions[0].max(1);
    let height = render_dimensions[1].max(1);
    let layer_count = render_dimensions[2].max(1);
    let pixel_pos_x = pixel_x as f32 / width as f32 * 2.0 - 1.0;
    let pixel_pos_y = pixel_y as f32 / height as f32 * 2.0 - 1.0;
    let aspect_ratio =
        (present_dimensions[0].max(1)) as f32 / (present_dimensions[1].max(1)) as f32;
    let sx = pixel_pos_x / focal_length_xy;
    let sy = (-pixel_pos_y / aspect_ratio) / focal_length_xy;
    let view_angle = (std::f32::consts::PI / 2.0) / focal_length_zw;
    let z_norm = ((layer as f32 + 0.5) / layer_count as f32) * 2.0 - 1.0;
    let zw_angle = z_norm * (view_angle * 0.5) + (std::f32::consts::PI * 0.25);
    let mut dir = [0.0f32; 4];
    for axis in 0..4 {
        dir[axis] = world_dir_x[axis] * sx
            + world_dir_y[axis] * sy
            + world_dir_z[axis] * zw_angle.cos()
            + world_dir_w[axis] * zw_angle.sin();
    }
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] + dir[3] * dir[3]).sqrt();
    if len > 1e-8 {
        for c in &mut dir {
            *c /= len;
        }
    }
    dir
}

#[allow(clippy::too_many_arguments)]
fn build_world_bvh_ray_diag_expected_records(
    meta: &vte::GpuVoxelFrameMeta,
    render_dimensions: [u32; 3],
    present_dimensions: [u32; 2],
    focal_length_xy: f32,
    focal_length_zw: f32,
    world_origin: [f32; 4],
    world_dir_x: [f32; 4],
    world_dir_y: [f32; 4],
    world_dir_z: [f32; 4],
    world_dir_w: [f32; 4],
    chunk_headers: &[GpuVoxelChunkHeader],
    occupancy_words: &[u32],
    material_words: &[u32],
    leaf_headers: &[GpuVoxelLeafHeader],
    region_bvh_nodes: &[GpuVoxelChunkBvhNode],
    leaf_chunk_entries: &[u32],
    macro_words: &[u32],
) -> Vec<VteWorldBvhRayDiagExpectedRecord> {
    let requested = usize::min(
        meta.world_bvh_diag_sample_count as usize,
        vte::VTE_WORLD_BVH_RAY_DIAG_CAPACITY,
    );
    if requested == 0 {
        return Vec::new();
    }
    let width = render_dimensions[0].max(1);
    let height = render_dimensions[1].max(1);
    let dispatch_layers = if meta.stage_b_mode == VteDisplayMode::Integral.as_u32()
        && render_dimensions[2].max(1) > 1
    {
        1
    } else {
        render_dimensions[2].max(1)
    };
    let plane = width as usize * height as usize;
    let total_dispatch_threads = plane * dispatch_layers as usize;
    if total_dispatch_threads == 0 {
        return Vec::new();
    }
    let stride = usize::max(1, total_dispatch_threads / requested);
    let offset = (meta.world_bvh_diag_seed as usize) % stride;
    let center_layer = meta
        .stage_b_slice_layer
        .min(render_dimensions[2].max(1).saturating_sub(1));

    let mut out = Vec::with_capacity(requested);
    for linear in 0..total_dispatch_threads {
        if linear % stride != offset {
            continue;
        }
        let slot = linear / stride;
        if slot >= vte::VTE_WORLD_BVH_RAY_DIAG_CAPACITY {
            continue;
        }
        let dispatch_z = linear / plane;
        let in_plane = linear % plane;
        let pixel_y = (in_plane / width as usize) as u32;
        let pixel_x = (in_plane % width as usize) as u32;
        let layer = if dispatch_layers == 1 && render_dimensions[2].max(1) > 1 {
            center_layer
        } else {
            dispatch_z as u32
        };
        let ray_dir = vte_world_ray_direction_for_pixel_layer(
            pixel_x,
            pixel_y,
            layer,
            render_dimensions,
            present_dimensions,
            focal_length_xy,
            focal_length_zw,
            world_dir_x,
            world_dir_y,
            world_dir_z,
            world_dir_w,
        );
        let traced = vte_cpu_trace_ray_linear(
            world_origin,
            ray_dir,
            meta,
            region_bvh_nodes,
            leaf_headers,
            leaf_chunk_entries,
            chunk_headers,
            occupancy_words,
            material_words,
            macro_words,
        );
        out.push(VteWorldBvhRayDiagExpectedRecord {
            slot: slot as u32,
            pixel_x,
            pixel_y,
            layer,
            hit: traced.hit,
            hit_material: traced.hit_material,
            hit_chunk: traced.hit_chunk,
            hit_t: traced.hit_t,
            miss_reason: traced.miss_reason,
            chunk_steps: traced.chunk_steps,
            remaining_voxels: traced.remaining_voxels,
        });
    }
    out
}

pub struct RenderContext {
    pub window: Option<Arc<Window>>,
    swapchain: Option<Arc<Swapchain>>,
    render_pass: Option<Arc<RenderPass>>,
    framebuffers: Option<Vec<Arc<Framebuffer>>>,
    present_pipeline: Option<PresentPipelineContext>,
    compute_pipeline: ComputePipelineContext,
    viewport: Viewport,
    recreate_swapchain: bool,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    one_time_buffers: OneTimeBuffers,
    sized_buffers: SizedBuffers,
    frames_in_flight: Vec<FrameInFlight>,
    cpu_screen_capture_buffer: Subbuffer<[u8]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    frames_rendered: usize,
    bvh_scene_hash: u64,
    vte_non_voxel_scene_hash: u64,
    vte_non_voxel_bvh_topology_tet_count: usize,
    vte_non_voxel_bvh_refit_frames: usize,
    last_clipped_tet_count: u32,
    profiler: GpuProfiler,
    hud_font: Option<FontArc>,
    hud_resources: Option<HudResources>,
    egui_resources: Option<EguiResources>,
    material_icons_view: Option<Arc<ImageView>>,
    material_icons_sampler: Option<Arc<Sampler>>,
    aetna_overlay: Option<AetnaOverlay>,
    hud_breadcrumbs: VecDeque<[f32; 4]>,
    hud_previous_camera: Option<[f32; 4]>,
    hud_previous_sample_time: Option<Instant>,
    hud_w_velocity: f32,
    frame_time_ms: f32,
    last_render_start: Option<Instant>,
    stall_trace: bool,
    last_backend: RenderBackend,
    vte_debug_counters: VteDebugCounters,
    vte_compare_stats: vte::VteCompareStats,
    vte_first_mismatch: vte::VteFirstMismatch,
    vte_backend_notice_printed: bool,
    vte_entity_diag_enabled: bool,
    vte_entity_diag_verbose: bool,
    vte_entity_diag_bvh_readback: bool,
    vte_entity_diag_bvh_topology: bool,
    vte_entity_diag_interval: usize,
    vte_entity_diag_last_log_frame: Option<usize>,
    vte_stage_a_breakdown_enabled: bool,
    vte_stage_a_breakdown_interval: usize,
    vte_stage_a_breakdown_last_log_frame: Option<usize>,
    vte_world_bvh_ray_diag_enabled: bool,
    vte_world_bvh_ray_diag_samples: usize,
    vte_world_bvh_ray_diag_interval: usize,
    vte_world_bvh_ray_diag_last_log_frame: Option<usize>,
    vte_entity_diag_prev_used_non_voxel: Option<usize>,
    vte_entity_diag_prev_tets_non_voxel: Option<usize>,
    drop_next_profile_sample: bool,
    texture_pool: TexturePool,
}

impl RenderContext {
    fn reset_vte_compare_buffers(&mut self, frame_idx: usize) {
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .vte_compare_stats_buffer
                .write()
                .unwrap();
            writer.fill(0u32);
        }
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .vte_first_mismatch_buffer
                .write()
                .unwrap();
            writer.fill(0u32);
        }
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .vte_world_bvh_ray_diag_buffer
                .write()
                .unwrap();
            writer.fill(0u32);
        }
    }

    fn clear_vte_compare_diagnostics(&mut self) {
        self.vte_compare_stats = vte::VteCompareStats::default();
        self.vte_first_mismatch = vte::VteFirstMismatch::default();
    }

    fn refresh_vte_compare_diagnostics(&mut self, frame_idx: usize) {
        let stats_words = self.frames_in_flight[frame_idx]
            .live_buffers
            .vte_compare_stats_buffer
            .read()
            .unwrap();
        if stats_words.len() >= vte::VTE_COMPARE_STATS_WORD_COUNT {
            self.vte_compare_stats = vte::VteCompareStats {
                compared: stats_words[vte::VTE_COMPARE_STAT_COMPARED],
                matches: stats_words[vte::VTE_COMPARE_STAT_MATCHES],
                mismatches: stats_words[vte::VTE_COMPARE_STAT_MISMATCHES],
                hit_state_mismatches: stats_words[vte::VTE_COMPARE_STAT_HIT_STATE_MISMATCHES],
                chunk_material_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_CHUNK_MATERIAL_MISMATCHES],
                fast_miss_ref_hit: stats_words[vte::VTE_COMPARE_STAT_FAST_MISS_REF_HIT],
                fast_hit_ref_miss: stats_words[vte::VTE_COMPARE_STAT_FAST_HIT_REF_MISS],
                miss_reason_counts: [
                    stats_words[vte::VTE_COMPARE_STAT_REASON_NONE],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_TOUCHED_VISIBLE],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_VOXEL_BUDGET],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_CHUNK_BUDGET],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_MAX_DISTANCE],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_LOOKUP_FALSE_NEGATIVE],
                ],
                zero_interval_flags: stats_words[vte::VTE_COMPARE_STAT_ZERO_INTERVAL_FLAG],
                tie_stepped_flags: stats_words[vte::VTE_COMPARE_STAT_TIE_STEPPED_FLAG],
                lookup_fallback_flags: stats_words[vte::VTE_COMPARE_STAT_LOOKUP_FALLBACK_FLAG],
                entity_bvh_samples: stats_words[vte::VTE_COMPARE_STAT_ENTITY_BVH_SAMPLE],
                entity_bvh_mismatches: stats_words[vte::VTE_COMPARE_STAT_ENTITY_BVH_MISMATCH],
                entity_bvh_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_HIT_STATE_MISMATCH],
                entity_bvh_material_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_MATERIAL_MISMATCH],
                entity_bvh_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_DISTANCE_MISMATCH],
                entity_bvh_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_TETRA_MISMATCH],
                entity_bvh_miss_linear_hit: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_MISS_LINEAR_HIT],
                entity_bvh_hit_linear_miss: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_HIT_LINEAR_MISS],
                entity_bvh_noprune_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_MISMATCH],
                entity_bvh_noprune_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_HIT_STATE_MISMATCH],
                entity_bvh_noprune_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_DISTANCE_MISMATCH],
                entity_bvh_noprune_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_TETRA_MISMATCH],
                entity_bvh_noaabb_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_MISMATCH],
                entity_bvh_noaabb_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_HIT_STATE_MISMATCH],
                entity_bvh_noaabb_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_DISTANCE_MISMATCH],
                entity_bvh_noaabb_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_TETRA_MISMATCH],
                entity_linear_order_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_MISMATCH],
                entity_linear_order_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_HIT_STATE_MISMATCH],
                entity_linear_order_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_DISTANCE_MISMATCH],
                entity_linear_order_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_TETRA_MISMATCH],
                entity_bvh_leafarray_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_MISMATCH],
                entity_bvh_leafarray_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_HIT_STATE_MISMATCH],
                entity_bvh_leafarray_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_DISTANCE_MISMATCH],
                entity_bvh_leafarray_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_TETRA_MISMATCH],
                stagea_samples: stats_words[vte::VTE_COMPARE_STAT_STAGEA_SAMPLES],
                stagea_entity_queries: stats_words[vte::VTE_COMPARE_STAT_STAGEA_ENTITY_QUERIES],
                stagea_entity_hits: stats_words[vte::VTE_COMPARE_STAT_STAGEA_ENTITY_HITS],
                stagea_voxel_hits: stats_words[vte::VTE_COMPARE_STAT_STAGEA_VOXEL_HITS],
                stagea_sky_misses: stats_words[vte::VTE_COMPARE_STAT_STAGEA_SKY_MISSES],
                stagea_chunk_steps_sum: stats_words[vte::VTE_COMPARE_STAT_STAGEA_CHUNK_STEPS_SUM],
                stagea_voxel_steps_sum: stats_words[vte::VTE_COMPARE_STAT_STAGEA_VOXEL_STEPS_SUM],
                stagea_node_visits_sum: stats_words[vte::VTE_COMPARE_STAT_STAGEA_NODE_VISITS_SUM],
            };
        } else {
            self.vte_compare_stats = vte::VteCompareStats::default();
        }

        let first_words = self.frames_in_flight[frame_idx]
            .live_buffers
            .vte_first_mismatch_buffer
            .read()
            .unwrap();
        if first_words.len() >= vte::VTE_FIRST_MISMATCH_WORD_COUNT
            && first_words[vte::VTE_FIRST_MISMATCH_VALID] != 0
        {
            let hit_mask = first_words[vte::VTE_FIRST_MISMATCH_HIT_MASK];
            self.vte_first_mismatch = vte::VteFirstMismatch {
                valid: true,
                pixel_x: first_words[vte::VTE_FIRST_MISMATCH_PIXEL_X],
                pixel_y: first_words[vte::VTE_FIRST_MISMATCH_PIXEL_Y],
                layer: first_words[vte::VTE_FIRST_MISMATCH_LAYER],
                mismatch_kind: first_words[vte::VTE_FIRST_MISMATCH_KIND],
                miss_reason: first_words[vte::VTE_FIRST_MISMATCH_MISS_REASON],
                debug_flags: first_words[vte::VTE_FIRST_MISMATCH_DEBUG_FLAGS],
                fast_hit: (hit_mask & 0x1) != 0,
                ref_hit: (hit_mask & 0x2) != 0,
                fast_chunk: [
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_X] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_Y] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_Z] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_W] as i32,
                ],
                ref_chunk: [
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_X] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_Y] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_Z] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_W] as i32,
                ],
                fast_material: first_words[vte::VTE_FIRST_MISMATCH_FAST_MATERIAL],
                ref_material: first_words[vte::VTE_FIRST_MISMATCH_REF_MATERIAL],
                fast_hit_t: f32::from_bits(first_words[vte::VTE_FIRST_MISMATCH_FAST_HIT_T]),
                ref_hit_t: f32::from_bits(first_words[vte::VTE_FIRST_MISMATCH_REF_HIT_T]),
                chunk_steps_taken: first_words[vte::VTE_FIRST_MISMATCH_CHUNK_STEPS],
                remaining_voxel_steps: first_words[vte::VTE_FIRST_MISMATCH_REMAINING_VOXELS],
                final_t: f32::from_bits(first_words[vte::VTE_FIRST_MISMATCH_FINAL_T]),
                last_chunk: [
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_X] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_Y] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_Z] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_W] as i32,
                ],
            };
        } else {
            self.vte_first_mismatch = vte::VteFirstMismatch::default();
        }
    }

    fn refresh_vte_world_bvh_ray_diagnostics(&mut self, frame_idx: usize) {
        let expected =
            std::mem::take(&mut self.frames_in_flight[frame_idx].vte_world_bvh_ray_diag_expected);
        if expected.is_empty() {
            return;
        }

        let words = self.frames_in_flight[frame_idx]
            .live_buffers
            .vte_world_bvh_ray_diag_buffer
            .read()
            .unwrap();
        if words.len() < vte::VTE_WORLD_BVH_RAY_DIAG_WORD_COUNT {
            eprintln!(
                "[vte-world-bvh-ray-diag] invalid readback word count: got={} expected={}",
                words.len(),
                vte::VTE_WORLD_BVH_RAY_DIAG_WORD_COUNT
            );
            return;
        }

        let gpu_count = words[vte::VTE_WORLD_BVH_RAY_DIAG_COUNT_WORD]
            .min(vte::VTE_WORLD_BVH_RAY_DIAG_CAPACITY as u32) as usize;
        let mut mismatches = 0usize;
        let mut missing = 0usize;
        let mut hit_state_mismatches = 0usize;
        let mut material_mismatches = 0usize;
        let mut chunk_mismatches = 0usize;
        let mut distance_mismatches = 0usize;
        let mut metadata_mismatches = 0usize;
        let mut miss_reason_mismatches = 0usize;
        let mut first_mismatch_detail: Option<String> = None;

        for expected_record in &expected {
            let slot = expected_record.slot as usize;
            if slot >= gpu_count {
                missing += 1;
                mismatches += 1;
                if first_mismatch_detail.is_none() {
                    first_mismatch_detail = Some(format!(
                        "slot={} missing_gpu_record expected(px={},py={},l={})",
                        expected_record.slot,
                        expected_record.pixel_x,
                        expected_record.pixel_y,
                        expected_record.layer
                    ));
                }
                continue;
            }

            let base = vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_BASE_WORD
                + slot * vte::VTE_WORLD_BVH_RAY_DIAG_WORDS_PER_RECORD;
            if base + vte::VTE_WORLD_BVH_RAY_DIAG_WORDS_PER_RECORD > words.len() {
                missing += 1;
                mismatches += 1;
                if first_mismatch_detail.is_none() {
                    first_mismatch_detail =
                        Some(format!("slot={} out_of_bounds_record_base={}", slot, base));
                }
                continue;
            }

            let gpu_record = vte::VteWorldBvhRayDiagRecord {
                pixel_x: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_PIXEL_X],
                pixel_y: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_PIXEL_Y],
                layer: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_LAYER],
                hit: (words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_HIT_MASK] & 0x1) != 0,
                miss_reason: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_MISS_REASON],
                hit_material: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_HIT_MATERIAL],
                hit_chunk: [
                    words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_X] as i32,
                    words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_Y] as i32,
                    words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_Z] as i32,
                    words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_W] as i32,
                ],
                hit_t: f32::from_bits(words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_HIT_T_BITS]),
                chunk_steps: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_CHUNK_STEPS],
                remaining_voxels: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_REMAINING_VOXELS],
                node_visits: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_NODE_VISITS],
                path_hash: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_PATH_HASH],
                flags: words[base + vte::VTE_WORLD_BVH_RAY_DIAG_RECORD_FLAGS],
            };

            let mut record_mismatch = false;
            if gpu_record.pixel_x != expected_record.pixel_x
                || gpu_record.pixel_y != expected_record.pixel_y
                || gpu_record.layer != expected_record.layer
            {
                metadata_mismatches += 1;
                record_mismatch = true;
            }
            if gpu_record.hit != expected_record.hit {
                hit_state_mismatches += 1;
                record_mismatch = true;
            }
            if gpu_record.hit && expected_record.hit {
                if gpu_record.hit_material != expected_record.hit_material {
                    material_mismatches += 1;
                    record_mismatch = true;
                }
                if gpu_record.hit_chunk != expected_record.hit_chunk {
                    chunk_mismatches += 1;
                    record_mismatch = true;
                }
                let tol =
                    1e-3f32.max(1e-3f32 * expected_record.hit_t.abs().max(gpu_record.hit_t.abs()));
                if (gpu_record.hit_t - expected_record.hit_t).abs() > tol {
                    distance_mismatches += 1;
                    record_mismatch = true;
                }
            } else if gpu_record.miss_reason != expected_record.miss_reason {
                miss_reason_mismatches += 1;
                record_mismatch = true;
            }

            if record_mismatch {
                mismatches += 1;
                if first_mismatch_detail.is_none() {
                    first_mismatch_detail = Some(format!(
                        "slot={} exp(px={},py={},l={},hit={},mat={},chunk={:?},t={:.6},reason={},steps={},rem={}) \
gpu(px={},py={},l={},hit={},mat={},chunk={:?},t={:.6},reason={},steps={},rem={},node_visits={},path_hash=0x{:08x},flags=0x{:08x})",
                        expected_record.slot,
                        expected_record.pixel_x,
                        expected_record.pixel_y,
                        expected_record.layer,
                        expected_record.hit as u32,
                        expected_record.hit_material,
                        expected_record.hit_chunk,
                        expected_record.hit_t,
                        expected_record.miss_reason,
                        expected_record.chunk_steps,
                        expected_record.remaining_voxels,
                        gpu_record.pixel_x,
                        gpu_record.pixel_y,
                        gpu_record.layer,
                        gpu_record.hit as u32,
                        gpu_record.hit_material,
                        gpu_record.hit_chunk,
                        gpu_record.hit_t,
                        gpu_record.miss_reason,
                        gpu_record.chunk_steps,
                        gpu_record.remaining_voxels,
                        gpu_record.node_visits,
                        gpu_record.path_hash,
                        gpu_record.flags
                    ));
                }
            }
        }

        let now_frame = self.frames_rendered;
        let should_log = mismatches > 0
            || self
                .vte_world_bvh_ray_diag_last_log_frame
                .map(|last| now_frame.saturating_sub(last) >= self.vte_world_bvh_ray_diag_interval)
                .unwrap_or(true);
        if should_log {
            eprintln!(
                "[vte-world-bvh-ray-diag] frame={} samples={} gpu_records={} mismatches={} missing={} hit_state={} material={} chunk={} distance={} metadata={} miss_reason={}",
                now_frame,
                expected.len(),
                gpu_count,
                mismatches,
                missing,
                hit_state_mismatches,
                material_mismatches,
                chunk_mismatches,
                distance_mismatches,
                metadata_mismatches,
                miss_reason_mismatches
            );
            if let Some(detail) = first_mismatch_detail {
                eprintln!("[vte-world-bvh-ray-diag][first] {}", detail);
            }
            self.vte_world_bvh_ray_diag_last_log_frame = Some(now_frame);
        }
    }

    pub fn recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn aetna_pointer_moved(&mut self, x: f32, y: f32) -> (bool, Vec<aetna_core::UiEvent>) {
        let Some(aetna) = self.aetna_overlay.as_mut() else {
            return (false, Vec::new());
        };
        let moved = aetna.runner.pointer_moved(x, y);
        (moved.needs_redraw, moved.events)
    }

    pub fn aetna_pointer_left(&mut self) {
        if let Some(aetna) = self.aetna_overlay.as_mut() {
            aetna.runner.pointer_left();
        }
    }

    pub fn aetna_pointer_down(
        &mut self,
        x: f32,
        y: f32,
        button: aetna_core::PointerButton,
    ) -> Vec<aetna_core::UiEvent> {
        self.aetna_overlay
            .as_mut()
            .map(|aetna| aetna.runner.pointer_down(x, y, button))
            .unwrap_or_default()
    }

    pub fn aetna_pointer_up(
        &mut self,
        x: f32,
        y: f32,
        button: aetna_core::PointerButton,
    ) -> Vec<aetna_core::UiEvent> {
        self.aetna_overlay
            .as_mut()
            .map(|aetna| aetna.runner.pointer_up(x, y, button))
            .unwrap_or_default()
    }

    pub fn aetna_pointer_wheel(&mut self, x: f32, y: f32, dy: f32) -> bool {
        self.aetna_overlay
            .as_mut()
            .map(|aetna| aetna.runner.pointer_wheel(x, y, dy))
            .unwrap_or(false)
    }

    pub fn aetna_set_modifiers(&mut self, modifiers: aetna_core::KeyModifiers) {
        if let Some(aetna) = self.aetna_overlay.as_mut() {
            aetna.runner.set_modifiers(modifiers);
        }
    }

    /// Recreate all resolution-dependent GPU buffers at a new render size.
    /// Waits for in-flight GPU work to complete, then rebuilds sized buffers
    /// and all per-frame descriptor sets that reference them.
    pub fn recreate_sized_buffers(
        &mut self,
        new_dimensions: [u32; 3],
        pixel_storage_layers: Option<u32>,
    ) {
        self.wait_for_all_frames();

        let new_sized = SizedBuffers::new(
            self.memory_allocator.clone(),
            new_dimensions,
            pixel_storage_layers,
        );

        // The sized descriptor set layout is set_layouts[1] in the compute pipeline layout.
        let sized_ds_layout = self.compute_pipeline.pipeline_layout.set_layouts()[1].clone();

        for frame in &mut self.frames_in_flight {
            frame.sized_descriptor_set = new_sized.create_sized_descriptor_set(
                &frame.line_vertexes_buffer,
                self.descriptor_set_allocator.clone(),
                sized_ds_layout.clone(),
            );
        }

        self.sized_buffers = new_sized;

        // Reset profiler to avoid stale timing data from the old resolution.
        self.profiler.next_query = 0;
        self.profiler.phase_names.clear();
        self.profiler.accum.clear();
        self.profiler.total_frames = 0;
        self.profiler.last_frame_phases.clear();
        self.profiler.last_gpu_total_ms = 0.0;
        self.profiler.last_slow_report_frame = None;
        self.drop_next_profile_sample = true;

        eprintln!(
            "[render] Resized buffers to {}x{}x{}",
            new_dimensions[0], new_dimensions[1], new_dimensions[2]
        );
    }

    pub fn reset_gpu_profile_window(&mut self) {
        self.profiler.next_query = 0;
        self.profiler.phase_names.clear();
        self.profiler.accum.clear();
        self.profiler.total_frames = 0;
        self.profiler.last_frame_phases.clear();
        self.profiler.last_gpu_total_ms = 0.0;
        self.profiler.last_slow_report_frame = None;
        self.drop_next_profile_sample = true;
    }

    pub fn flush_gpu_profile_report_now(&mut self) {
        if self.profiler.total_frames == 0 && self.profiler.accum.is_empty() {
            return;
        }
        self.profiler.print_report();
        self.drop_next_profile_sample = true;
    }

    pub fn last_gpu_frame_ms(&self) -> f32 {
        self.profiler.last_gpu_total_ms
    }

    /// Last frame's per-phase GPU breakdown: Vec of (phase_name, ms).
    pub fn last_gpu_phase_breakdown(&self) -> &[(&'static str, f32)] {
        &self.profiler.last_frame_phases
    }

    /// Non-destructive snapshot of accumulated per-phase GPU averages.
    /// Returns Vec of (name, avg_ms, sample_count).
    pub fn gpu_phase_averages(&self) -> Vec<(&'static str, f64, usize)> {
        self.profiler.phase_averages_snapshot()
    }

    pub fn voxel_buffer_capacities(&self) -> (usize, usize, usize, usize) {
        self.frames_in_flight
            .first()
            .map(|frame| {
                let caps = frame.live_buffers.voxel_capacities();
                (
                    caps.dense_chunks,
                    caps.leaf_headers,
                    caps.region_bvh_nodes,
                    caps.leaf_chunk_entries,
                )
            })
            .unwrap_or((0, 0, 0, 0))
    }

    fn wait_for_all_frames(&mut self) {
        for frame in &mut self.frames_in_flight {
            if let Some(future) = frame.fence.take() {
                if self.stall_trace {
                    eprintln!("[stall] wait_for_all_frames: begin");
                }
                let wait_start = Instant::now();
                let f = future.then_signal_fence_and_flush().unwrap();
                f.wait(None).unwrap();
                if self.stall_trace {
                    eprintln!(
                        "[stall] wait_for_all_frames: end ({:.2} ms)",
                        wait_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
        }
    }

    fn next_grow_capacity(current: usize, required: usize) -> usize {
        if current >= required {
            return current.max(1);
        }
        let mut cap = current.max(1);
        while cap < required {
            let next = cap.saturating_mul(2);
            if next <= cap {
                return required;
            }
            cap = next;
        }
        cap
    }

    fn ensure_live_voxel_buffer_capacity(
        &mut self,
        required: VoxelBufferCapacities,
    ) -> VoxelBufferCapacities {
        let required = required.with_minimums();
        let current = self.frames_in_flight[0].live_buffers.voxel_capacities();
        let grown = VoxelBufferCapacities {
            dense_chunks: Self::next_grow_capacity(current.dense_chunks, required.dense_chunks),
            leaf_headers: Self::next_grow_capacity(current.leaf_headers, required.leaf_headers),
            region_bvh_nodes: Self::next_grow_capacity(
                current.region_bvh_nodes,
                required.region_bvh_nodes,
            ),
            leaf_chunk_entries: Self::next_grow_capacity(
                current.leaf_chunk_entries,
                required.leaf_chunk_entries,
            ),
        };
        if grown == current {
            return current;
        }

        self.wait_for_all_frames();
        let live_layout = self
            .compute_pipeline
            .pipeline_layout
            .set_layouts()
            .get(2)
            .cloned()
            .expect("live descriptor set layout");
        for frame in &mut self.frames_in_flight {
            // Only replace the voxel buffers, keeping non-voxel buffers
            // (working_data, model_instance, voxel_frame_meta) intact.
            // Creating entirely new LiveBuffers would zero those buffers,
            // discarding data already written earlier in this frame.
            let new_voxel = VoxelGpuBuffers::new(self.memory_allocator.clone(), grown);
            frame.live_buffers.install_voxel_buffers(
                new_voxel,
                self.descriptor_set_allocator.clone(),
                live_layout.clone(),
            );
            frame.last_voxel_metadata_generation = None;
            frame.vte_compare_enabled = false;
            frame.vte_world_bvh_ray_diag_enabled = false;
            frame.vte_world_bvh_ray_diag_expected.clear();
        }
        eprintln!(
            "[vte-buffers-grow] dense {}->{} leaf_headers {}->{} bvh_nodes {}->{} leaf_entries {}->{}",
            current.dense_chunks,
            grown.dense_chunks,
            current.leaf_headers,
            grown.leaf_headers,
            current.region_bvh_nodes,
            grown.region_bvh_nodes,
            current.leaf_chunk_entries,
            grown.leaf_chunk_entries
        );
        grown
    }

    /// Install pre-populated voxel GPU buffers from a background rebuild.
    /// Waits for all in-flight frames, replaces voxel buffers and descriptor sets,
    /// then sets metadata generation so render_internal skips redundant uploads.
    pub fn install_new_voxel_gpu_buffers(
        &mut self,
        gpu_buffers: VoxelGpuBuffers,
        metadata_generation: u64,
    ) {
        self.wait_for_all_frames();
        let live_layout = self
            .compute_pipeline
            .pipeline_layout
            .set_layouts()
            .get(2)
            .cloned()
            .expect("live descriptor set layout");
        for frame in &mut self.frames_in_flight {
            let new_voxel = VoxelGpuBuffers::new(self.memory_allocator.clone(), gpu_buffers.caps);
            // Copy data from the pre-populated buffers into per-frame buffers.
            // Each frame needs its own buffer set for safe concurrent GPU access.
            {
                let src = gpu_buffers.chunk_headers_buffer.read().unwrap();
                let mut dst = new_voxel.chunk_headers_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            {
                let src = gpu_buffers.occupancy_words_buffer.read().unwrap();
                let mut dst = new_voxel.occupancy_words_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            {
                let src = gpu_buffers.material_words_buffer.read().unwrap();
                let mut dst = new_voxel.material_words_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            {
                let src = gpu_buffers.orientation_words_buffer.read().unwrap();
                let mut dst = new_voxel.orientation_words_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            {
                let src = gpu_buffers.macro_words_buffer.read().unwrap();
                let mut dst = new_voxel.macro_words_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            {
                let src = gpu_buffers.leaf_headers_buffer.read().unwrap();
                let mut dst = new_voxel.leaf_headers_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            {
                let src = gpu_buffers.region_bvh_nodes_buffer.read().unwrap();
                let mut dst = new_voxel.region_bvh_nodes_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            {
                let src = gpu_buffers.leaf_chunk_entries_buffer.read().unwrap();
                let mut dst = new_voxel.leaf_chunk_entries_buffer.write().unwrap();
                dst[..src.len()].copy_from_slice(&src);
            }
            frame.live_buffers.install_voxel_buffers(
                new_voxel,
                self.descriptor_set_allocator.clone(),
                live_layout.clone(),
            );
            frame.last_voxel_metadata_generation = Some(metadata_generation);
        }
        eprintln!(
            "[vte-buffers-swap] installed pre-built GPU buffers (gen={}, caps={:?})",
            metadata_generation, gpu_buffers.caps
        );
    }

    /// Get a clone of the GPU memory allocator for use in background voxel rebuilds.
    pub fn memory_allocator(&self) -> Arc<dyn MemoryAllocator> {
        self.memory_allocator.clone()
    }

    /// Upload a 3D texture to the GPU texture pool.
    /// Returns the pool slot index (0..255), or None if the pool is full.
    pub fn upload_texture_3d(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        depth: u32,
        format: Format,
    ) -> Option<u16> {
        self.texture_pool
            .upload_texture_3d(data, width, height, depth, format)
    }

    pub fn render_tetra_frame(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        mut frame_params: FrameParams,
        tetra_input: TetraFrameInput<'_>,
    ) {
        if frame_params.render_options.render_backend == RenderBackend::VoxelTraversal {
            eprintln!(
                "render_tetra_frame called with '{}' backend; forcing '{}'.",
                RenderBackend::VoxelTraversal.label(),
                RenderBackend::TetraRaster.label()
            );
            frame_params.render_options.render_backend = RenderBackend::TetraRaster;
        }
        self.render_internal(
            device,
            queue,
            frame_params,
            tetra_input.model_instances,
            &[],
            None,
        );
    }

    pub fn render_voxel_frame(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        mut frame_params: FrameParams,
        voxel_input: VoxelFrameInput<'_>,
        tetra_entity_instances: &[common::ModelInstance],
        tetra_overlay_instances: &[common::ModelInstance],
    ) {
        frame_params.render_options.render_backend = RenderBackend::VoxelTraversal;
        self.render_internal(
            device,
            queue,
            frame_params,
            tetra_entity_instances,
            tetra_overlay_instances,
            Some(&voxel_input),
        );
    }

    fn render_internal(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        frame_params: FrameParams,
        model_instances_input: &[common::ModelInstance],
        raster_overlay_instances_input: &[common::ModelInstance],
        voxel_input: Option<&VoxelFrameInput<'_>>,
    ) {
        let FrameParams {
            view_matrix,
            time_ticks_ms,
            focal_length_xy,
            focal_length_zw,
            mut render_options,
        } = frame_params;
        let mut aetna_ui = render_options.aetna_ui.take();
        let view_matrix_view = view_matrix.into_owned();

        // Guard against non-finite transforms/material data poisoning shared
        // non-voxel preprocess/BVH buffers for the entire frame.
        let mut dropped_model_instance_count = 0usize;
        let mut dropped_overlay_instance_count = 0usize;
        let mut dropped_custom_overlay_edge_instance_count = 0usize;
        let model_instances =
            filter_finite_instances(model_instances_input, &mut dropped_model_instance_count);
        let raster_overlay_instances = filter_finite_instances(
            raster_overlay_instances_input,
            &mut dropped_overlay_instance_count,
        );
        let custom_overlay_edge_instances = filter_finite_instances(
            &render_options.custom_overlay_edge_instances,
            &mut dropped_custom_overlay_edge_instance_count,
        );

        if dropped_model_instance_count > 0
            || dropped_overlay_instance_count > 0
            || dropped_custom_overlay_edge_instance_count > 0
        {
            eprintln!(
                "Dropped non-finite model instances before render: non-voxel {} overlay {} edge_overlay {} (frame {}).",
                dropped_model_instance_count,
                dropped_overlay_instance_count,
                dropped_custom_overlay_edge_instance_count,
                self.frames_rendered
            );
        }

        let slice = view_matrix_view.view().to_slice().unwrap();
        let view_matrix_nalgebra: nalgebra::OMatrix<f32, nalgebra::U5, nalgebra::U5> =
            nalgebra::Matrix5::from_column_slice(slice).transpose();
        let view_matrix_nalgebra_inv = view_matrix_nalgebra.try_inverse().unwrap();

        if let Some(window) = self.window.clone() {
            let window_size = window.inner_size();
            // Do not draw the frame when the screen size is zero. On Windows, this can occur
            // when minimizing the application.
            if window_size.width == 0 || window_size.height == 0 {
                return;
            }
        }

        // CPU frame time tracking
        let render_start = Instant::now();
        if let Some(prev_start) = self.last_render_start {
            self.frame_time_ms = (render_start - prev_start).as_secs_f32() * 1000.0;
        }
        self.last_render_start = Some(render_start);

        let frame_idx = self.frames_rendered % FRAMES_IN_FLIGHT;

        // Wait for this frame slot's previous GPU work to complete before writing its buffers.
        // This protects the per-frame LiveBuffers/line vertex buffer from being overwritten
        // while still in use by the GPU.
        if let Some(prev_fence) = self.frames_in_flight[frame_idx].fence.take() {
            if self.stall_trace {
                eprintln!(
                    "[stall] frame={} slot_wait begin (slot={})",
                    self.frames_rendered, frame_idx
                );
            }
            let wait_start = Instant::now();
            let f = prev_fence.then_signal_fence_and_flush().unwrap();
            f.wait(None).unwrap();
            if self.stall_trace {
                eprintln!(
                    "[stall] frame={} slot_wait end ({:.2} ms)",
                    self.frames_rendered,
                    wait_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        if self.frames_in_flight[frame_idx].vte_compare_enabled {
            self.refresh_vte_compare_diagnostics(frame_idx);
        } else {
            self.clear_vte_compare_diagnostics();
        }
        if self.frames_in_flight[frame_idx].vte_world_bvh_ray_diag_enabled {
            self.refresh_vte_world_bvh_ray_diagnostics(frame_idx);
        } else {
            self.frames_in_flight[frame_idx]
                .vte_world_bvh_ray_diag_expected
                .clear();
        }

        if let Some(swapchain) = self.swapchain.clone() {
            if let Some(window) = self.window.clone() {
                if let Some(render_pass) = self.render_pass.clone() {
                    let window_size = window.inner_size();
                    // Whenever the window resizes we need to recreate everything dependent on the
                    // window size. In this example that includes the swapchain, the framebuffers and
                    // the dynamic state viewport.
                    if self.recreate_swapchain {
                        // Use the new dimensions of the window.

                        let (new_swapchain, new_images) = swapchain
                            .recreate(SwapchainCreateInfo {
                                image_extent: window_size.into(),
                                ..swapchain.create_info()
                            })
                            .expect("failed to recreate swapchain");

                        self.swapchain = Some(new_swapchain);

                        // Because framebuffers contains a reference to the old swapchain, we need to
                        // recreate framebuffers as well.
                        self.framebuffers =
                            Some(window_size_dependent_setup(&new_images, &render_pass));

                        self.viewport.extent = window_size.into();
                        if let Some(aetna) = self.aetna_overlay.as_mut() {
                            aetna.runner.set_surface_size(
                                window_size.width.max(1),
                                window_size.height.max(1),
                            );
                        }

                        self.recreate_swapchain = false;

                        let [capture_w, capture_h] = self
                            .swapchain
                            .as_ref()
                            .map(|s| s.image_extent())
                            .unwrap_or([window_size.width, window_size.height]);
                        let capture_format = self
                            .swapchain
                            .as_ref()
                            .map(|s| s.image_format())
                            .unwrap_or(Format::R8G8B8A8_UNORM);
                        self.cpu_screen_capture_buffer = create_cpu_screencapture_buffer(
                            self.memory_allocator.clone(),
                            capture_w,
                            capture_h,
                            capture_format,
                        );
                    }
                }
            }
        }

        let model_instance_capacity = usize::try_from(
            self.frames_in_flight[frame_idx]
                .live_buffers
                .model_instance_buffer
                .len(),
        )
        .unwrap_or(usize::MAX);
        let use_split_voxel_instances = voxel_input.is_some();
        let non_voxel_used_instance_count = model_instances.len().min(model_instance_capacity);
        let overlay_instance_capacity =
            model_instance_capacity.saturating_sub(non_voxel_used_instance_count);
        let overlay_used_instance_count = if use_split_voxel_instances {
            raster_overlay_instances
                .len()
                .min(overlay_instance_capacity)
        } else {
            0
        };
        let used_instance_count = non_voxel_used_instance_count + overlay_used_instance_count;
        let custom_overlay_edge_instance_base = used_instance_count;
        let custom_overlay_edge_instance_capacity =
            model_instance_capacity.saturating_sub(custom_overlay_edge_instance_base);
        let custom_overlay_edge_used_instance_count = custom_overlay_edge_instances
            .len()
            .min(custom_overlay_edge_instance_capacity);

        let requested_tetrahedron_count =
            self.one_time_buffers.model_tetrahedron_count * non_voxel_used_instance_count;
        let total_tetrahedron_count =
            requested_tetrahedron_count.min(self.sized_buffers.max_tetrahedrons);
        let requested_raster_overlay_tetrahedron_count =
            self.one_time_buffers.model_tetrahedron_count * overlay_used_instance_count;
        let raster_overlay_tetrahedron_count =
            requested_raster_overlay_tetrahedron_count.min(self.sized_buffers.max_tetrahedrons);
        let raster_instance_base = if use_split_voxel_instances {
            non_voxel_used_instance_count
        } else {
            0
        };
        let raster_tetrahedron_count = if use_split_voxel_instances {
            raster_overlay_tetrahedron_count
        } else {
            total_tetrahedron_count
        };
        let non_voxel_bvh_leaf_count = if use_split_voxel_instances {
            non_voxel_used_instance_count
        } else {
            total_tetrahedron_count
        };
        let (non_voxel_translation_abs_max, non_voxel_basis_abs_max, non_voxel_outlier_count) =
            model_instance_transform_extrema(&model_instances[..non_voxel_used_instance_count]);

        // Debug: print scene info on first frame only
        if self.frames_rendered == 0 {
            if use_split_voxel_instances {
                if model_instances.len() > non_voxel_used_instance_count {
                    eprintln!(
                        "VTE non-voxel input truncated to buffer capacity: {} -> {}",
                        model_instances.len(),
                        non_voxel_used_instance_count
                    );
                }
                if raster_overlay_instances.len() > overlay_used_instance_count {
                    eprintln!(
                        "VTE raster overlay input truncated to buffer capacity: {} -> {}",
                        raster_overlay_instances.len(),
                        overlay_used_instance_count
                    );
                }
                if requested_tetrahedron_count > total_tetrahedron_count {
                    eprintln!(
                        "VTE non-voxel tetrahedrons truncated to buffer capacity: {} -> {}",
                        requested_tetrahedron_count, total_tetrahedron_count
                    );
                }
                if requested_raster_overlay_tetrahedron_count > raster_overlay_tetrahedron_count {
                    eprintln!(
                        "VTE raster overlay tetrahedrons truncated to buffer capacity: {} -> {}",
                        requested_raster_overlay_tetrahedron_count,
                        raster_overlay_tetrahedron_count
                    );
                }
                println!(
                    "VTE scene: {} non-voxel tetrahedrons, {} raster overlay tetrahedrons ({} per instance × non_voxel_instances={}, overlays={})",
                    total_tetrahedron_count,
                    raster_overlay_tetrahedron_count,
                    self.one_time_buffers.model_tetrahedron_count,
                    non_voxel_used_instance_count,
                    overlay_used_instance_count
                );
                println!(
                    "VTE non-voxel BVH: {} internal nodes, {} total nodes (leaf_count={})",
                    non_voxel_bvh_leaf_count.saturating_sub(1),
                    2 * non_voxel_bvh_leaf_count.saturating_sub(1) + 1,
                    non_voxel_bvh_leaf_count
                );
            } else {
                if model_instances.len() > used_instance_count {
                    eprintln!(
                        "Model instance input truncated to buffer capacity: {} -> {}",
                        model_instances.len(),
                        used_instance_count
                    );
                }
                if requested_tetrahedron_count > total_tetrahedron_count {
                    eprintln!(
                        "Tetrahedron input truncated to buffer capacity: {} -> {}",
                        requested_tetrahedron_count, total_tetrahedron_count
                    );
                }
                println!(
                    "Scene: {} tetrahedrons ({} per instance × {} instances)",
                    total_tetrahedron_count,
                    self.one_time_buffers.model_tetrahedron_count,
                    used_instance_count
                );
                println!(
                    "BVH: {} internal nodes, {} total nodes",
                    total_tetrahedron_count.saturating_sub(1),
                    2 * total_tetrahedron_count.saturating_sub(1) + 1
                );
            }
            if custom_overlay_edge_instances.len() > custom_overlay_edge_used_instance_count {
                eprintln!(
                    "Custom edge-overlay instances truncated to buffer capacity: {} -> {}",
                    custom_overlay_edge_instances.len(),
                    custom_overlay_edge_used_instance_count
                );
            }
        }
        let world_origin_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 0.0, 0.0, 0.0, 1.0]);
        let world_origin_inv_w = if world_origin_h[4].abs() > 1e-6 {
            1.0 / world_origin_h[4]
        } else {
            1.0
        };
        let world_origin = glam::Vec4::new(
            world_origin_h[0] * world_origin_inv_w,
            world_origin_h[1] * world_origin_inv_w,
            world_origin_h[2] * world_origin_inv_w,
            world_origin_h[3] * world_origin_inv_w,
        );
        let world_dir_x_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [1.0, 0.0, 0.0, 0.0, 0.0]);
        let world_dir_y_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 1.0, 0.0, 0.0, 0.0]);
        let world_dir_z_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 0.0, 1.0, 0.0, 0.0]);
        let world_dir_w_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 0.0, 0.0, 1.0, 0.0]);
        let world_dir_x = glam::Vec4::new(
            world_dir_x_h[0],
            world_dir_x_h[1],
            world_dir_x_h[2],
            world_dir_x_h[3],
        );
        let world_dir_y = glam::Vec4::new(
            world_dir_y_h[0],
            world_dir_y_h[1],
            world_dir_y_h[2],
            world_dir_y_h[3],
        );
        let world_dir_z = glam::Vec4::new(
            world_dir_z_h[0],
            world_dir_z_h[1],
            world_dir_z_h[2],
            world_dir_z_h[3],
        );
        let world_dir_w = glam::Vec4::new(
            world_dir_w_h[0],
            world_dir_w_h[1],
            world_dir_w_h[2],
            world_dir_w_h[3],
        );
        let present_dimensions = match self.window.clone() {
            None => [
                self.sized_buffers.render_dimensions[0].max(1),
                self.sized_buffers.render_dimensions[1].max(1),
            ],
            Some(window) => {
                let window_size = window.inner_size();
                [window_size.width.max(1), window_size.height.max(1)]
            }
        };
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .working_data_buffer
                .write()
                .unwrap();
            writer.view_matrix = view_matrix_nalgebra.into();
            writer.view_matrix_inverse = view_matrix_nalgebra_inv.into();
            writer.render_dimensions = glam::UVec4::new(
                self.sized_buffers.render_dimensions[0],
                self.sized_buffers.render_dimensions[1],
                self.sized_buffers.render_dimensions[2],
                0,
            );
            writer.present_dimensions =
                glam::UVec2::new(present_dimensions[0], present_dimensions[1]);
            writer.total_num_tetrahedrons = non_voxel_bvh_leaf_count as u32;
            writer.raytrace_seed = 6364136223846793005u64
                .wrapping_mul(self.frames_rendered as u64)
                .wrapping_add(1442695040888963407);
            writer.time_ticks_ms = time_ticks_ms;
            writer.focal_length_xy = focal_length_xy;
            writer.focal_length_zw = focal_length_zw;
            let mut working_flags = 0u32;
            if voxel_input.is_some() {
                working_flags |= WORKING_FLAG_VTE_COLLAPSED;
            }
            if render_options.zw_angle_color_shift_enabled {
                working_flags |= WORKING_FLAG_ZW_ANGLE_COLOR_SHIFT;
            }
            let zw_shift_strength_q = (render_options.zw_angle_color_shift_strength.clamp(0.0, 1.0)
                * 255.0)
                .round() as u32;
            working_flags |= zw_shift_strength_q << WORKING_ZW_SHIFT_STRENGTH_SHIFT;
            // Flag used by present shader:
            // - padding[0] bit0: 0 = legacy per-layer accumulation, 1 = VTE Stage-B-collapsed output in layer 0.
            // - padding[0] bit1: ZW angle color shift enabled.
            // - padding[0] bits8..15: quantized ZW angle color shift strength [0, 255].
            // padding[1] carries VTE stage_b_mode so present shader can conditionally
            // bypass tone mapping for debug compare output.
            writer.padding = [working_flags, render_options.vte_display_mode.as_u32()];
            writer.world_origin = world_origin;
            writer.world_dir_x = world_dir_x;
            writer.world_dir_y = world_dir_y;
            writer.world_dir_z = world_dir_z;
            writer.world_dir_w = world_dir_w;
        }

        // Compute non-voxel scene hash BEFORE writing to the GPU buffer so the
        // BVH rebuild decision later in this frame reflects what is actually in
        // the buffer.  Using a per-frame hash avoids stale comparisons when
        // multiple frames are in flight.
        let non_voxel_scene_hash = if non_voxel_bvh_leaf_count == 0 {
            0u64
        } else {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            non_voxel_used_instance_count.hash(&mut hasher);
            non_voxel_bvh_leaf_count.hash(&mut hasher);
            bytemuck::cast_slice::<_, u8>(&model_instances[..non_voxel_used_instance_count])
                .hash(&mut hasher);
            hasher.finish()
        };

        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .model_instance_buffer
                .write()
                .unwrap();
            for i in 0..non_voxel_used_instance_count {
                writer[i] = model_instances[i];
            }
            for i in 0..overlay_used_instance_count {
                writer[non_voxel_used_instance_count + i] = raster_overlay_instances[i];
            }
            for i in 0..custom_overlay_edge_used_instance_count {
                writer[custom_overlay_edge_instance_base + i] = custom_overlay_edge_instances[i];
            }
        }

        let mut vte_chunk_count: usize = 0;
        let mut vte_leaf_count: usize = 0;
        let mut vte_region_bvh_node_count: usize = 0;
        let mut vte_region_bvh_root_index: u32 = vte::VTE_REGION_BVH_INVALID_NODE;
        let mut vte_leaf_chunk_entry_count: usize = 0;
        let mut vte_occupancy_word_count: usize = 0;
        let mut vte_material_word_count: usize = 0;
        let mut vte_orientation_word_count: usize = 0;
        let mut vte_macro_word_count: usize = 0;
        let mut vte_visible_lod_counts = [0u32; 3];
        let mut vte_visible_world_min = [0.0f32; 4];
        let mut vte_visible_world_max = [0.0f32; 4];
        let mut vte_buffer_caps = self.frames_in_flight[frame_idx]
            .live_buffers
            .voxel_capacities();
        if let Some(input) = voxel_input {
            let ceil_div = |value: usize, divisor: usize| -> usize {
                if divisor == 0 {
                    0
                } else {
                    value.saturating_add(divisor - 1) / divisor
                }
            };
            let dense_required = input
                .chunk_headers
                .len()
                .max(ceil_div(
                    input.occupancy_words.len(),
                    vte::VTE_OCCUPANCY_WORDS_PER_CHUNK,
                ))
                .max(ceil_div(
                    input.material_words.len(),
                    vte::VTE_MATERIAL_WORDS_PER_CHUNK,
                ))
                .max(ceil_div(
                    input.orientation_words.len(),
                    vte::VTE_ORIENTATION_WORDS_PER_CHUNK,
                ))
                .max(ceil_div(
                    input.macro_words.len(),
                    vte::VTE_MACRO_WORDS_PER_CHUNK,
                ));
            let required_caps = VoxelBufferCapacities {
                dense_chunks: dense_required,
                leaf_headers: input.leaf_headers.len(),
                region_bvh_nodes: input.region_bvh_nodes.len(),
                leaf_chunk_entries: input.leaf_chunk_entries.len(),
            };
            vte_buffer_caps = self.ensure_live_voxel_buffer_capacity(required_caps);

            vte_chunk_count = input.chunk_headers.len().min(vte_buffer_caps.dense_chunks);
            vte_leaf_count = input.leaf_headers.len().min(vte_buffer_caps.leaf_headers);
            vte_region_bvh_node_count = input
                .region_bvh_nodes
                .len()
                .min(vte_buffer_caps.region_bvh_nodes);
            vte_region_bvh_root_index = input.region_bvh_root_index;
            vte_leaf_chunk_entry_count = input
                .leaf_chunk_entries
                .len()
                .min(vte_buffer_caps.leaf_chunk_entries);
            vte_occupancy_word_count = input
                .occupancy_words
                .len()
                .min(vte_buffer_caps.occupancy_words());
            vte_material_word_count = input
                .material_words
                .len()
                .min(vte_buffer_caps.material_words());
            vte_orientation_word_count = input
                .orientation_words
                .len()
                .min(vte_buffer_caps.orientation_words());
            vte_macro_word_count = input.macro_words.len().min(vte_buffer_caps.macro_words());
            vte_visible_lod_counts[0] = vte_leaf_count as u32;

            if input.chunk_headers.len() > vte_chunk_count
                || input.leaf_headers.len() > vte_leaf_count
                || input.region_bvh_nodes.len() > vte_region_bvh_node_count
                || input.leaf_chunk_entries.len() > vte_leaf_chunk_entry_count
                || input.occupancy_words.len() > vte_occupancy_word_count
                || input.material_words.len() > vte_material_word_count
                || input.orientation_words.len() > vte_orientation_word_count
                || input.macro_words.len() > vte_macro_word_count
            {
                eprintln!(
                    "VTE input truncated to capacities: dense_chunks {}->{}, leaves {}->{}, bvh_nodes {}->{}, leaf_entries {}->{}, occ_words {}->{}, mat_words {}->{}, ori_words {}->{}, macro_words {}->{}",
                    input.chunk_headers.len(),
                    vte_chunk_count,
                    input.leaf_headers.len(),
                    vte_leaf_count,
                    input.region_bvh_nodes.len(),
                    vte_region_bvh_node_count,
                    input.leaf_chunk_entries.len(),
                    vte_leaf_chunk_entry_count,
                    input.occupancy_words.len(),
                    vte_occupancy_word_count,
                    input.material_words.len(),
                    vte_material_word_count,
                    input.orientation_words.len(),
                    vte_orientation_word_count,
                    input.macro_words.len(),
                    vte_macro_word_count
                );
            }
            if (vte_region_bvh_root_index as usize) >= vte_region_bvh_node_count {
                vte_region_bvh_root_index = vte::VTE_REGION_BVH_INVALID_NODE;
            }

            let frame = &mut self.frames_in_flight[frame_idx];
            let metadata_dirty =
                frame.last_voxel_metadata_generation != Some(input.metadata_generation);

            if metadata_dirty {
                let chunk_headers = &input.chunk_headers[..vte_chunk_count];
                let occupancy_words = &input.occupancy_words[..vte_occupancy_word_count];
                let material_words = &input.material_words[..vte_material_word_count];
                let orientation_words = &input.orientation_words[..vte_orientation_word_count];
                let leaf_headers = &input.leaf_headers[..vte_leaf_count];
                let region_bvh_nodes = &input.region_bvh_nodes[..vte_region_bvh_node_count];
                let leaf_chunk_entries = &input.leaf_chunk_entries[..vte_leaf_chunk_entry_count];
                let macro_words = &input.macro_words[..vte_macro_word_count];
                // When the frame has no previous generation (e.g. after voxel buffer
                // capacity growth), we must do a full upload. The mutation_batch only
                // covers incremental changes — it assumes buffers already contain
                // the baseline data.
                let force_full_upload = frame.last_voxel_metadata_generation.is_none();
                if let Some(batch) = input.mutation_batch.filter(|_| !force_full_upload) {
                    for write in &batch.chunk_header_writes {
                        let Some((dst, src)) =
                            clamp_write_span(write.start, write.values.len(), vte_chunk_count)
                        else {
                            continue;
                        };
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .chunk_headers_buffer
                            .write()
                            .unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }

                    for write in &batch.occupancy_word_writes {
                        let Some((dst, src)) = clamp_write_span(
                            write.start,
                            write.values.len(),
                            vte_occupancy_word_count,
                        ) else {
                            continue;
                        };
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .occupancy_words_buffer
                            .write()
                            .unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }

                    for write in &batch.material_word_writes {
                        let Some((dst, src)) = clamp_write_span(
                            write.start,
                            write.values.len(),
                            vte_material_word_count,
                        ) else {
                            continue;
                        };
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .material_words_buffer
                            .write()
                            .unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }

                    for write in &batch.orientation_word_writes {
                        let Some((dst, src)) = clamp_write_span(
                            write.start,
                            write.values.len(),
                            vte_orientation_word_count,
                        ) else {
                            continue;
                        };
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .orientation_words_buffer
                            .write()
                            .unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }

                    for write in &batch.macro_word_writes {
                        let Some((dst, src)) =
                            clamp_write_span(write.start, write.values.len(), vte_macro_word_count)
                        else {
                            continue;
                        };
                        let mut writer =
                            frame.live_buffers.voxel.macro_words_buffer.write().unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }

                    for write in &batch.region_bvh_node_writes {
                        let Some((dst, src)) = clamp_write_span(
                            write.start,
                            write.values.len(),
                            vte_region_bvh_node_count,
                        ) else {
                            continue;
                        };
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .region_bvh_nodes_buffer
                            .write()
                            .unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }

                    for write in &batch.leaf_header_writes {
                        let Some((dst, src)) =
                            clamp_write_span(write.start, write.values.len(), vte_leaf_count)
                        else {
                            continue;
                        };
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .leaf_headers_buffer
                            .write()
                            .unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }

                    for write in &batch.leaf_chunk_entry_writes {
                        let Some((dst, src)) = clamp_write_span(
                            write.start,
                            write.values.len(),
                            vte_leaf_chunk_entry_count,
                        ) else {
                            continue;
                        };
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .leaf_chunk_entries_buffer
                            .write()
                            .unwrap();
                        writer[dst].copy_from_slice(&write.values[src]);
                    }
                } else {
                    // No mutation batch (or forced full upload for empty buffers).
                    // When force_full_upload is true, set dirty to None so all
                    // .or_else() fallbacks trigger full-range uploads.
                    let dirty = if force_full_upload {
                        None
                    } else {
                        input.dirty_ranges
                    };

                    let chunk_headers_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.chunk_headers.clone()),
                        vte_chunk_count,
                    )
                    .or_else(|| (vte_chunk_count > 0).then_some(0..vte_chunk_count));
                    if let Some(range) = chunk_headers_dirty {
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .chunk_headers_buffer
                            .write()
                            .unwrap();
                        writer[range.clone()].copy_from_slice(&chunk_headers[range]);
                    }

                    let occupancy_words_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.occupancy_words.clone()),
                        vte_occupancy_word_count,
                    )
                    .or_else(|| {
                        (vte_occupancy_word_count > 0).then_some(0..vte_occupancy_word_count)
                    });
                    if let Some(range) = occupancy_words_dirty {
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .occupancy_words_buffer
                            .write()
                            .unwrap();
                        writer[range.clone()].copy_from_slice(&occupancy_words[range]);
                    }

                    let material_words_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.material_words.clone()),
                        vte_material_word_count,
                    )
                    .or_else(|| {
                        (vte_material_word_count > 0).then_some(0..vte_material_word_count)
                    });
                    if let Some(range) = material_words_dirty {
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .material_words_buffer
                            .write()
                            .unwrap();
                        writer[range.clone()].copy_from_slice(&material_words[range]);
                    }

                    let orientation_words_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.orientation_words.clone()),
                        vte_orientation_word_count,
                    )
                    .or_else(|| {
                        (vte_orientation_word_count > 0).then_some(0..vte_orientation_word_count)
                    });
                    if let Some(range) = orientation_words_dirty {
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .orientation_words_buffer
                            .write()
                            .unwrap();
                        writer[range.clone()].copy_from_slice(&orientation_words[range]);
                    }

                    let leaf_headers_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.leaf_headers.clone()),
                        vte_leaf_count,
                    )
                    .or_else(|| (vte_leaf_count > 0).then_some(0..vte_leaf_count));
                    if let Some(range) = leaf_headers_dirty {
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .leaf_headers_buffer
                            .write()
                            .unwrap();
                        writer[range.clone()].copy_from_slice(&leaf_headers[range]);
                    }

                    let region_bvh_nodes_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.region_bvh_nodes.clone()),
                        vte_region_bvh_node_count,
                    )
                    .or_else(|| {
                        (vte_region_bvh_node_count > 0).then_some(0..vte_region_bvh_node_count)
                    });
                    if let Some(range) = region_bvh_nodes_dirty {
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .region_bvh_nodes_buffer
                            .write()
                            .unwrap();
                        writer[range.clone()].copy_from_slice(&region_bvh_nodes[range]);
                    }

                    let leaf_chunk_entries_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.leaf_chunk_entries.clone()),
                        vte_leaf_chunk_entry_count,
                    )
                    .or_else(|| {
                        (vte_leaf_chunk_entry_count > 0).then_some(0..vte_leaf_chunk_entry_count)
                    });
                    if let Some(range) = leaf_chunk_entries_dirty {
                        let mut writer = frame
                            .live_buffers
                            .voxel
                            .leaf_chunk_entries_buffer
                            .write()
                            .unwrap();
                        writer[range.clone()].copy_from_slice(&leaf_chunk_entries[range]);
                    }

                    let macro_words_dirty = clamp_dirty_range(
                        dirty.and_then(|ranges| ranges.macro_words.clone()),
                        vte_macro_word_count,
                    )
                    .or_else(|| (vte_macro_word_count > 0).then_some(0..vte_macro_word_count));
                    if let Some(range) = macro_words_dirty {
                        let mut writer =
                            frame.live_buffers.voxel.macro_words_buffer.write().unwrap();
                        writer[range.clone()].copy_from_slice(&macro_words[range]);
                    }
                }

                frame.last_voxel_metadata_generation = Some(input.metadata_generation);
            }
            if vte_region_bvh_node_count > 0
                && (vte_region_bvh_root_index as usize) < vte_region_bvh_node_count
            {
                let root = input.region_bvh_nodes[vte_region_bvh_root_index as usize];
                vte_visible_world_min = root.world_min;
                vte_visible_world_max = root.world_max;
            } else {
                vte_visible_world_min = [0.0; 4];
                vte_visible_world_max = [0.0; 4];
            }
        }
        let vte_world_bvh_ray_diag_active =
            voxel_input.is_some() && self.vte_world_bvh_ray_diag_enabled;
        let vte_world_bvh_ray_diag_sample_count = if vte_world_bvh_ray_diag_active {
            self.vte_world_bvh_ray_diag_samples
                .min(vte::VTE_WORLD_BVH_RAY_DIAG_CAPACITY) as u32
        } else {
            0
        };
        let vte_world_bvh_ray_diag_seed = (self.frames_rendered as u32)
            .wrapping_mul(747_796_405)
            .wrapping_add(2_891_336_453);
        let written_voxel_frame_meta;
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .voxel_frame_meta_buffer
                .write()
                .unwrap();
            let layer_count = self.sized_buffers.render_dimensions[2].max(1);
            let default_slice_layer = (layer_count - 1) / 2;
            let stage_b_slice_layer = render_options
                .vte_slice_layer
                .unwrap_or(default_slice_layer)
                .min(layer_count - 1);
            let mut highlight_flags = 0u32;
            let mut highlight_hit_min = [0.0f32; 4];
            let mut highlight_hit_max = [0.0f32; 4];
            let mut highlight_face_axis = 0u32;
            let mut highlight_face_sign = 0i32;
            let highlight_mode_supported = matches!(
                render_options.vte_display_mode,
                VteDisplayMode::Integral | VteDisplayMode::Slice | VteDisplayMode::ThickSlice
            );
            if highlight_mode_supported {
                if let Some(hit_min) = render_options.vte_highlight_hit_min {
                    highlight_flags |= vte::VTE_HIGHLIGHT_FLAG_HIT_VOXEL;
                    highlight_hit_min = hit_min;
                    highlight_hit_max = render_options.vte_highlight_hit_max;
                    highlight_face_axis = render_options.vte_highlight_face_axis;
                    highlight_face_sign = render_options.vte_highlight_face_sign;
                }
            }
            // Reuse frame-meta padding words for fused-integral controls.
            // Values are consumed in shader via asfloat().
            let integral_sky_scale = if render_options.vte_integral_sky_emissive_tweak {
                render_options.vte_integral_sky_scale.max(0.0)
            } else {
                1.0
            };
            let integral_hit_emissive_boost = if render_options.vte_integral_sky_emissive_tweak {
                render_options.vte_integral_hit_emissive_boost.max(0.0)
            } else {
                0.0
            };
            let integral_log_merge_k = if render_options.vte_integral_log_merge_tweak {
                render_options.vte_integral_log_merge_k.max(0.0)
            } else {
                0.0
            };
            let max_trace_distance = render_options.vte_max_trace_distance.max(1.0);
            let debug_flags = {
                let mut flags = 0;
                if vte_diagnostics_feature_enabled() {
                    if render_options.vte_reference_compare {
                        flags |= vte::VTE_DEBUG_FLAG_REFERENCE_COMPARE;
                    }
                    if render_options.vte_reference_mismatch_only {
                        flags |= vte::VTE_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY;
                    }
                    if render_options.vte_compare_slice_only {
                        flags |= vte::VTE_DEBUG_FLAG_COMPARE_SLICE_ONLY;
                    }
                }
                if vte_lod_tint_enabled() {
                    flags |= vte::VTE_DEBUG_FLAG_LOD_TINT;
                }
                if vte_entity_linear_only_enabled() {
                    flags |= vte::VTE_DEBUG_FLAG_ENTITY_LINEAR_ONLY;
                }
                if vte_entity_bvh_compare_enabled() {
                    flags |= vte::VTE_DEBUG_FLAG_ENTITY_BVH_COMPARE;
                }
                if vte_world_bvh_ray_diag_active {
                    flags |= vte::VTE_DEBUG_FLAG_WORLD_BVH_RAY_DIAG;
                }
                if self.vte_stage_a_breakdown_enabled {
                    flags |= vte::VTE_DEBUG_FLAG_STAGE_A_BREAKDOWN;
                }
                flags
            };
            written_voxel_frame_meta = vte::GpuVoxelFrameMeta {
                chunk_count: vte_chunk_count as u32,
                leaf_count: vte_leaf_count as u32,
                occupancy_word_count: vte_occupancy_word_count as u32,
                material_word_count: vte_material_word_count as u32,
                macro_word_count: vte_macro_word_count as u32,
                max_trace_steps: render_options.vte_max_trace_steps.max(1),
                max_trace_distance,
                region_bvh_node_count: vte_region_bvh_node_count as u32,
                region_bvh_root_index: vte_region_bvh_root_index,
                leaf_chunk_entry_count: vte_leaf_chunk_entry_count as u32,
                stage_b_mode: render_options.vte_display_mode.as_u32(),
                stage_b_slice_layer,
                stage_b_thick_half_width: render_options.vte_thick_half_width,
                debug_flags,
                world_bvh_diag_seed: vte_world_bvh_ray_diag_seed,
                world_bvh_diag_sample_count: vte_world_bvh_ray_diag_sample_count,
                orientation_word_count: vte_orientation_word_count as u32,
                _world_bvh_diag_padding1: 0,
                visible_world_min: vte_visible_world_min,
                visible_world_max: vte_visible_world_max,
                highlight_flags,
                _highlight_padding: [
                    integral_sky_scale.to_bits(),
                    integral_hit_emissive_boost.to_bits(),
                    integral_log_merge_k.to_bits(),
                ],
                highlight_hit_min,
                highlight_face_axis,
                highlight_face_sign,
                _highlight_reserved: [0; 2],
                highlight_hit_max,
            };
            *writer = written_voxel_frame_meta;
        }

        if vte_world_bvh_ray_diag_active {
            if let Some(input) = voxel_input {
                let expected = build_world_bvh_ray_diag_expected_records(
                    &written_voxel_frame_meta,
                    self.sized_buffers.render_dimensions,
                    present_dimensions,
                    focal_length_xy,
                    focal_length_zw,
                    [
                        world_origin.x,
                        world_origin.y,
                        world_origin.z,
                        world_origin.w,
                    ],
                    [world_dir_x.x, world_dir_x.y, world_dir_x.z, world_dir_x.w],
                    [world_dir_y.x, world_dir_y.y, world_dir_y.z, world_dir_y.w],
                    [world_dir_z.x, world_dir_z.y, world_dir_z.z, world_dir_z.w],
                    [world_dir_w.x, world_dir_w.y, world_dir_w.z, world_dir_w.w],
                    &input.chunk_headers[..vte_chunk_count],
                    &input.occupancy_words[..vte_occupancy_word_count],
                    &input.material_words[..vte_material_word_count],
                    &input.leaf_headers[..vte_leaf_count],
                    &input.region_bvh_nodes[..vte_region_bvh_node_count],
                    &input.leaf_chunk_entries[..vte_leaf_chunk_entry_count],
                    &input.macro_words[..vte_macro_word_count],
                );
                self.frames_in_flight[frame_idx].vte_world_bvh_ray_diag_expected = expected;
            } else {
                self.frames_in_flight[frame_idx]
                    .vte_world_bvh_ray_diag_expected
                    .clear();
            }
        } else {
            self.frames_in_flight[frame_idx]
                .vte_world_bvh_ray_diag_expected
                .clear();
        }

        let mut line_render_count = 0;

        // Do compute stage

        // In order to draw, we have to record a *command buffer*. The command buffer
        // object holds the list of commands that are going to be executed.
        //
        // Recording a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to
        // be optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The
        // command buffer will only be executable on that given queue family.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let (image_index, acquire_future) = match self.swapchain.clone() {
            Some(swapchain) => {
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} acquire_next_image begin",
                        self.frames_rendered
                    );
                }
                let acquire_start = Instant::now();
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            self.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} acquire_next_image end ({:.2} ms)",
                        self.frames_rendered,
                        acquire_start.elapsed().as_secs_f64() * 1000.0
                    );
                }

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    self.recreate_swapchain = true;
                }

                (Some(image_index), Some(acquire_future))
            }
            None => (None, None),
        };

        let mut do_raster = render_options.do_raster;
        let mut do_raytrace = render_options.do_raytrace;
        let mut do_edges = render_options.do_edges;
        let mut do_tetrahedron_edges = render_options.do_tetrahedron_edges;
        let mut do_voxel_vte = false;
        let mut vte_compare_diagnostics_enabled = false;
        let logical_layers = self.sized_buffers.render_dimensions[2].max(1);
        let storage_layers = self.sized_buffers.pixel_storage_layers.max(1);

        match render_options.render_backend {
            RenderBackend::Auto => {}
            RenderBackend::TetraRaster => {
                do_raster = true;
                do_raytrace = false;
            }
            RenderBackend::TetraRaytrace => {
                do_raster = false;
                do_raytrace = true;
            }
            RenderBackend::VoxelTraversal => {
                // VTE resolves non-voxel tetrahedra directly in Stage A for depth correctness.
                // Optional post-raster overlays (e.g. held block preview) are rasterized from
                // a separate instance range to avoid double-rendering depth-tested entities.
                do_raster = raster_tetrahedron_count > 0;
                do_raytrace = false;
                do_edges = false;
                do_tetrahedron_edges = false;
                do_voxel_vte = true;
            }
        }

        let previous_vte_non_voxel_scene_hash = self.vte_non_voxel_scene_hash;
        let mut vte_non_voxel_rebuild_needed = false;
        let mut vte_non_voxel_rebuild_executed = false;
        let mut vte_non_voxel_rebuild_reason = "non_vte";
        let mut vte_non_voxel_bvh_update_mode = "non_vte";

        if do_voxel_vte {
            vte_non_voxel_rebuild_reason = "empty";
            vte_compare_diagnostics_enabled = vte_diagnostics_feature_enabled()
                && render_options.vte_reference_compare
                && matches!(
                    render_options.vte_display_mode,
                    VteDisplayMode::DebugCompare | VteDisplayMode::DebugIntegral
                );
            let vte_entity_bvh_compare = vte_entity_bvh_compare_enabled();
            let vte_stage_a_breakdown_active = self.vte_stage_a_breakdown_enabled;
            let vte_world_bvh_ray_diag_active =
                self.vte_world_bvh_ray_diag_enabled && voxel_input.is_some();
            if vte_compare_diagnostics_enabled
                || vte_entity_bvh_compare
                || vte_stage_a_breakdown_active
                || vte_world_bvh_ray_diag_active
            {
                self.reset_vte_compare_buffers(frame_idx);
            } else {
                self.clear_vte_compare_diagnostics();
            }
            let (
                candidate_chunks,
                visible_leaves,
                empty_chunks,
                full_chunks,
                visible_set_hash_valid,
                visible_set_hash,
            ) = if let Some(input) = voxel_input {
                let empty = 0u32;
                let full = input
                    .chunk_headers
                    .iter()
                    .filter(|h| (h.flags & GpuVoxelChunkHeader::FLAG_FULL) != 0)
                    .count() as u32;
                let collect_visible_set_hash =
                    vte_diagnostics_feature_enabled() && render_options.vte_reference_compare;
                let hash = if collect_visible_set_hash {
                    let mut hash = 0x811C_9DC5u32;
                    for node in input.region_bvh_nodes {
                        let h = vte::vte_hash_world_bounds(node.world_min)
                            ^ vte::vte_hash_world_bounds(node.world_max);
                        hash ^= h;
                        hash = hash.wrapping_mul(0x0100_0193);
                    }
                    hash
                } else {
                    0
                };
                (
                    input.chunk_headers.len() as u32,
                    input.leaf_headers.len() as u32,
                    empty,
                    full,
                    collect_visible_set_hash,
                    hash,
                )
            } else {
                (0, 0, 0, 0, false, 0)
            };

            self.vte_debug_counters = VteDebugCounters {
                candidate_chunks,
                frustum_culled_chunks: candidate_chunks.saturating_sub(visible_leaves),
                empty_chunks_skipped: empty_chunks,
                macro_cells_skipped: 0,
                chunk_steps: 0,
                voxel_steps: 0,
                primary_hits: full_chunks as u64,
                s_samples: self.sized_buffers.render_dimensions[0] as u64
                    * self.sized_buffers.render_dimensions[1] as u64
                    * self.sized_buffers.render_dimensions[2] as u64,
                visible_set_hash_valid,
                visible_set_hash,
            };
            if !self.vte_backend_notice_printed {
                println!(
                    "Render backend '{}' selected: VTE active (dense_chunks={}, leaves={}, region_bvh_nodes={}, leaf_entries={}, max_trace_steps={}, max_trace_distance={:.1}, stage_b={}, slice_layer={:?}, thick_half_width={}, reference_compare={}, mismatch_only={}, compare_slice_only={}, lod_tint={}, entity_linear_only={}, entity_bvh_compare={}, stagea_breakdown={}@{}, world_bvh_ray_diag={}@{}, storage_layers={}/{}).",
                    RenderBackend::VoxelTraversal.label(),
                    candidate_chunks,
                    visible_leaves,
                    vte_region_bvh_node_count,
                    vte_leaf_chunk_entry_count,
                    render_options.vte_max_trace_steps.max(1),
                    render_options.vte_max_trace_distance.max(1.0),
                    render_options.vte_display_mode.label(),
                    render_options.vte_slice_layer,
                    render_options.vte_thick_half_width,
                    render_options.vte_reference_compare,
                    render_options.vte_reference_mismatch_only,
                    render_options.vte_compare_slice_only,
                    vte_lod_tint_enabled(),
                    vte_entity_linear_only_enabled(),
                    vte_entity_bvh_compare_enabled(),
                    self.vte_stage_a_breakdown_enabled,
                    self.vte_stage_a_breakdown_interval,
                    self.vte_world_bvh_ray_diag_enabled,
                    self.vte_world_bvh_ray_diag_samples,
                    storage_layers,
                    logical_layers,
                );
                self.vte_backend_notice_printed = true;
            }
        } else {
            // Non-VTE passes reuse the same tetra/BVH buffers for other pipelines.
            // Force non-voxel reprovisioning when VTE is re-enabled.
            self.vte_non_voxel_scene_hash = 0;
            self.vte_non_voxel_bvh_topology_tet_count = 0;
            self.vte_non_voxel_bvh_refit_frames = 0;
            self.clear_vte_compare_diagnostics();
        }

        let reduced_storage_supported =
            do_voxel_vte && render_options.vte_display_mode == VteDisplayMode::Integral;
        self.profiler.record_scene_stats(
            do_voxel_vte,
            vte_chunk_count,
            vte_leaf_count,
            vte_visible_lod_counts,
            0,
            vte_buffer_caps.dense_chunks,
            vte_buffer_caps.leaf_headers,
            vte_buffer_caps.region_bvh_nodes,
            vte_buffer_caps.leaf_chunk_entries,
            raster_tetrahedron_count,
            total_tetrahedron_count,
        );
        if storage_layers < logical_layers && !reduced_storage_supported {
            panic!(
                "pixel storage layers ({storage_layers}) are less than logical render layers ({logical_layers}); \
this reduced-storage configuration currently supports only '--backend voxel-traversal --vte-display-mode integral'."
            );
        }

        self.frames_in_flight[frame_idx].vte_compare_enabled = do_voxel_vte
            && (vte_compare_diagnostics_enabled
                || vte_entity_bvh_compare_enabled()
                || self.vte_stage_a_breakdown_enabled);
        self.frames_in_flight[frame_idx].vte_world_bvh_ray_diag_enabled =
            do_voxel_vte && self.vte_world_bvh_ray_diag_enabled;

        self.last_backend = if do_voxel_vte {
            RenderBackend::VoxelTraversal
        } else if do_raytrace {
            RenderBackend::TetraRaytrace
        } else if do_raster || do_tetrahedron_edges || do_edges {
            RenderBackend::TetraRaster
        } else {
            RenderBackend::Auto
        };

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.pipeline_layout.clone(),
                0,
                vec![
                    self.one_time_buffers.descriptor_set.clone(),
                    self.frames_in_flight[frame_idx]
                        .sized_descriptor_set
                        .clone(),
                    self.frames_in_flight[frame_idx]
                        .live_buffers
                        .descriptor_set
                        .clone(),
                    self.texture_pool.descriptor_set().clone(),
                ],
            )
            .unwrap();

        // Set default push constants (required by pipeline layout even for shaders that don't use them)
        let dummy_push_data: [u32; 4] = [0, 0, 0, 0];
        builder
            .push_constants(
                self.compute_pipeline.pipeline_layout.clone(),
                0,
                dummy_push_data,
            )
            .unwrap();

        // GPU profiling: reset query pool and write start timestamp
        self.profiler.begin_frame();
        unsafe {
            builder.reset_query_pool(
                self.frames_in_flight[frame_idx].query_pool.clone(),
                0..PROFILER_MAX_TIMESTAMPS,
            )
        }
        .unwrap();
        {
            let q = self.profiler.next_query_index("start");
            unsafe {
                builder.write_timestamp(
                    self.frames_in_flight[frame_idx].query_pool.clone(),
                    q,
                    PipelineStage::AllCommands,
                )
            }
            .unwrap();
        }

        if do_voxel_vte {
            let non_voxel_leaf_count = non_voxel_bvh_leaf_count;
            if non_voxel_leaf_count == 0 {
                vte_non_voxel_rebuild_needed = false;
                vte_non_voxel_rebuild_reason = "empty";
                vte_non_voxel_bvh_update_mode = "empty";
                self.vte_non_voxel_scene_hash = 0;
                self.vte_non_voxel_bvh_topology_tet_count = 0;
                self.vte_non_voxel_bvh_refit_frames = 0;
            } else {
                // In overlay-raster mode, the shared tetra output buffer is reused by the
                // overlay tet preprocess pass later in the frame. Force non-voxel
                // reprovision every frame in that mode so Stage A never consumes stale
                // non-voxel tetra/BVH data on the next frame.
                vte_non_voxel_rebuild_needed =
                    do_raster || non_voxel_scene_hash != self.vte_non_voxel_scene_hash;
                vte_non_voxel_rebuild_reason = match (
                    do_raster,
                    non_voxel_scene_hash != self.vte_non_voxel_scene_hash,
                ) {
                    (true, true) => "overlay_raster+scene_hash",
                    (true, false) => "overlay_raster",
                    (false, true) => "scene_hash",
                    (false, false) => "unchanged",
                };
                if vte_non_voxel_rebuild_needed {
                    vte_non_voxel_rebuild_executed = true;
                    // Preprocess one proxy tetrahedron per non-voxel instance using the
                    // transformed [0,1]^4 instance AABB as the leaf primitive.
                    let vte_preprocess_push_data: [u32; 4] = [0, non_voxel_leaf_count as u32, 0, 0];
                    builder
                        .push_constants(
                            self.compute_pipeline.pipeline_layout.clone(),
                            0,
                            vte_preprocess_push_data,
                        )
                        .unwrap();
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline
                                .entity_instance_aabb_pre_pipeline
                                .clone(),
                        )
                        .unwrap();
                    unsafe {
                        builder.dispatch([(non_voxel_leaf_count as u32).div_ceil(64u32), 1, 1])
                    }
                    .unwrap();
                    {
                        let q = self.profiler.next_query_index("vte_non_voxel_preprocess");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    if non_voxel_leaf_count > vte::VTE_ENTITY_LINEAR_THRESHOLD_TETS {
                        let n = non_voxel_leaf_count as u32;
                        let topology_tet_count_matches =
                            self.vte_non_voxel_bvh_topology_tet_count == non_voxel_leaf_count;
                        let periodic_rebuild_due = self.vte_non_voxel_bvh_refit_frames
                            >= VTE_ENTITY_BVH_REFIT_REBUILD_INTERVAL;
                        let can_refit_only = topology_tet_count_matches && !periodic_rebuild_due;
                        if can_refit_only {
                            // Refit-only update for rapid movers: keep tree topology and
                            // update leaf/internal bounds from current tetrahedra.
                            let num_internal_nodes = n.saturating_sub(1);
                            if num_internal_nodes > 0 {
                                builder
                                    .bind_pipeline_compute(
                                        self.compute_pipeline.bvh_link_parents_pipeline.clone(),
                                    )
                                    .unwrap();
                                unsafe {
                                    builder.dispatch([num_internal_nodes.div_ceil(64), 1, 1])
                                }
                                .unwrap();
                            }
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_propagate_aabbs_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([n.div_ceil(64u32), 1, 1]) }.unwrap();
                            {
                                let q = self.profiler.next_query_index("vte_non_voxel_bvh_refit");
                                unsafe {
                                    builder.write_timestamp(
                                        self.frames_in_flight[frame_idx].query_pool.clone(),
                                        q,
                                        PipelineStage::AllCommands,
                                    )
                                }
                                .unwrap();
                            }
                            self.vte_non_voxel_bvh_refit_frames =
                                self.vte_non_voxel_bvh_refit_frames.saturating_add(1);
                            vte_non_voxel_bvh_update_mode = "refit";
                        } else {
                            // Full build path (first build, changed tetra count, or periodic
                            // topology refresh after repeated refits).
                            let n_pow2 = n.next_power_of_two();

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_scene_bounds_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_morton_codes_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([n_pow2.div_ceil(64u32), 1, 1]) }.unwrap();

                            let num_stages = n_pow2.trailing_zeros();
                            let local_stages = 6u32.min(num_stages);
                            let workgroups = n_pow2.div_ceil(64);

                            let push_data: [u32; 4] = [0, 0, n_pow2, 0];
                            builder
                                .push_constants(
                                    self.compute_pipeline.pipeline_layout.clone(),
                                    0,
                                    push_data,
                                )
                                .unwrap();
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline
                                        .bvh_bitonic_sort_local_pipeline
                                        .clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();

                            for stage in local_stages..num_stages {
                                builder
                                    .bind_pipeline_compute(
                                        self.compute_pipeline.bvh_bitonic_sort_pipeline.clone(),
                                    )
                                    .unwrap();
                                for step in (local_stages..=stage).rev() {
                                    let push_data: [u32; 4] = [stage, step, n_pow2, 0];
                                    builder
                                        .push_constants(
                                            self.compute_pipeline.pipeline_layout.clone(),
                                            0,
                                            push_data,
                                        )
                                        .unwrap();
                                    unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();
                                }

                                let push_data: [u32; 4] = [stage, 0, n_pow2, 0];
                                builder
                                    .push_constants(
                                        self.compute_pipeline.pipeline_layout.clone(),
                                        0,
                                        push_data,
                                    )
                                    .unwrap();
                                builder
                                    .bind_pipeline_compute(
                                        self.compute_pipeline
                                            .bvh_bitonic_sort_local_merge_pipeline
                                            .clone(),
                                    )
                                    .unwrap();
                                unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();
                            }

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_init_leaves_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([n.div_ceil(64u32), 1, 1]) }.unwrap();

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_build_tree_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([n.div_ceil(64u32), 1, 1]) }.unwrap();

                            let num_internal_nodes = n.saturating_sub(1);
                            if num_internal_nodes > 0 {
                                builder
                                    .bind_pipeline_compute(
                                        self.compute_pipeline.bvh_link_parents_pipeline.clone(),
                                    )
                                    .unwrap();
                                unsafe {
                                    builder.dispatch([num_internal_nodes.div_ceil(64), 1, 1])
                                }
                                .unwrap();
                            }
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_propagate_aabbs_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([n.div_ceil(64u32), 1, 1]) }.unwrap();

                            {
                                let q = self.profiler.next_query_index("vte_non_voxel_bvh");
                                unsafe {
                                    builder.write_timestamp(
                                        self.frames_in_flight[frame_idx].query_pool.clone(),
                                        q,
                                        PipelineStage::AllCommands,
                                    )
                                }
                                .unwrap();
                            }
                            self.vte_non_voxel_bvh_topology_tet_count = non_voxel_leaf_count;
                            self.vte_non_voxel_bvh_refit_frames = 0;
                            vte_non_voxel_bvh_update_mode =
                                if periodic_rebuild_due && topology_tet_count_matches {
                                    "rebuild_interval"
                                } else {
                                    "rebuild"
                                };
                        }
                    } else {
                        self.vte_non_voxel_bvh_topology_tet_count = 0;
                        self.vte_non_voxel_bvh_refit_frames = 0;
                        vte_non_voxel_bvh_update_mode = "linear";
                    }

                    self.vte_non_voxel_scene_hash = non_voxel_scene_hash;
                } else {
                    vte_non_voxel_bvh_update_mode = "reuse";
                }
            }

            let fuse_integral_in_stage_a =
                render_options.vte_display_mode == VteDisplayMode::Integral;
            {
                let q = self.profiler.next_query_index("vte_stage_a_setup");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }
            let voxel_trace_stage_a_pipeline = if fuse_integral_in_stage_a {
                self.compute_pipeline
                    .voxel_trace_stage_a_integral_fused_pipeline
                    .clone()
            } else {
                self.compute_pipeline
                    .voxel_trace_stage_a_layered_pipeline
                    .clone()
            };
            builder
                .bind_pipeline_compute(voxel_trace_stage_a_pipeline)
                .unwrap();
            unsafe {
                builder.dispatch([
                    self.sized_buffers.render_dimensions[0].div_ceil(8),
                    self.sized_buffers.render_dimensions[1].div_ceil(8),
                    if fuse_integral_in_stage_a {
                        1
                    } else {
                        self.sized_buffers.render_dimensions[2].max(1)
                    },
                ])
            }
            .unwrap();
            {
                let q = self.profiler.next_query_index("vte_stage_a");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }

            if !fuse_integral_in_stage_a {
                {
                    let q = self.profiler.next_query_index("vte_stage_b_setup");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }
                builder
                    .bind_pipeline_compute(
                        self.compute_pipeline.voxel_display_stage_b_pipeline.clone(),
                    )
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        self.sized_buffers.render_dimensions[0].div_ceil(8),
                        self.sized_buffers.render_dimensions[1].div_ceil(8),
                        1,
                    ])
                }
                .unwrap();
            }
            {
                let q = self.profiler.next_query_index("vte_stage_b");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }
        } else if (render_options.do_frame_clear) && !do_raytrace {
            builder
                .bind_pipeline_compute(self.compute_pipeline.raytrace_clear_pipeline.clone())
                .unwrap();
            unsafe {
                builder.dispatch([
                    self.sized_buffers.render_dimensions[0].div_ceil(8),
                    self.sized_buffers.render_dimensions[1].div_ceil(8),
                    1,
                ])
            }
            .unwrap();
            let q = self.profiler.next_query_index("clear");
            unsafe {
                builder.write_timestamp(
                    self.frames_in_flight[frame_idx].query_pool.clone(),
                    q,
                    PipelineStage::AllCommands,
                )
            }
            .unwrap();
        }

        if do_raster || do_tetrahedron_edges {
            let raster_preprocess_tetrahedron_count = if do_voxel_vte {
                raster_tetrahedron_count
            } else {
                total_tetrahedron_count
            };
            let raster_preprocess_push_data: [u32; 4] = [
                raster_instance_base as u32,
                raster_preprocess_tetrahedron_count as u32,
                0,
                0,
            ];

            // Tetrahedron pre-raster
            // Reset atomic counter to 0 before clipping dispatch
            builder
                .fill_buffer(self.sized_buffers.atomic_counter_buffer.clone(), 0u32)
                .unwrap();
            {
                let q = self.profiler.next_query_index("tet_counter_clear");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }

            builder
                .push_constants(
                    self.compute_pipeline.pipeline_layout.clone(),
                    0,
                    raster_preprocess_push_data,
                )
                .unwrap();
            builder
                .bind_pipeline_compute(self.compute_pipeline.tetrahedron_pipeline.clone())
                .unwrap();
            unsafe {
                builder.dispatch([
                    (raster_preprocess_tetrahedron_count as u32).div_ceil(64u32),
                    1,
                    1,
                ])
            }
            .unwrap();
            {
                let q = self.profiler.next_query_index("tet_clip");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }

            // Copy atomic counter for CPU readback (clipped tet count diagnostic)
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.sized_buffers.atomic_counter_buffer.clone(),
                    self.frames_in_flight[frame_idx]
                        .cpu_clipped_tet_count_buffer
                        .clone(),
                ))
                .unwrap();
            {
                let q = self.profiler.next_query_index("tet_counter_copy");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }

            if do_tetrahedron_edges {
                line_render_count = raster_preprocess_tetrahedron_count * 6;
            }
        }

        if do_raster {
            if !do_voxel_vte {
                // Zero tile counts
                builder
                    .fill_buffer(self.sized_buffers.tile_tet_counts_buffer.clone(), 0u32)
                    .unwrap();
                {
                    let q = self.profiler.next_query_index("tet_bin_clear");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                // Bin tetrahedra into tiles
                builder
                    .bind_pipeline_compute(self.compute_pipeline.bin_tets_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        (self.sized_buffers.max_tetrahedrons as u32).div_ceil(64),
                        1,
                        1,
                    ])
                }
                .unwrap();
                {
                    let q = self.profiler.next_query_index("tet_bin");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }
            }

            // Tetrahedron pixel raster (tile-based)
            builder
                .bind_pipeline_compute(self.compute_pipeline.tetrahedron_pixel_pipeline.clone())
                .unwrap();
            let (raster_dispatch_push, raster_dispatch_dims) = if do_voxel_vte {
                let region = self.vte_overlay_raster_region();
                let overlay_work_w = region[2].div_ceil(VTE_OVERLAY_RASTER_SCALE);
                let overlay_work_h = region[3].div_ceil(VTE_OVERLAY_RASTER_SCALE);
                if self.frames_rendered == 0 {
                    println!(
                        "VTE overlay raster region: origin=({}, {}) size={}x{} (work {}x{} @{}x upsample)",
                        region[0], region[1], region[2], region[3], overlay_work_w, overlay_work_h, VTE_OVERLAY_RASTER_SCALE
                    );
                }
                (
                    region,
                    [overlay_work_w.div_ceil(8), overlay_work_h.div_ceil(8), 1],
                )
            } else {
                (
                    [0, 0, 0, 0],
                    [
                        self.sized_buffers.render_dimensions[0].div_ceil(8),
                        self.sized_buffers.render_dimensions[1].div_ceil(8),
                        1,
                    ],
                )
            };
            builder
                .push_constants(
                    self.compute_pipeline.pipeline_layout.clone(),
                    0,
                    raster_dispatch_push,
                )
                .unwrap();
            unsafe { builder.dispatch(raster_dispatch_dims) }.unwrap();
            {
                let q = self.profiler.next_query_index("tet_raster");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }
        }

        if do_edges || custom_overlay_edge_used_instance_count > 0 {
            // Edge pre-raster supports dedicated instance ranges (for debug edges and
            // GPU-only custom overlay boxes) and appends into the shared line buffer.
            let model_edge_count = self.one_time_buffers.model_edge_count;
            let max_lines = LINE_VERTEX_CAPACITY / 2;
            let mut dispatched_any_edges = false;

            builder
                .bind_pipeline_compute(self.compute_pipeline.edge_pipeline.clone())
                .unwrap();

            if model_edge_count > 0 {
                if do_edges {
                    let available_lines = max_lines.saturating_sub(line_render_count);
                    let max_instances = available_lines / model_edge_count;
                    let edge_instance_count = used_instance_count.min(max_instances);
                    if edge_instance_count > 0 {
                        let edge_line_count = edge_instance_count * model_edge_count;
                        let edge_push_data: [u32; 4] = [
                            0,
                            edge_instance_count.min(u32::MAX as usize) as u32,
                            line_render_count.min(u32::MAX as usize) as u32,
                            0,
                        ];
                        builder
                            .push_constants(
                                self.compute_pipeline.pipeline_layout.clone(),
                                0,
                                edge_push_data,
                            )
                            .unwrap();
                        unsafe {
                            builder.dispatch([(edge_line_count as u32).div_ceil(64u32), 1, 1])
                        }
                        .unwrap();
                        line_render_count += edge_line_count;
                        dispatched_any_edges = true;
                    }
                }

                if custom_overlay_edge_used_instance_count > 0 {
                    let available_lines = max_lines.saturating_sub(line_render_count);
                    let max_instances = available_lines / model_edge_count;
                    let edge_overlay_instance_count =
                        custom_overlay_edge_used_instance_count.min(max_instances);
                    if edge_overlay_instance_count > 0 {
                        let edge_line_count = edge_overlay_instance_count * model_edge_count;
                        let edge_push_data: [u32; 4] = [
                            custom_overlay_edge_instance_base.min(u32::MAX as usize) as u32,
                            edge_overlay_instance_count.min(u32::MAX as usize) as u32,
                            line_render_count.min(u32::MAX as usize) as u32,
                            0,
                        ];
                        builder
                            .push_constants(
                                self.compute_pipeline.pipeline_layout.clone(),
                                0,
                                edge_push_data,
                            )
                            .unwrap();
                        unsafe {
                            builder.dispatch([(edge_line_count as u32).div_ceil(64u32), 1, 1])
                        }
                        .unwrap();
                        line_render_count += edge_line_count;
                        dispatched_any_edges = true;
                    }
                }
            }

            if dispatched_any_edges {
                let q = self.profiler.next_query_index("edges");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }
        }

        if do_raytrace {
            // Compute a hash of the scene inputs that affect the BVH.
            // If unchanged, skip tetrahedron preprocessing and BVH construction.
            let scene_hash = {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                for &val in view_matrix_nalgebra.as_slice() {
                    val.to_bits().hash(&mut hasher);
                }
                focal_length_xy.to_bits().hash(&mut hasher);
                focal_length_zw.to_bits().hash(&mut hasher);
                model_instances.len().hash(&mut hasher);
                bytemuck::cast_slice::<_, u8>(&model_instances).hash(&mut hasher);
                hasher.finish()
            };
            let bvh_needs_rebuild = scene_hash != self.bvh_scene_hash;
            let raytrace_should_clear = render_options.do_frame_clear || bvh_needs_rebuild;

            if raytrace_should_clear {
                builder
                    .bind_pipeline_compute(self.compute_pipeline.raytrace_clear_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        self.sized_buffers.render_dimensions[0].div_ceil(8),
                        self.sized_buffers.render_dimensions[1].div_ceil(8),
                        1,
                    ])
                }
                .unwrap();
                let q = self.profiler.next_query_index("clear");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }

            if bvh_needs_rebuild {
                // 1. Tetrahedron preprocessing (transform to view space)
                let raytrace_preprocess_push_data: [u32; 4] =
                    [0, total_tetrahedron_count as u32, 0, 0];
                builder
                    .push_constants(
                        self.compute_pipeline.pipeline_layout.clone(),
                        0,
                        raytrace_preprocess_push_data,
                    )
                    .unwrap();
                builder
                    .bind_pipeline_compute(self.compute_pipeline.raytrace_pre_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([(total_tetrahedron_count as u32).div_ceil(64u32), 1, 1])
                }
                .unwrap();
                {
                    let q = self.profiler.next_query_index("rt_preprocess");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                // 2. BVH Construction
                if total_tetrahedron_count > 0 {
                    // 2a. Compute scene bounds
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline.bvh_scene_bounds_pipeline.clone(),
                        )
                        .unwrap();
                    unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

                    // 2b. Compute Morton codes (dispatch n_pow2 threads to fill sentinels for padding)
                    let n = total_tetrahedron_count as u32;
                    let n_pow2 = n.next_power_of_two();
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline.bvh_morton_codes_pipeline.clone(),
                        )
                        .unwrap();
                    unsafe { builder.dispatch([n_pow2.div_ceil(64u32), 1, 1]) }.unwrap();

                    // 2c. Bitonic sort using shared memory optimization
                    // Sort all n_pow2 elements (including sentinel-padded entries)
                    let num_stages = n_pow2.trailing_zeros(); // log2(n_pow2)
                    let local_stages = 6u32.min(num_stages); // stages 0-5 fit in 64-element workgroups
                    let workgroups = n_pow2.div_ceil(64);

                    // Phase 1: Sort each 64-element block in shared memory (stages 0-5)
                    let push_data: [u32; 4] = [0, 0, n_pow2, 0];
                    builder
                        .push_constants(self.compute_pipeline.pipeline_layout.clone(), 0, push_data)
                        .unwrap();
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline
                                .bvh_bitonic_sort_local_pipeline
                                .clone(),
                        )
                        .unwrap();
                    unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();

                    // Phase 2: Global merge stages (stages 6+)
                    for stage in local_stages..num_stages {
                        // Global steps: stepSize >= 64 (step >= 6)
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.bvh_bitonic_sort_pipeline.clone(),
                            )
                            .unwrap();
                        for step in (local_stages..=stage).rev() {
                            let push_data: [u32; 4] = [stage, step, n_pow2, 0];
                            builder
                                .push_constants(
                                    self.compute_pipeline.pipeline_layout.clone(),
                                    0,
                                    push_data,
                                )
                                .unwrap();
                            unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();
                        }
                        // Local merge: steps 5-0 in shared memory (1 dispatch)
                        let push_data: [u32; 4] = [stage, 0, n_pow2, 0];
                        builder
                            .push_constants(
                                self.compute_pipeline.pipeline_layout.clone(),
                                0,
                                push_data,
                            )
                            .unwrap();
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline
                                    .bvh_bitonic_sort_local_merge_pipeline
                                    .clone(),
                            )
                            .unwrap();
                        unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();
                    }

                    // 2d. Initialize leaf nodes
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline.bvh_init_leaves_pipeline.clone(),
                        )
                        .unwrap();
                    unsafe {
                        builder.dispatch([(total_tetrahedron_count as u32).div_ceil(64u32), 1, 1])
                    }
                    .unwrap();

                    // 2e. Build internal nodes (Karras algorithm)
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline.bvh_build_tree_pipeline.clone(),
                        )
                        .unwrap();
                    unsafe {
                        builder.dispatch([(total_tetrahedron_count as u32).div_ceil(64u32), 1, 1])
                    }
                    .unwrap();

                    // 2f. Link parent pointers for leaf-to-root propagation.
                    let num_internal_nodes = total_tetrahedron_count.saturating_sub(1) as u32;
                    if num_internal_nodes > 0 {
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.bvh_link_parents_pipeline.clone(),
                            )
                            .unwrap();
                        unsafe { builder.dispatch([num_internal_nodes.div_ceil(64), 1, 1]) }
                            .unwrap();
                    }

                    // 2g. Compute leaf AABBs and propagate all parent bounds in one pass.
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline.bvh_propagate_aabbs_pipeline.clone(),
                        )
                        .unwrap();
                    unsafe {
                        builder.dispatch([(total_tetrahedron_count as u32).div_ceil(64u32), 1, 1])
                    }
                    .unwrap();
                }

                self.bvh_scene_hash = scene_hash;

                // Debug: copy BVH data to CPU on first rebuild
                if self.frames_rendered == 0 {
                    builder
                        .copy_buffer(CopyBufferInfo::buffers(
                            self.sized_buffers.bvh_nodes_buffer.clone(),
                            self.sized_buffers.cpu_bvh_nodes_buffer.clone(),
                        ))
                        .unwrap();
                    builder
                        .copy_buffer(CopyBufferInfo::buffers(
                            self.sized_buffers.morton_codes_buffer.clone(),
                            self.sized_buffers.cpu_morton_codes_buffer.clone(),
                        ))
                        .unwrap();
                }

                {
                    let q = self.profiler.next_query_index("bvh_build");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }
            }

            // 3. Raytrace pixels (using BVH) - always runs, only seed changes
            builder
                .bind_pipeline_compute(self.compute_pipeline.raytrace_pixel_pipeline.clone())
                .unwrap();
            unsafe {
                builder.dispatch([
                    self.sized_buffers.render_dimensions[0].div_ceil(8),
                    self.sized_buffers.render_dimensions[1].div_ceil(8),
                    1,
                ])
            }
            .unwrap();
            {
                let q = self.profiler.next_query_index("raytrace");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }
        }

        let prev_used_non_voxel = self.vte_entity_diag_prev_used_non_voxel;
        let prev_tets_non_voxel = self.vte_entity_diag_prev_tets_non_voxel;
        let used_non_voxel_went_zero = do_voxel_vte
            && prev_used_non_voxel
                .map(|prev| prev > 0 && non_voxel_used_instance_count == 0)
                .unwrap_or(false);
        let tets_non_voxel_went_zero = do_voxel_vte
            && prev_tets_non_voxel
                .map(|prev| prev > 0 && non_voxel_bvh_leaf_count == 0)
                .unwrap_or(false);
        if do_voxel_vte {
            self.vte_entity_diag_prev_used_non_voxel = Some(non_voxel_used_instance_count);
            self.vte_entity_diag_prev_tets_non_voxel = Some(non_voxel_bvh_leaf_count);
        } else {
            self.vte_entity_diag_prev_used_non_voxel = None;
            self.vte_entity_diag_prev_tets_non_voxel = None;
        }

        let vte_entity_diag_anomaly = do_voxel_vte
            && (dropped_model_instance_count > 0
                || non_voxel_outlier_count > 0
                || used_non_voxel_went_zero
                || tets_non_voxel_went_zero
                || (!model_instances_input.is_empty() && non_voxel_used_instance_count == 0)
                || (non_voxel_used_instance_count > 0 && non_voxel_bvh_leaf_count == 0));
        let vte_entity_diag_periodic_due = self
            .vte_entity_diag_last_log_frame
            .map(|last| {
                self.frames_rendered.saturating_sub(last) >= self.vte_entity_diag_interval.max(1)
            })
            .unwrap_or(true);
        let vte_entity_diag_copy_requested = self.vte_entity_diag_enabled
            && self.vte_entity_diag_bvh_readback
            && do_voxel_vte
            && non_voxel_bvh_leaf_count > 0
            && (self.vte_entity_diag_verbose
                || vte_entity_diag_anomaly
                || self
                    .frames_rendered
                    .is_multiple_of(self.vte_entity_diag_interval.max(1)));
        self.frames_in_flight[frame_idx].vte_entity_diag_copy_scheduled =
            vte_entity_diag_copy_requested;
        self.frames_in_flight[frame_idx].vte_entity_diag_non_voxel_tet_count = if do_voxel_vte {
            non_voxel_bvh_leaf_count
        } else {
            0
        };
        if vte_entity_diag_copy_requested {
            if self.vte_entity_diag_bvh_topology {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        self.sized_buffers.bvh_nodes_buffer.clone(),
                        self.sized_buffers.cpu_bvh_nodes_buffer.clone(),
                    ))
                    .unwrap();
            } else {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        self.sized_buffers.bvh_nodes_buffer.clone(),
                        self.sized_buffers.cpu_bvh_root_buffer.clone(),
                    ))
                    .unwrap();
            }
        }

        if self.vte_entity_diag_enabled
            && (self.vte_entity_diag_verbose
                || vte_entity_diag_anomaly
                || (do_voxel_vte && vte_entity_diag_periodic_due))
        {
            eprintln!(
                "[vte-entity-diag] frame={} backend={} mode={} input_non_voxel={} input_overlay={} used_non_voxel={} dropped_non_finite={} tets_non_voxel={} tets_overlay={} do_raster={} prev_hash=0x{:016x} hash=0x{:016x} rebuild_needed={} rebuild_executed={} rebuild_reason={} bvh_mode={} bvh_refit_frames={} bvh_topology_tets={} max_abs_translation={:.2} max_abs_basis={:.2} outlier_instances={}",
                self.frames_rendered,
                self.last_backend.label(),
                render_options.vte_display_mode.label(),
                model_instances_input.len(),
                raster_overlay_instances_input.len(),
                non_voxel_used_instance_count,
                dropped_model_instance_count,
                non_voxel_bvh_leaf_count,
                raster_overlay_tetrahedron_count,
                do_raster,
                previous_vte_non_voxel_scene_hash,
                non_voxel_scene_hash,
                vte_non_voxel_rebuild_needed,
                vte_non_voxel_rebuild_executed,
                vte_non_voxel_rebuild_reason,
                vte_non_voxel_bvh_update_mode,
                self.vte_non_voxel_bvh_refit_frames,
                self.vte_non_voxel_bvh_topology_tet_count,
                non_voxel_translation_abs_max,
                non_voxel_basis_abs_max,
                non_voxel_outlier_count
            );
            if self.vte_compare_stats.entity_bvh_samples > 0 {
                eprintln!(
                    "[vte-entity-diag][bvh-compare] frame={} samples={} mismatches={} hit_state={} material={} distance={} tetra={} bvh_miss_linear_hit={} bvh_hit_linear_miss={} noprune_mismatches={} noprune_hit_state={} noprune_distance={} noprune_tetra={} noaabb_mismatches={} noaabb_hit_state={} noaabb_distance={} noaabb_tetra={} linear_order_mismatches={} linear_order_hit_state={} linear_order_distance={} linear_order_tetra={} leafarray_mismatches={} leafarray_hit_state={} leafarray_distance={} leafarray_tetra={}",
                    self.frames_rendered,
                    self.vte_compare_stats.entity_bvh_samples,
                    self.vte_compare_stats.entity_bvh_mismatches,
                    self.vte_compare_stats.entity_bvh_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_material_mismatches,
                    self.vte_compare_stats.entity_bvh_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_tetra_mismatches,
                    self.vte_compare_stats.entity_bvh_miss_linear_hit,
                    self.vte_compare_stats.entity_bvh_hit_linear_miss,
                    self.vte_compare_stats.entity_bvh_noprune_mismatches,
                    self.vte_compare_stats.entity_bvh_noprune_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_noprune_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_noprune_tetra_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_tetra_mismatches,
                    self.vte_compare_stats.entity_linear_order_mismatches,
                    self.vte_compare_stats.entity_linear_order_hit_state_mismatches,
                    self.vte_compare_stats.entity_linear_order_distance_mismatches,
                    self.vte_compare_stats.entity_linear_order_tetra_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_tetra_mismatches,
                );
            }
            self.vte_entity_diag_last_log_frame = Some(self.frames_rendered);
        }
        if self.vte_stage_a_breakdown_enabled && do_voxel_vte {
            let periodic_due = self
                .vte_stage_a_breakdown_last_log_frame
                .map(|last| {
                    self.frames_rendered.saturating_sub(last)
                        >= self.vte_stage_a_breakdown_interval.max(1)
                })
                .unwrap_or(true);
            let samples = self.vte_compare_stats.stagea_samples;
            if periodic_due && samples > 0 {
                let s = samples as f64;
                let entity_queries = self.vte_compare_stats.stagea_entity_queries as f64;
                let entity_hits = self.vte_compare_stats.stagea_entity_hits as f64;
                let voxel_hits = self.vte_compare_stats.stagea_voxel_hits as f64;
                let sky_misses = self.vte_compare_stats.stagea_sky_misses as f64;
                let avg_chunk_steps = self.vte_compare_stats.stagea_chunk_steps_sum as f64 / s;
                let avg_voxel_steps = self.vte_compare_stats.stagea_voxel_steps_sum as f64 / s;
                let avg_node_visits = self.vte_compare_stats.stagea_node_visits_sum as f64 / s;
                eprintln!(
                    "[vte-stagea-breakdown] frame={} samples={} entity_queries={} ({:.1}%) entity_hits={} ({:.1}%) voxel_hits={} ({:.1}%) sky_misses={} ({:.1}%) avg_chunk_steps={:.2} avg_voxel_steps={:.2} avg_node_visits={:.2}",
                    self.frames_rendered,
                    samples,
                    self.vte_compare_stats.stagea_entity_queries,
                    (entity_queries * 100.0) / s,
                    self.vte_compare_stats.stagea_entity_hits,
                    (entity_hits * 100.0) / s,
                    self.vte_compare_stats.stagea_voxel_hits,
                    (voxel_hits * 100.0) / s,
                    self.vte_compare_stats.stagea_sky_misses,
                    (sky_misses * 100.0) / s,
                    avg_chunk_steps,
                    avg_voxel_steps,
                    avg_node_visits,
                );
                self.vte_stage_a_breakdown_last_log_frame = Some(self.frames_rendered);
            }
        }
        if self.vte_entity_diag_enabled && (used_non_voxel_went_zero || tets_non_voxel_went_zero) {
            eprintln!(
                "[vte-entity-diag][transition-zero] frame={} backend={} mode={} used_non_voxel:{}->{} tets_non_voxel:{}->{} input_non_voxel={} input_overlay={} dropped_non_finite={} do_raster={} prev_hash=0x{:016x} hash=0x{:016x} rebuild_needed={} rebuild_executed={} rebuild_reason={} bvh_mode={} bvh_refit_frames={} bvh_topology_tets={}",
                self.frames_rendered,
                self.last_backend.label(),
                render_options.vte_display_mode.label(),
                prev_used_non_voxel.unwrap_or(0),
                non_voxel_used_instance_count,
                prev_tets_non_voxel.unwrap_or(0),
                non_voxel_bvh_leaf_count,
                model_instances_input.len(),
                raster_overlay_instances_input.len(),
                dropped_model_instance_count,
                do_raster,
                previous_vte_non_voxel_scene_hash,
                non_voxel_scene_hash,
                vte_non_voxel_rebuild_needed,
                vte_non_voxel_rebuild_executed,
                vte_non_voxel_rebuild_reason,
                vte_non_voxel_bvh_update_mode,
                self.vte_non_voxel_bvh_refit_frames,
                self.vte_non_voxel_bvh_topology_tet_count
            );
        }

        let mut hud_vertex_count = 0usize;
        let mut hud_batches: Vec<HudDrawBatch> = Vec::new();
        line_render_count += self.write_custom_overlay_lines(
            frame_idx,
            line_render_count,
            &render_options.custom_overlay_lines,
        );
        if render_options.do_navigation_hud {
            let (hud_line_count, hud_quad_count) = self.write_navigation_hud_overlay(
                frame_idx,
                line_render_count,
                &view_matrix_nalgebra,
                &view_matrix_nalgebra_inv,
                focal_length_xy,
                &model_instances,
                render_options.hud_readout_mode,
                render_options.hud_rotation_label.as_deref(),
                render_options.hud_target_hit_voxel,
                render_options.hud_target_hit_face,
                &render_options.hud_player_tags,
            );
            line_render_count += hud_line_count;
            hud_vertex_count = hud_quad_count;
            if hud_quad_count > 0 {
                hud_batches.push(HudDrawBatch {
                    first_vertex: 0,
                    vertex_count: hud_quad_count as u32,
                    scissor: self.full_hud_scissor(),
                    texture_slot: HudTextureSlot::Hud,
                });
            }
        }

        if let Some(egui_paint) = render_options.egui_paint.as_ref() {
            if !egui_paint.texture_updates.is_empty() {
                self.apply_egui_texture_updates(queue.clone(), &egui_paint.texture_updates);
            }
            let (egui_vertex_count, mut egui_batches) =
                self.write_egui_overlay(frame_idx, hud_vertex_count, &egui_paint.meshes);
            hud_vertex_count += egui_vertex_count;
            hud_batches.append(&mut egui_batches);
        }

        let aetna_draw_ready =
            if let (Some(tree), Some(aetna)) = (aetna_ui.as_mut(), self.aetna_overlay.as_mut()) {
                let (present_size, scale_factor) = match self.window.as_ref() {
                    Some(window) => {
                        let size = window.inner_size();
                        (
                            [size.width.max(1), size.height.max(1)],
                            window.scale_factor() as f32,
                        )
                    }
                    None => (
                        [
                            self.sized_buffers.render_dimensions[0].max(1),
                            self.sized_buffers.render_dimensions[1].max(1),
                        ],
                        1.0,
                    ),
                };
                aetna
                    .runner
                    .set_surface_size(present_size[0], present_size[1]);
                aetna
                    .runner
                    .set_theme(aetna_core::Theme::radix_slate_blue_dark());
                let viewport = aetna_core::Rect::new(
                    0.0,
                    0.0,
                    present_size[0] as f32 / scale_factor,
                    present_size[1] as f32 / scale_factor,
                );
                let _ = aetna.runner.prepare(tree, viewport, scale_factor);
                true
            } else {
                false
            };

        if let Some(image_index) = image_index {
            if let Some(framebuffers) = self.framebuffers.clone() {
                if let Some(present_pipeline) = self.present_pipeline.as_ref() {
                    // Begin render pass
                    builder
                        // Before we can draw, we have to *enter a render pass*.
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                // A list of values to clear the attachments with. This list contains
                                // one item for each attachment in the render pass. In this case, there
                                // is only one attachment, and we clear it with a blue color.
                                //
                                // Only attachments that have `AttachmentLoadOp::Clear` are provided
                                // with clear values, any others should use `None` as the clear value.
                                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],

                                ..RenderPassBeginInfo::framebuffer(
                                    framebuffers[image_index as usize].clone(),
                                )
                            },
                            SubpassBeginInfo {
                                // The contents of the first (and only) subpass. This can be either
                                // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
                                // advanced and is not covered here.
                                contents: SubpassContents::Inline,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    builder
                        .set_viewport(0, [self.viewport.clone()].into_iter().collect())
                        .unwrap();
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            present_pipeline.pipeline_layout.clone(),
                            0,
                            vec![
                                self.one_time_buffers.descriptor_set.clone(),
                                self.frames_in_flight[frame_idx]
                                    .sized_descriptor_set
                                    .clone(),
                                self.frames_in_flight[frame_idx]
                                    .live_buffers
                                    .descriptor_set
                                    .clone(),
                                self.texture_pool.descriptor_set().clone(),
                            ],
                        )
                        .unwrap();
                    {
                        let q = self.profiler.next_query_index("present_begin");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // Render from compute shader buffer
                    {
                        builder
                            .bind_pipeline_graphics(present_pipeline.buffer_pipeline.clone())
                            .unwrap();
                        unsafe { builder.draw(6, 1, 0, 0) }.unwrap();
                    }
                    {
                        let q = self.profiler.next_query_index("present_buffer");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // Render the edge lines
                    if line_render_count > 0 {
                        builder
                            .bind_pipeline_graphics(present_pipeline.line_pipeline.clone())
                            .unwrap();
                        unsafe { builder.draw(line_render_count as u32 * 2, 1, 0, 0) }.unwrap();
                    }
                    {
                        let q = self.profiler.next_query_index("present_lines");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // Render HUD quads (text + panels, alpha-blended on top)
                    if hud_vertex_count > 0 {
                        builder
                            .bind_pipeline_graphics(present_pipeline.hud_pipeline.clone())
                            .unwrap();

                        let mut bound_texture_slot: Option<HudTextureSlot> = None;
                        for batch in &hud_batches {
                            let frame = &self.frames_in_flight[frame_idx];
                            let descriptor_set = match batch.texture_slot {
                                HudTextureSlot::Hud => frame.hud_descriptor_set.as_ref(),
                                HudTextureSlot::EguiAtlas => frame.egui_descriptor_set.as_ref(),
                                HudTextureSlot::MaterialIcons => frame
                                    .material_icons_descriptor_set
                                    .as_ref()
                                    .or(frame.egui_descriptor_set.as_ref()),
                            };
                            let Some(descriptor_set) = descriptor_set else {
                                continue;
                            };

                            if bound_texture_slot != Some(batch.texture_slot) {
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Graphics,
                                        present_pipeline.hud_pipeline_layout.clone(),
                                        0,
                                        vec![descriptor_set.clone()],
                                    )
                                    .unwrap();
                                bound_texture_slot = Some(batch.texture_slot);
                            }

                            builder
                                .set_scissor(0, [batch.scissor].into_iter().collect())
                                .unwrap();
                            builder
                                .push_constants(
                                    present_pipeline.hud_pipeline_layout.clone(),
                                    0,
                                    [batch.first_vertex],
                                )
                                .unwrap();
                            unsafe { builder.draw(batch.vertex_count, 1, 0, 0) }.unwrap();
                        }
                    }
                    if aetna_draw_ready {
                        if let Some(aetna) = self.aetna_overlay.as_ref() {
                            aetna.runner.draw(&mut builder);
                        }
                    }
                    {
                        let q = self.profiler.next_query_index("present_hud");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // End render pass
                    builder
                        // We leave the render pass. Note that if we had multiple subpasses we could
                        // have called `next_subpass` to jump to the next subpass.
                        .end_render_pass(Default::default())
                        .unwrap();
                    {
                        let q = self.profiler.next_query_index("present_end");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }
                    if render_options.take_framebuffer_screenshot {
                        builder
                            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                                framebuffers[image_index as usize].attachments()[0]
                                    .image()
                                    .clone(),
                                self.cpu_screen_capture_buffer.clone(),
                            ))
                            .unwrap();
                        {
                            let q = self.profiler.next_query_index("present_screenshot_copy");
                            unsafe {
                                builder.write_timestamp(
                                    self.frames_in_flight[frame_idx].query_pool.clone(),
                                    q,
                                    PipelineStage::AllCommands,
                                )
                            }
                            .unwrap();
                        }
                    }
                }
            }
        }

        if render_options.prepare_render_screenshot {
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.sized_buffers.output_pixel_buffer.clone(),
                    self.sized_buffers.output_cpu_pixel_buffer.clone(),
                ))
                .unwrap();
            {
                let q = self.profiler.next_query_index("render_screenshot_copy");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }
        }

        // Final frame marker so profiling includes graphics/present-tail work
        // after the last compute phase timestamp.
        {
            let q = self.profiler.next_query_index("frame_end");
            unsafe {
                builder.write_timestamp(
                    self.frames_in_flight[frame_idx].query_pool.clone(),
                    q,
                    PipelineStage::AllCommands,
                )
            }
            .unwrap();
        }

        // Finish recording the command buffer by calling `end`.
        let command_buffer = builder.build().unwrap();

        // Wait for the most recently submitted frame to complete before submitting.
        // This protects shared SizedBuffers (pixel buffer, BVH, tetrahedra) and
        // ensures GPU ordering is maintained.
        if self.frames_rendered > 0 {
            let prev_idx = (self.frames_rendered - 1) % FRAMES_IN_FLIGHT;
            if let Some(prev_fence) = self.frames_in_flight[prev_idx].fence.take() {
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} prev_submit_wait begin (slot={})",
                        self.frames_rendered, prev_idx
                    );
                }
                let wait_start = Instant::now();
                let f = prev_fence.then_signal_fence_and_flush().unwrap();
                f.wait(None).unwrap();
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} prev_submit_wait end ({:.2} ms)",
                        self.frames_rendered,
                        wait_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                // Read back clipped tetrahedron count for diagnostics
                let clipped_count = self.frames_in_flight[prev_idx]
                    .cpu_clipped_tet_count_buffer
                    .read()
                    .map(|data| data[0])
                    .unwrap_or(0);
                self.last_clipped_tet_count = clipped_count;
                if self.drop_next_profile_sample {
                    self.drop_next_profile_sample = false;
                } else {
                    self.profiler.read_results_and_accumulate(
                        &self.frames_in_flight[prev_idx].query_pool,
                        clipped_count,
                    );
                }

                if self.vte_entity_diag_enabled
                    && self.vte_entity_diag_bvh_readback
                    && self.frames_in_flight[prev_idx].vte_entity_diag_copy_scheduled
                {
                    let prev_non_voxel_tets =
                        self.frames_in_flight[prev_idx].vte_entity_diag_non_voxel_tet_count;
                    if prev_non_voxel_tets > 0 {
                        if self.vte_entity_diag_bvh_topology {
                            if let Ok(bvh_data) = self.sized_buffers.cpu_bvh_nodes_buffer.read() {
                                if let Some(root) = bvh_data.first() {
                                    let finite = root.min_bounds.x.is_finite()
                                        && root.min_bounds.y.is_finite()
                                        && root.min_bounds.z.is_finite()
                                        && root.min_bounds.w.is_finite()
                                        && root.max_bounds.x.is_finite()
                                        && root.max_bounds.y.is_finite()
                                        && root.max_bounds.z.is_finite()
                                        && root.max_bounds.w.is_finite();
                                    let ordered = root.min_bounds.x <= root.max_bounds.x
                                        && root.min_bounds.y <= root.max_bounds.y
                                        && root.min_bounds.z <= root.max_bounds.z
                                        && root.min_bounds.w <= root.max_bounds.w;
                                    if let Some(summary) =
                                        summarize_bvh_topology(&bvh_data, prev_non_voxel_tets)
                                    {
                                        let root_child_valid = root.is_leaf != 0
                                            || (root.left_child < summary.total_nodes as u32
                                                && root.right_child < summary.total_nodes as u32);
                                        let root_ready =
                                            root.is_leaf != 0 || root.atomic_visit_count >= 2;
                                        let topology_anomaly = !finite
                                            || !ordered
                                            || !root_child_valid
                                            || !root_ready
                                            || summary.invalid_child_edges > 0
                                            || summary.self_child_edges > 0
                                            || summary.nodes_with_multiple_parents > 0
                                            || summary.nodes_without_parent_excluding_root > 0
                                            || summary.unreachable_internal_nodes > 0
                                            || summary.unreachable_leaf_nodes > 0
                                            || summary.leaf_invalid_tetra_indices > 0
                                            || summary.leaf_duplicate_tetra_indices > 0
                                            || summary.leaf_missing_tetra_indices > 0
                                            || summary.internal_ready < summary.internal_nodes;
                                        if self.vte_entity_diag_verbose || topology_anomaly {
                                            eprintln!(
                                                "[vte-entity-diag][bvh-topology] frame={} prev_frame={} tets={} total_nodes={} internal_ready={}/{} root_ready={} root_finite={} root_ordered={} root_child_valid={} invalid_child_edges={} self_child_edges={} no_parent_ex_root={} multi_parent={} unreachable_internal={} unreachable_leaf={} leaf_invalid_tet={} leaf_duplicate_tet={} leaf_missing_tet={} root(is_leaf={}, visit={}, left={}, right={}) min=({:.3},{:.3},{:.3},{:.3}) max=({:.3},{:.3},{:.3},{:.3})",
                                                self.frames_rendered,
                                                self.frames_rendered.saturating_sub(1),
                                                prev_non_voxel_tets,
                                                summary.total_nodes,
                                                summary.internal_ready,
                                                summary.internal_nodes,
                                                root_ready,
                                                finite,
                                                ordered,
                                                root_child_valid,
                                                summary.invalid_child_edges,
                                                summary.self_child_edges,
                                                summary.nodes_without_parent_excluding_root,
                                                summary.nodes_with_multiple_parents,
                                                summary.unreachable_internal_nodes,
                                                summary.unreachable_leaf_nodes,
                                                summary.leaf_invalid_tetra_indices,
                                                summary.leaf_duplicate_tetra_indices,
                                                summary.leaf_missing_tetra_indices,
                                                root.is_leaf,
                                                root.atomic_visit_count,
                                                root.left_child,
                                                root.right_child,
                                                root.min_bounds.x,
                                                root.min_bounds.y,
                                                root.min_bounds.z,
                                                root.min_bounds.w,
                                                root.max_bounds.x,
                                                root.max_bounds.y,
                                                root.max_bounds.z,
                                                root.max_bounds.w,
                                            );
                                        }
                                    }
                                }
                            }
                        } else if let Ok(root_data) = self.sized_buffers.cpu_bvh_root_buffer.read()
                        {
                            if let Some(root) = root_data.first() {
                                let finite = root.min_bounds.x.is_finite()
                                    && root.min_bounds.y.is_finite()
                                    && root.min_bounds.z.is_finite()
                                    && root.min_bounds.w.is_finite()
                                    && root.max_bounds.x.is_finite()
                                    && root.max_bounds.y.is_finite()
                                    && root.max_bounds.z.is_finite()
                                    && root.max_bounds.w.is_finite();
                                let ordered = root.min_bounds.x <= root.max_bounds.x
                                    && root.min_bounds.y <= root.max_bounds.y
                                    && root.min_bounds.z <= root.max_bounds.z
                                    && root.min_bounds.w <= root.max_bounds.w;
                                let total_nodes =
                                    prev_non_voxel_tets.saturating_mul(2).saturating_sub(1) as u32;
                                let child_valid = root.is_leaf != 0
                                    || (root.left_child < total_nodes
                                        && root.right_child < total_nodes);
                                if self.vte_entity_diag_verbose
                                    || !finite
                                    || !ordered
                                    || !child_valid
                                {
                                    eprintln!(
                                        "[vte-entity-diag] frame={} prev_frame={} bvh_root finite={} ordered={} child_valid={} is_leaf={} left={} right={} tets={} min=({:.3},{:.3},{:.3},{:.3}) max=({:.3},{:.3},{:.3},{:.3})",
                                        self.frames_rendered,
                                        self.frames_rendered.saturating_sub(1),
                                        finite,
                                        ordered,
                                        child_valid,
                                        root.is_leaf,
                                        root.left_child,
                                        root.right_child,
                                        prev_non_voxel_tets,
                                        root.min_bounds.x,
                                        root.min_bounds.y,
                                        root.min_bounds.z,
                                        root.min_bounds.w,
                                        root.max_bounds.x,
                                        root.max_bounds.y,
                                        root.max_bounds.z,
                                        root.max_bounds.w,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Submit the command buffer
        let base_future = sync::now(device.clone());
        match acquire_future {
            Some(acquire_future) => {
                let future = base_future
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            self.swapchain.clone().unwrap().clone(),
                            image_index.unwrap(),
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        self.frames_in_flight[frame_idx].fence = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
            None => {
                let future = base_future
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        self.frames_in_flight[frame_idx].fence = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
        };

        if let Some(window) = self.window.clone() {
            let window_size = window.inner_size();
            // Save frame
            if self.frames_rendered > 3 && render_options.take_framebuffer_screenshot {
                self.wait_for_all_frames();

                let _ = std::fs::create_dir_all("frames");
                let result = self.cpu_screen_capture_buffer.read();
                match result {
                    Ok(buffer_content) => {
                        let screenshot_index = self.frames_rendered - 3;
                        let screenshot_webp_path =
                            format!("frames/framebuffer_{}.webp", screenshot_index);
                        let screenshot_png_path =
                            format!("frames/framebuffer_{}.png", screenshot_index);
                        let (capture_w, capture_h, capture_format) = self
                            .swapchain
                            .as_ref()
                            .map(|swapchain| {
                                let [w, h] = swapchain.image_extent();
                                (w, h, swapchain.image_format())
                            })
                            .unwrap_or((
                                window_size.width,
                                window_size.height,
                                Format::R8G8B8A8_UNORM,
                            ));
                        let expected_bytes = (capture_w as usize)
                            .saturating_mul(capture_h as usize)
                            .saturating_mul(4);

                        let rgba_bytes = match capture_format {
                            Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SRGB => {
                                if buffer_content.len() < expected_bytes {
                                    eprintln!(
                                        "Framebuffer screenshot buffer too small: have {}, need {}",
                                        buffer_content.len(),
                                        expected_bytes
                                    );
                                    None
                                } else {
                                    Some(buffer_content[..expected_bytes].to_vec())
                                }
                            }
                            Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB => {
                                if buffer_content.len() < expected_bytes {
                                    eprintln!(
                                        "Framebuffer screenshot buffer too small: have {}, need {}",
                                        buffer_content.len(),
                                        expected_bytes
                                    );
                                    None
                                } else {
                                    let mut bytes = vec![0u8; expected_bytes];
                                    for (src, dst) in buffer_content[..expected_bytes]
                                        .chunks_exact(4)
                                        .zip(bytes.chunks_exact_mut(4))
                                    {
                                        dst[0] = src[2];
                                        dst[1] = src[1];
                                        dst[2] = src[0];
                                        dst[3] = src[3];
                                    }
                                    Some(bytes)
                                }
                            }
                            _ => {
                                eprintln!(
                                    "Framebuffer screenshot not supported for swapchain format {:?}",
                                    capture_format
                                );
                                None
                            }
                        };

                        if let Some(bytes) = rgba_bytes {
                            if let Some(image) =
                                ImageBuffer::<Rgba<u8>, _>::from_raw(capture_w, capture_h, bytes)
                            {
                                if let Err(err) = image.save(screenshot_webp_path.clone()) {
                                    eprintln!(
                                        "Failed to save screenshot to {}: {}",
                                        screenshot_webp_path, err
                                    );
                                } else {
                                    println!("Saved screenshot to {}", screenshot_webp_path);
                                }
                                if let Err(err) = image.save(screenshot_png_path.clone()) {
                                    eprintln!(
                                        "Failed to save screenshot to {}: {}",
                                        screenshot_png_path, err
                                    );
                                } else {
                                    println!("Saved screenshot to {}", screenshot_png_path);
                                }

                                let camera_h = mat5_mul_vec5(
                                    &view_matrix_nalgebra_inv,
                                    [0.0, 0.0, 0.0, 0.0, 1.0],
                                );
                                let inv_w = if camera_h[4].abs() > 1e-6 {
                                    1.0 / camera_h[4]
                                } else {
                                    1.0
                                };
                                let camera_pos = [
                                    camera_h[0] * inv_w,
                                    camera_h[1] * inv_w,
                                    camera_h[2] * inv_w,
                                    camera_h[3] * inv_w,
                                ];
                                let look_h = mat5_mul_vec5(
                                    &view_matrix_nalgebra_inv,
                                    [
                                        0.0,
                                        0.0,
                                        std::f32::consts::FRAC_1_SQRT_2,
                                        std::f32::consts::FRAC_1_SQRT_2,
                                        0.0,
                                    ],
                                );
                                let mut look = [look_h[0], look_h[1], look_h[2], look_h[3]];
                                let look_len = (look[0] * look[0]
                                    + look[1] * look[1]
                                    + look[2] * look[2]
                                    + look[3] * look[3])
                                    .sqrt();
                                if look_len > 1e-6 {
                                    for c in &mut look {
                                        *c /= look_len;
                                    }
                                }
                                println!(
                                    "Screenshot meta frame={} backend={} size={}x{} layers={} focal_xy={:.3} focal_zw={:.3}",
                                    screenshot_index,
                                    self.last_backend.label(),
                                    capture_w,
                                    capture_h,
                                    self.sized_buffers.render_dimensions[2],
                                    focal_length_xy,
                                    focal_length_zw
                                );
                                if self.last_backend == RenderBackend::VoxelTraversal {
                                    println!(
                                        "  VTE mode={} slice_layer={:?} thick_half_width={} max_steps={} max_distance={:.1} reference_compare={} mismatch_only={} compare_slice_only={} lod_tint={}",
                                        render_options.vte_display_mode.label(),
                                        render_options.vte_slice_layer,
                                        render_options.vte_thick_half_width,
                                        render_options.vte_max_trace_steps,
                                        render_options.vte_max_trace_distance,
                                        render_options.vte_reference_compare,
                                        render_options.vte_reference_mismatch_only,
                                        render_options.vte_compare_slice_only,
                                        vte_lod_tint_enabled(),
                                    );
                                    if self.vte_compare_stats.compared > 0
                                        || self.vte_compare_stats.mismatches > 0
                                    {
                                        println!(
                                            "  VTE compare compared={} match={} mismatch={} hit_state={} chunk_material={} fm_ref={} fh_ref={} reason=[none:{} touched:{} voxel:{} chunk:{} dist:{} lookup:{}] flags=[zero:{} tie:{} fallback:{}]",
                                            self.vte_compare_stats.compared,
                                            self.vte_compare_stats.matches,
                                            self.vte_compare_stats.mismatches,
                                            self.vte_compare_stats.hit_state_mismatches,
                                            self.vte_compare_stats.chunk_material_mismatches,
                                            self.vte_compare_stats.fast_miss_ref_hit,
                                            self.vte_compare_stats.fast_hit_ref_miss,
                                            self.vte_compare_stats.miss_reason_counts[0],
                                            self.vte_compare_stats.miss_reason_counts[1],
                                            self.vte_compare_stats.miss_reason_counts[2],
                                            self.vte_compare_stats.miss_reason_counts[3],
                                            self.vte_compare_stats.miss_reason_counts[4],
                                            self.vte_compare_stats.miss_reason_counts[5],
                                            self.vte_compare_stats.zero_interval_flags,
                                            self.vte_compare_stats.tie_stepped_flags,
                                            self.vte_compare_stats.lookup_fallback_flags,
                                        );
                                        if self.vte_first_mismatch.valid {
                                            println!(
                                                "  VTE first_mismatch px=({}, {}, l={}) kind={} miss_reason={} debug=0x{:x} hit=({}/{}) fast_chunk=({},{},{},{}) ref_chunk=({},{},{},{}) mat=({}/{}) t=({:.5}/{:.5}) steps={} rem_vox={} final_t={:.5} last_chunk=({},{},{},{})",
                                                self.vte_first_mismatch.pixel_x,
                                                self.vte_first_mismatch.pixel_y,
                                                self.vte_first_mismatch.layer,
                                                self.vte_first_mismatch.mismatch_kind,
                                                self.vte_first_mismatch.miss_reason,
                                                self.vte_first_mismatch.debug_flags,
                                                self.vte_first_mismatch.fast_hit as u32,
                                                self.vte_first_mismatch.ref_hit as u32,
                                                self.vte_first_mismatch.fast_chunk[0],
                                                self.vte_first_mismatch.fast_chunk[1],
                                                self.vte_first_mismatch.fast_chunk[2],
                                                self.vte_first_mismatch.fast_chunk[3],
                                                self.vte_first_mismatch.ref_chunk[0],
                                                self.vte_first_mismatch.ref_chunk[1],
                                                self.vte_first_mismatch.ref_chunk[2],
                                                self.vte_first_mismatch.ref_chunk[3],
                                                self.vte_first_mismatch.fast_material,
                                                self.vte_first_mismatch.ref_material,
                                                self.vte_first_mismatch.fast_hit_t,
                                                self.vte_first_mismatch.ref_hit_t,
                                                self.vte_first_mismatch.chunk_steps_taken,
                                                self.vte_first_mismatch.remaining_voxel_steps,
                                                self.vte_first_mismatch.final_t,
                                                self.vte_first_mismatch.last_chunk[0],
                                                self.vte_first_mismatch.last_chunk[1],
                                                self.vte_first_mismatch.last_chunk[2],
                                                self.vte_first_mismatch.last_chunk[3],
                                            );
                                        }
                                    }
                                    if self.vte_compare_stats.stagea_samples > 0 {
                                        let s = self.vte_compare_stats.stagea_samples as f64;
                                        println!(
                                            "  VTE stagea samples={} entity_query={} ({:.1}%) entity_hit={} ({:.1}%) voxel_hit={} ({:.1}%) sky={} ({:.1}%) avg_chunk_steps={:.2} avg_voxel_steps={:.2} avg_node_visits={:.2}",
                                            self.vte_compare_stats.stagea_samples,
                                            self.vte_compare_stats.stagea_entity_queries,
                                            self.vte_compare_stats.stagea_entity_queries as f64 * 100.0 / s,
                                            self.vte_compare_stats.stagea_entity_hits,
                                            self.vte_compare_stats.stagea_entity_hits as f64 * 100.0 / s,
                                            self.vte_compare_stats.stagea_voxel_hits,
                                            self.vte_compare_stats.stagea_voxel_hits as f64 * 100.0 / s,
                                            self.vte_compare_stats.stagea_sky_misses,
                                            self.vte_compare_stats.stagea_sky_misses as f64 * 100.0 / s,
                                            self.vte_compare_stats.stagea_chunk_steps_sum as f64 / s,
                                            self.vte_compare_stats.stagea_voxel_steps_sum as f64 / s,
                                            self.vte_compare_stats.stagea_node_visits_sum as f64 / s,
                                        );
                                    }
                                }
                                println!(
                                    "  POS {:+.4} {:+.4} {:+.4} {:+.4}",
                                    camera_pos[0], camera_pos[1], camera_pos[2], camera_pos[3]
                                );
                                println!(
                                    "  LOOK {:+.4} {:+.4} {:+.4} {:+.4}",
                                    look[0], look[1], look[2], look[3]
                                );
                            } else {
                                eprintln!(
                                    "Failed to build screenshot image buffer ({}x{})",
                                    capture_w, capture_h
                                );
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!("Error saving screenshot: {:?}", error);
                    }
                };
            }
        }

        // Debug: print BVH diagnostics on first frame
        if self.frames_rendered == 0 && do_raytrace {
            // Ensure GPU work is done
            self.wait_for_all_frames();

            let bvh_nodes = self.sized_buffers.cpu_bvh_nodes_buffer.read().unwrap();
            let morton_codes = self.sized_buffers.cpu_morton_codes_buffer.read().unwrap();
            let num_leaves = total_tetrahedron_count;
            let num_internal = num_leaves.saturating_sub(1);
            let total_nodes = num_leaves.saturating_mul(2).saturating_sub(1);

            // Check Morton code sorting
            let mut sorted = true;
            for i in 1..num_leaves {
                if morton_codes[i].code < morton_codes[i - 1].code {
                    println!(
                        "  SORT ERROR at {}: code[{}]={} > code[{}]={}",
                        i,
                        i - 1,
                        morton_codes[i - 1].code,
                        i,
                        morton_codes[i].code
                    );
                    sorted = false;
                }
            }
            println!("Morton codes sorted: {}", sorted);

            // Check root node
            let root = &bvh_nodes[0];
            println!(
                "Root node: left={}, right={}, isLeaf={}, visitCount={}",
                root.left_child, root.right_child, root.is_leaf, root.atomic_visit_count
            );
            println!(
                "Root AABB: min=({:.2},{:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2},{:.2})",
                root.min_bounds.x,
                root.min_bounds.y,
                root.min_bounds.z,
                root.min_bounds.w,
                root.max_bounds.x,
                root.max_bounds.y,
                root.max_bounds.z,
                root.max_bounds.w
            );

            // Count valid internal nodes
            let mut valid_internal = 0;
            let mut invalid_children = 0;
            let mut zero_aabb_internal = 0;
            for i in 0..num_internal {
                if bvh_nodes[i].atomic_visit_count >= 2 {
                    valid_internal += 1;
                }
                if (bvh_nodes[i].left_child >= total_nodes as u32
                    || bvh_nodes[i].right_child >= total_nodes as u32)
                    && bvh_nodes[i].left_child != 0xFFFFFFFF
                    && bvh_nodes[i].right_child != 0xFFFFFFFF
                {
                    invalid_children += 1;
                }
                let aabb_size = (bvh_nodes[i].max_bounds - bvh_nodes[i].min_bounds).length();
                if aabb_size < 0.001 && bvh_nodes[i].atomic_visit_count >= 2 {
                    zero_aabb_internal += 1;
                }
            }
            println!(
                "Internal nodes: {}/{} valid (visitCount>=2), {} invalid children, {} zero-AABB",
                valid_internal, num_internal, invalid_children, zero_aabb_internal
            );

            // Count valid leaves
            let mut zero_aabb_leaves = 0;
            for i in 0..num_leaves {
                let leaf_idx = num_internal + i;
                let aabb_size =
                    (bvh_nodes[leaf_idx].max_bounds - bvh_nodes[leaf_idx].min_bounds).length();
                if aabb_size < 0.001 {
                    zero_aabb_leaves += 1;
                }
            }
            println!("Leaves: {}/{} with zero AABB", zero_aabb_leaves, num_leaves);

            // Print first few internal nodes for inspection
            println!("First 5 internal nodes:");
            for i in 0..5.min(num_internal) {
                let n = &bvh_nodes[i];
                let aabb_size = (n.max_bounds - n.min_bounds).length();
                println!("  [{}] L={} R={} leaf={} visit={} aabb_size={:.2} min=({:.1},{:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1},{:.1})",
                         i, n.left_child, n.right_child, n.is_leaf, n.atomic_visit_count,
                         aabb_size,
                         n.min_bounds.x, n.min_bounds.y, n.min_bounds.z, n.min_bounds.w,
                         n.max_bounds.x, n.max_bounds.y, n.max_bounds.z, n.max_bounds.w);
            }

            // Print a few leaves
            println!("First 5 leaf nodes:");
            for i in 0..5.min(num_leaves) {
                let leaf_idx = num_internal + i;
                let n = &bvh_nodes[leaf_idx];
                let aabb_size = (n.max_bounds - n.min_bounds).length();
                println!("  [{}] tetIdx={} leaf={} aabb_size={:.2} min=({:.1},{:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1},{:.1})",
                         leaf_idx, n.tetrahedron_index, n.is_leaf, aabb_size,
                         n.min_bounds.x, n.min_bounds.y, n.min_bounds.z, n.min_bounds.w,
                         n.max_bounds.x, n.max_bounds.y, n.max_bounds.z, n.max_bounds.w);
            }
        }

        self.frames_rendered += 1;
    }
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn create_cpu_screencapture_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    width: u32,
    height: u32,
    format: Format,
) -> Subbuffer<[u8]> {
    let block_extent = format.block_extent();
    let blocks_x = width.div_ceil(block_extent[0]) as u64;
    let blocks_y = height.div_ceil(block_extent[1]) as u64;
    let byte_len = blocks_x
        .saturating_mul(blocks_y)
        .saturating_mul(format.block_size()) as usize;

    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        vec![0; byte_len],
    )
    .unwrap()
}
