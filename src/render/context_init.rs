use super::*;

impl RenderContext {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        instance: Arc<Instance>,
        window: Option<Arc<Window>>,
        render_dimensions: [u32; 3],
    ) -> RenderContext {
        Self::new_with_pixel_storage_layers(
            device,
            queue,
            instance,
            window,
            render_dimensions,
            None,
        )
    }

    pub fn new_with_pixel_storage_layers(
        device: Arc<Device>,
        queue: Arc<Queue>,
        instance: Arc<Instance>,
        window: Option<Arc<Window>>,
        render_dimensions: [u32; 3],
        pixel_storage_layers: Option<u32>,
    ) -> RenderContext {
        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let (surface, window_size) = match window.clone() {
            Some(window) => (
                Some(Surface::from_window(instance.clone(), window.clone()).unwrap()),
                window.inner_size(),
            ),
            None => (
                None,
                PhysicalSize {
                    width: render_dimensions[0],
                    height: render_dimensions[1],
                },
            ),
        };

        // Before we can draw on the surface, we have to create what is called a swapchain.
        // Creating a swapchain allocates the color buffers that will contain the image that will
        // ultimately be visible on the screen. These images are returned alongside the swapchain.
        let (swapchain, images) = match surface {
            Some(surface) => {
                // Querying the capabilities of the surface. When we create the swapchain we can only
                // pass values that are allowed by the capabilities.
                let surface_capabilities = device
                    .physical_device()
                    .surface_capabilities(&surface, Default::default())
                    .unwrap();

                // Choosing the internal format that the images will have.
                let image_formats = device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap();

                let preferred_swapchain_formats = [
                    Format::B8G8R8A8_SRGB,
                    Format::B8G8R8A8_UNORM,
                    Format::R8G8B8A8_SRGB,
                    Format::R8G8B8A8_UNORM,
                ];
                let (image_format, image_color_space) = preferred_swapchain_formats
                    .iter()
                    .find_map(|preferred| {
                        image_formats
                            .iter()
                            .copied()
                            .find(|(fmt, _)| *fmt == *preferred)
                    })
                    .or_else(|| {
                        image_formats
                            .iter()
                            .copied()
                            .find(|(fmt, _)| fmt.block_size() == 4)
                    })
                    .unwrap_or(image_formats[0]);
                //let image_format = R8G8B8A8_UNORM;
                // Please take a look at the docs for the meaning of the parameters we didn't mention.
                let (swapchain, images) = Swapchain::new(
                    device.clone(),
                    surface,
                    SwapchainCreateInfo {
                        // Some drivers report an `min_image_count` of 1, but fullscreen mode requires
                        // at least 2. Therefore we must ensure the count is at least 2, otherwise the
                        // program would crash when entering fullscreen mode on those drivers.
                        min_image_count: surface_capabilities.min_image_count.max(2),

                        image_format,
                        image_color_space,

                        // The size of the window, only used to initially setup the swapchain.
                        //
                        // NOTE:
                        // On some drivers the swapchain extent is specified by
                        // `surface_capabilities.current_extent` and the swapchain size must use this
                        // extent. This extent is always the same as the window size.
                        //
                        // However, other drivers don't specify a value, i.e.
                        // `surface_capabilities.current_extent` is `None`. These drivers will allow
                        // anything, but the only sensible value is the window size.
                        //
                        // Both of these cases need the swapchain to use the window size, so we just
                        // use that.
                        image_extent: window_size.into(),

                        image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,

                        // The alpha mode indicates how the alpha value of the final image will behave.
                        // For example, you can choose whether the window will be
                        // opaque or transparent.
                        composite_alpha: surface_capabilities
                            .supported_composite_alpha
                            .into_iter()
                            .next()
                            .unwrap(),

                        ..Default::default()
                    },
                )
                .unwrap();
                (Some(swapchain), Some(images))
            }
            None => (None, None),
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Load shaders from SPIR-V bytes embedded at compile time
        // (vulkano_shaders macro can't parse Slang's SPIR-V 1.4 output).
        fn load_shader(device: Arc<Device>, spirv: &[u8]) -> Arc<ShaderModule> {
            let words = vulkano::shader::spirv::bytes_to_words(spirv)
                .expect("SPIR-V bytes length must be a multiple of 4");
            unsafe {
                ShaderModule::new(device, ShaderModuleCreateInfo::new(words.as_ref()))
                    .expect("Failed to load shader module")
            }
        }

        macro_rules! shader_spirv {
            ($name:literal) => {
                include_bytes!(concat!(env!("SPIRV_OUT_DIR"), "/", $name, ".spv"))
            };
        }

        let raytrace_pixel = load_shader(device.clone(), shader_spirv!("mainRaytracerPixel"));
        let raytrace_preprocess = load_shader(
            device.clone(),
            shader_spirv!("mainRaytracerTetrahedronPreprocessor"),
        );
        let entity_instance_aabb_preprocess = load_shader(
            device.clone(),
            shader_spirv!("mainEntityInstanceAabbPreprocessor"),
        );
        let raytrace_clear = load_shader(device.clone(), shader_spirv!("mainRaytracerClear"));
        let voxel_trace_stage_a_integral_fused = load_shader(
            device.clone(),
            shader_spirv!("mainVoxelTraceStageAIntegralFused"),
        );
        let voxel_trace_stage_a_layered =
            load_shader(device.clone(), shader_spirv!("mainVoxelTraceStageALayered"));
        let voxel_display_stage_b =
            load_shader(device.clone(), shader_spirv!("mainVoxelDisplayStageB"));
        let raster_tet = load_shader(device.clone(), shader_spirv!("mainTetrahedronCS"));
        let raster_edge = load_shader(device.clone(), shader_spirv!("mainEdgeCS"));
        let raster_pixel = load_shader(device.clone(), shader_spirv!("mainTetrahedronPixelCS"));
        let bin_tets_cs = load_shader(device.clone(), shader_spirv!("mainBinTetsCS"));
        let present_line_vs = load_shader(device.clone(), shader_spirv!("mainLineVS"));
        let present_line_fs = load_shader(device.clone(), shader_spirv!("mainLineFS"));
        let present_buffer_vs = load_shader(device.clone(), shader_spirv!("mainBufferVS"));
        let present_buffer_fs = load_shader(device.clone(), shader_spirv!("mainBufferFS"));
        // HUD shaders
        let hud_vs = load_shader(device.clone(), shader_spirv!("mainHudVS"));
        let hud_fs = load_shader(device.clone(), shader_spirv!("mainHudFS"));
        // BVH shaders
        let bvh_scene_bounds = load_shader(device.clone(), shader_spirv!("mainBVHSceneBounds"));
        let bvh_morton_codes = load_shader(device.clone(), shader_spirv!("mainBVHMortonCodes"));
        let bvh_bitonic_sort_local =
            load_shader(device.clone(), shader_spirv!("mainBVHBitonicSortLocal"));
        let bvh_bitonic_sort = load_shader(device.clone(), shader_spirv!("mainBVHBitonicSort"));
        let bvh_bitonic_sort_local_merge = load_shader(
            device.clone(),
            shader_spirv!("mainBVHBitonicSortLocalMerge"),
        );
        let bvh_init_leaves = load_shader(device.clone(), shader_spirv!("mainBVHInitLeaves"));
        let bvh_build_tree = load_shader(device.clone(), shader_spirv!("mainBVHBuildTree"));
        let bvh_link_parents = load_shader(device.clone(), shader_spirv!("mainBVHLinkParents"));
        let bvh_propagate_aabbs =
            load_shader(device.clone(), shader_spirv!("mainBVHPropagateAABBs"));

        let render_pass = swapchain.clone().map(|swapchain| {
            // The next step is to create a *render pass*, which is an object that describes where the
            // output of the graphics pipeline will go. It describes the layout of the images where the
            // colors, depth and/or stencil information will be written.
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    // `color` is a custom name we give to the first and only attachment.
                    color: {
                        // `format: <ty>` indicates the type of the format of the image. This has to be
                        // one of the types of the `vulkano::format` module (or alternatively one of
                        // your structs that implements the `FormatDesc` trait). Here we use the same
                        // format as the swapchain.
                        format: swapchain.image_format(),
                        // `samples: 1` means that we ask the GPU to use one sample to determine the
                        // value of each pixel in the color attachment. We could use a larger value
                        // (multisampling) for antialiasing. An example of this can be found in
                        // msaa-renderpass.rs.
                        samples: 1,
                        // `load_op: Clear` means that we ask the GPU to clear the content of this
                        // attachment at the start of the drawing.
                        load_op: Clear,
                        // `store_op: Store` means that we ask the GPU to store the output of the draw
                        // in the actual image. We could also ask it to discard the result.
                        store_op: Store,
                    },
                },
                pass: {
                    // We use the attachment named `color` as the one and only color attachment.
                    color: [color],
                    // No depth-stencil attachment is indicated with empty brackets.
                    depth_stencil: {},
                },
            )
            .unwrap()
        });

        let framebuffers = render_pass.clone().and_then(|render_pass| {
            images.map(|images| window_size_dependent_setup(&images, &render_pass))
        });
        let aetna_overlay = swapchain.as_ref().map(|swapchain| {
            let mut runner =
                aetna_vulkano::Runner::new(device.clone(), queue.clone(), swapchain.image_format());
            runner.set_theme(aetna_core::Theme::radix_slate_blue_dark());
            runner.set_surface_size(window_size.width.max(1), window_size.height.max(1));
            AetnaOverlay { runner }
        });

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let one_time_descriptor_set_layout =
            OneTimeBuffers::create_descriptor_set_layout(device.clone());
        let one_time_buffers = OneTimeBuffers::new(
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            one_time_descriptor_set_layout.clone(),
        );

        let sized_descriptor_set_layout =
            SizedBuffers::create_descriptor_set_layout(device.clone());
        let sized_buffers = SizedBuffers::new(
            memory_allocator.clone(),
            render_dimensions,
            pixel_storage_layers,
        );

        let live_descriptor_set_layout = LiveBuffers::create_descriptor_set_layout(device.clone());

        let texture_pool = TexturePool::new(
            device.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            queue.clone(),
        );

        // We must now create a **pipeline layout** object, which describes the locations and
        // types of descriptor sets and push constants used by the shaders in the pipeline.
        //
        // Multiple pipelines can share a common layout object, which is more efficient. The
        // shaders in a pipeline must use a subset of the resources described in its pipeline
        // layout, but the pipeline layout is allowed to contain resources that are not present
        // in the shaders; they can be used by shaders in other pipelines that share the same
        // layout. Thus, it is a good idea to design shaders so that many pipelines have common
        // resource locations, which allows them to share pipeline layouts.
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: Vec::from([
                    one_time_descriptor_set_layout.clone(),
                    sized_descriptor_set_layout.clone(),
                    live_descriptor_set_layout.clone(),
                    texture_pool.descriptor_set_layout().clone(),
                ]),
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: 16, // 4 u32s: stage, step, count, padding
                }],
                ..Default::default()
            },
        )
        .unwrap();

        // Create ShaderModules struct from individually loaded shaders
        let shaders = ShaderModules {
            line_vs: present_line_vs,
            line_fs: present_line_fs,
            buffer_vs: present_buffer_vs,
            buffer_fs: present_buffer_fs,
            hud_vs,
            hud_fs,
            tetrahedron_cs: raster_tet,
            edge_cs: raster_edge,
            tetrahedron_pixel_cs: raster_pixel,
            bin_tets_cs,
            raytrace_preprocess,
            entity_instance_aabb_preprocess,
            raytrace_pixel,
            raytrace_clear,
            voxel_trace_stage_a_integral_fused,
            voxel_trace_stage_a_layered,
            voxel_display_stage_b,
            bvh_scene_bounds,
            bvh_morton_codes,
            bvh_bitonic_sort_local,
            bvh_bitonic_sort,
            bvh_bitonic_sort_local_merge,
            bvh_init_leaves,
            bvh_build_tree,
            bvh_link_parents,
            bvh_propagate_aabbs,
        };

        let present_pipeline = render_pass.clone().map(|render_pass| {
            PresentPipelineContext::new(
                device.clone(),
                render_pass.clone(),
                &shaders,
                pipeline_layout.clone(),
            )
        });
        let compute_pipeline =
            ComputePipelineContext::new(device.clone(), &shaders, pipeline_layout.clone());

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let cpu_screen_capture_buffer = match swapchain.as_ref() {
            Some(swapchain) => {
                let [width, height] = swapchain.image_extent();
                create_cpu_screencapture_buffer(
                    memory_allocator.clone(),
                    width,
                    height,
                    swapchain.image_format(),
                )
            }
            None => create_cpu_screencapture_buffer(
                memory_allocator.clone(),
                window_size.width,
                window_size.height,
                Format::R8G8B8A8_UNORM,
            ),
        };

        // In some situations, the swapchain will become invalid by itself. This includes for
        // example when the window is resized (as the images of the swapchain will no longer match
        // the window's) or, on Android, when the application went to the background and goes back
        // to the foreground.
        //
        // In this situation, acquiring a swapchain image or presenting it will return an error.
        // Rendering to an image of that swapchain will not produce any error, but may or may not
        // work. To continue rendering, we need to recreate the swapchain by creating a new
        // swapchain. Here, we remember that we need to do this for the next loop iteration.
        let recreate_swapchain = false;

        let profiler = GpuProfiler::new(device.clone());
        let hud_font = load_hud_font();

        // Build HUD resources (font atlas + sampler).
        let hud_resources = match (&hud_font, &present_pipeline) {
            (Some(font), Some(present_ctx)) => {
                let font_atlas = build_font_atlas(font, 32.0);
                let atlas_view = create_rgba8_srgb_texture_view(
                    memory_allocator.clone(),
                    command_buffer_allocator.clone(),
                    queue.clone(),
                    font_atlas.width,
                    font_atlas.height,
                    &font_atlas.pixels,
                );
                let atlas_sampler = Sampler::new(
                    device.clone(),
                    SamplerCreateInfo {
                        mag_filter: Filter::Linear,
                        min_filter: Filter::Linear,
                        address_mode: [SamplerAddressMode::ClampToEdge; 3],
                        ..Default::default()
                    },
                )
                .unwrap();

                let hud_descriptor_set_layout = present_ctx
                    .hud_pipeline_layout
                    .set_layouts()
                    .first()
                    .unwrap()
                    .clone();

                Some(HudResources {
                    font_atlas,
                    atlas_view,
                    atlas_sampler,
                    hud_descriptor_set_layout,
                })
            }
            _ => None,
        };

        let egui_resources = match &present_pipeline {
            Some(_) => {
                let texture_pixels = vec![255u8, 255, 255, 255];
                let atlas_view = create_rgba8_srgb_texture_view(
                    memory_allocator.clone(),
                    command_buffer_allocator.clone(),
                    queue.clone(),
                    1,
                    1,
                    &texture_pixels,
                );
                let atlas_sampler = Sampler::new(
                    device.clone(),
                    SamplerCreateInfo {
                        mag_filter: Filter::Linear,
                        min_filter: Filter::Linear,
                        address_mode: [SamplerAddressMode::ClampToEdge; 3],
                        ..Default::default()
                    },
                )
                .unwrap();
                Some(EguiResources {
                    atlas_view,
                    atlas_sampler,
                    texture_size: [1, 1],
                    texture_pixels,
                    retired_atlas_views: Vec::new(),
                })
            }
            None => None,
        };

        let hud_descriptor_set_layout = present_pipeline.as_ref().map(|present_ctx| {
            present_ctx
                .hud_pipeline_layout
                .set_layouts()
                .first()
                .unwrap()
                .clone()
        });

        // Create per-frame resources
        let mut frames_in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let live_buffers = LiveBuffers::new(
                memory_allocator.clone(),
                descriptor_set_allocator.clone(),
                live_descriptor_set_layout.clone(),
            );

            let line_vertexes_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![LineVertex::zeroed(); LINE_VERTEX_CAPACITY],
            )
            .unwrap();

            let sized_descriptor_set = sized_buffers.create_sized_descriptor_set(
                &line_vertexes_buffer,
                descriptor_set_allocator.clone(),
                sized_descriptor_set_layout.clone(),
            );

            let cpu_clipped_tet_count_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                vec![0u32; 1],
            )
            .unwrap();

            let (hud_vertex_buffer, hud_descriptor_set, egui_descriptor_set) =
                match hud_descriptor_set_layout.as_ref() {
                    Some(layout) => {
                        let hud_vertex_buffer = Buffer::from_iter(
                            memory_allocator.clone(),
                            BufferCreateInfo {
                                usage: BufferUsage::STORAGE_BUFFER,
                                ..Default::default()
                            },
                            AllocationCreateInfo {
                                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                ..Default::default()
                            },
                            vec![HudVertex::zeroed(); HUD_VERTEX_CAPACITY],
                        )
                        .unwrap();

                        let hud_descriptor_set = hud_resources.as_ref().map(|hud_res| {
                            create_hud_descriptor_set(
                                descriptor_set_allocator.clone(),
                                layout.clone(),
                                hud_vertex_buffer.clone(),
                                hud_res.atlas_view.clone(),
                                hud_res.atlas_sampler.clone(),
                            )
                        });
                        let egui_descriptor_set = egui_resources.as_ref().map(|egui_res| {
                            create_hud_descriptor_set(
                                descriptor_set_allocator.clone(),
                                layout.clone(),
                                hud_vertex_buffer.clone(),
                                egui_res.atlas_view.clone(),
                                egui_res.atlas_sampler.clone(),
                            )
                        });

                        (
                            Some(hud_vertex_buffer),
                            hud_descriptor_set,
                            egui_descriptor_set,
                        )
                    }
                    None => (None, None, None),
                };

            let query_pool = GpuProfiler::create_query_pool(&device);

            frames_in_flight.push(FrameInFlight {
                live_buffers,
                line_vertexes_buffer,
                hud_vertex_buffer,
                hud_descriptor_set,
                egui_descriptor_set,
                material_icons_descriptor_set: None,
                sized_descriptor_set,
                cpu_clipped_tet_count_buffer,
                query_pool,
                fence: None,
                vte_compare_enabled: false,
                vte_world_bvh_ray_diag_enabled: false,
                last_voxel_metadata_generation: None,
                vte_world_bvh_ray_diag_expected: Vec::new(),
                vte_entity_diag_copy_scheduled: false,
                vte_entity_diag_non_voxel_tet_count: 0,
            });
        }

        let vte_entity_diag_enabled =
            vte_diagnostics_feature_enabled() && env_flag_enabled(VTE_ENTITY_DIAG_ENV);
        let vte_entity_diag_verbose =
            vte_entity_diag_enabled && env_flag_enabled(VTE_ENTITY_DIAG_VERBOSE_ENV);
        let vte_entity_diag_bvh_readback =
            vte_entity_diag_enabled && env_flag_enabled(VTE_ENTITY_DIAG_BVH_READBACK_ENV);
        let vte_entity_diag_bvh_topology = vte_entity_diag_bvh_readback
            && vte_entity_diag_enabled
            && env_flag_enabled(VTE_ENTITY_DIAG_BVH_TOPOLOGY_ENV);
        let vte_entity_diag_interval = env_usize(
            VTE_ENTITY_DIAG_BVH_INTERVAL_ENV,
            VTE_ENTITY_DIAG_DEFAULT_INTERVAL,
        );
        let vte_stage_a_breakdown_enabled = vte_stage_a_breakdown_enabled();
        let vte_stage_a_breakdown_interval = env_usize(
            VTE_STAGE_A_BREAKDOWN_INTERVAL_ENV,
            VTE_STAGE_A_BREAKDOWN_DEFAULT_INTERVAL,
        )
        .max(1);
        let vte_world_bvh_ray_diag_enabled = vte_world_bvh_ray_diag_env_enabled();
        let vte_world_bvh_ray_diag_samples = env_usize(
            VTE_WORLD_BVH_RAY_DIAG_SAMPLES_ENV,
            VTE_WORLD_BVH_RAY_DIAG_DEFAULT_SAMPLES,
        )
        .clamp(1, vte::VTE_WORLD_BVH_RAY_DIAG_CAPACITY);
        let vte_world_bvh_ray_diag_interval = env_usize(
            VTE_WORLD_BVH_RAY_DIAG_INTERVAL_ENV,
            VTE_WORLD_BVH_RAY_DIAG_DEFAULT_INTERVAL,
        )
        .max(1);
        if vte_entity_diag_enabled {
            eprintln!(
                "VTE entity diagnostics enabled (verbose={}, bvh_readback={}, bvh_topology={}, interval={} frames).",
                vte_entity_diag_verbose,
                vte_entity_diag_bvh_readback,
                vte_entity_diag_bvh_topology,
                vte_entity_diag_interval
            );
        }
        if vte_world_bvh_ray_diag_enabled {
            eprintln!(
                "VTE world BVH ray diagnostics enabled (samples={}, interval={} frames).",
                vte_world_bvh_ray_diag_samples, vte_world_bvh_ray_diag_interval
            );
        }
        if vte_stage_a_breakdown_enabled {
            eprintln!(
                "VTE Stage A breakdown diagnostics enabled (interval={} frames).",
                vte_stage_a_breakdown_interval
            );
        }

        RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            present_pipeline,
            compute_pipeline,
            viewport,
            recreate_swapchain,
            command_buffer_allocator,
            descriptor_set_allocator,
            memory_allocator,
            one_time_buffers,
            sized_buffers,
            frames_in_flight,
            cpu_screen_capture_buffer,
            frames_rendered: 0,
            bvh_scene_hash: 0,
            vte_non_voxel_scene_hash: 0,
            vte_non_voxel_bvh_topology_tet_count: 0,
            vte_non_voxel_bvh_refit_frames: 0,
            last_clipped_tet_count: 0,
            profiler,
            hud_font,
            hud_resources,
            egui_resources,
            material_icons_view: None,
            material_icons_sampler: None,
            aetna_overlay,
            hud_breadcrumbs: VecDeque::new(),
            hud_previous_camera: None,
            hud_previous_sample_time: None,
            hud_w_velocity: 0.0,
            frame_time_ms: 0.0,
            last_render_start: None,
            stall_trace: std::env::var_os("R4D_TRACE_STALLS").is_some(),
            last_backend: RenderBackend::Auto,
            vte_debug_counters: VteDebugCounters::default(),
            vte_compare_stats: vte::VteCompareStats::default(),
            vte_first_mismatch: vte::VteFirstMismatch::default(),
            vte_backend_notice_printed: false,
            vte_entity_diag_enabled,
            vte_entity_diag_verbose,
            vte_entity_diag_bvh_readback,
            vte_entity_diag_bvh_topology,
            vte_entity_diag_interval,
            vte_entity_diag_last_log_frame: None,
            vte_stage_a_breakdown_enabled,
            vte_stage_a_breakdown_interval,
            vte_stage_a_breakdown_last_log_frame: None,
            vte_world_bvh_ray_diag_enabled,
            vte_world_bvh_ray_diag_samples,
            vte_world_bvh_ray_diag_interval,
            vte_world_bvh_ray_diag_last_log_frame: None,
            vte_entity_diag_prev_used_non_voxel: None,
            vte_entity_diag_prev_tets_non_voxel: None,
            drop_next_profile_sample: false,
            texture_pool,
        }
    }
}
