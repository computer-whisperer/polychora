use super::*;
use polychora_plugin_api::texture::TextureRef;

impl App {
    /// Paint an icon from the material icon sheet by TextureRef, falling back
    /// to a flat color swatch when the sheet entry is missing or unavailable.
    fn paint_icon(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        tex_ref: Option<TextureRef>,
        fallback_color: [u8; 3],
    ) {
        let resolved = tex_ref.and_then(|t| {
            let sheet = self.material_icon_sheet.as_ref()?;
            let tex_id = self.material_icons_texture_id?;
            let uv = sheet.uv_rect(t.namespace, t.texture_id)?;
            Some((tex_id, uv))
        });
        if let Some((tex_id, [u0, v0, u1, v1])) = resolved {
            painter.image(
                tex_id,
                rect,
                egui::Rect::from_min_max(egui::pos2(u0, v0), egui::pos2(u1, v1)),
                egui::Color32::WHITE,
            );
        } else {
            let [r, g, b] = fallback_color;
            painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(r, g, b));
        }
    }
    pub(super) fn draw_egui_pause_menu(
        &mut self,
        ctx: &egui::Context,
        close_menu: &mut bool,
        return_to_main_menu: &mut bool,
    ) {
        egui::Window::new("Polychora")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([460.0, 500.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Resume").clicked() {
                        *close_menu = true;
                    }
                    if ui.button("Controls").clicked() {
                        self.controls_dialog_open = !self.controls_dialog_open;
                    }
                    if ui.button("Main Menu").clicked() {
                        *return_to_main_menu = true;
                    }
                    if ui.button("Quit").clicked() {
                        self.should_exit_after_render = true;
                    }
                });

                ui.separator();

                // Page tabs
                ui.horizontal(|ui| {
                    for page in SettingsPage::ALL {
                        ui.selectable_value(&mut self.settings_page, page, page.label());
                    }
                });

                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| match self.settings_page {
                        SettingsPage::Gameplay => self.draw_settings_gameplay(ui),
                        SettingsPage::Rendering => self.draw_settings_rendering(ui),
                        SettingsPage::Advanced => self.draw_settings_advanced(ui),
                        SettingsPage::Debug => self.draw_settings_debug(ui),
                    });
            });
    }

    fn draw_settings_gameplay(&mut self, ui: &mut egui::Ui) {
        let mut selected_control_scheme = self.control_scheme;
        egui::ComboBox::from_label("Control Scheme")
            .selected_text(selected_control_scheme.label())
            .show_ui(ui, |ui| {
                for scheme in [
                    ControlScheme::IntuitiveUpright,
                    ControlScheme::LookTransport,
                    ControlScheme::TransportUniform,
                    ControlScheme::TransportDecoupled,
                    ControlScheme::TransportScaled,
                    ControlScheme::RotorFree,
                    ControlScheme::LegacySideButtonLayers,
                    ControlScheme::LegacyScrollCycle,
                ] {
                    ui.selectable_value(&mut selected_control_scheme, scheme, scheme.label());
                }
            });
        if selected_control_scheme != self.control_scheme {
            self.set_control_scheme(selected_control_scheme);
        }

        let mut selected_info_panel = self.info_panel_mode;
        egui::ComboBox::from_label("Info Panel")
            .selected_text(selected_info_panel.label())
            .show_ui(ui, |ui| {
                for mode in [
                    InfoPanelMode::Full,
                    InfoPanelMode::VectorTable,
                    InfoPanelMode::VectorTable2,
                    InfoPanelMode::Off,
                ] {
                    ui.selectable_value(&mut selected_info_panel, mode, mode.label());
                }
            });
        self.info_panel_mode = selected_info_panel;

        ui.separator();

        let mut selected_preview = self.placement_preview_mode;
        egui::ComboBox::from_label("Placement Preview")
            .selected_text(selected_preview.label())
            .show_ui(ui, |ui| {
                for mode in [
                    PlacementPreviewMode::Ghost,
                    PlacementPreviewMode::Wireframe,
                    PlacementPreviewMode::Off,
                ] {
                    ui.selectable_value(&mut selected_preview, mode, mode.label());
                }
            });
        self.placement_preview_mode = selected_preview;
        if self.placement_preview_mode != PlacementPreviewMode::Off {
            ui.checkbox(
                &mut self.placement_preview_hide_camera_intersect,
                "Hide preview when inside camera",
            );
            ui.checkbox(
                &mut self.placement_preview_hide_same_scale,
                "Hide preview at same scale as target",
            );
        }

        ui.separator();

        {
            let blocks: Vec<_> = self.content_registry.all_blocks_ordered().collect();
            let current_key = (
                self.selected_block.namespace,
                self.selected_block.block_type,
            );
            let mut block_idx = blocks
                .iter()
                .position(|b| (b.namespace, b.block_type) == current_key)
                .unwrap_or(0) as u32;
            let max_idx = blocks.len().saturating_sub(1) as u32;
            let response =
                ui.add(egui::Slider::new(&mut block_idx, 0..=max_idx).text("Place Block"));
            if response.changed() {
                if let Some(entry) = blocks.get(block_idx as usize) {
                    self.selected_block = polychora::shared::voxel::BlockData::simple(
                        entry.namespace,
                        entry.block_type,
                    );
                    self.inventory.set_slot(
                        self.hotbar_selected_index,
                        Some(polychora::shared::protocol::ItemStack::block(
                            entry.namespace,
                            entry.block_type,
                            1,
                            0,
                        )),
                    );
                    self.inventory_dirty = true;
                }
            }
        }

        ui.separator();

        ui.add(
            egui::Slider::new(&mut self.audio.master_volume, 0.0..=2.0)
                .text("Master Volume")
                .step_by(0.05),
        );
        ui.add(
            egui::Slider::new(
                &mut self.audio.spatial_falloff_power,
                AUDIO_SPATIAL_FALLOFF_POWER_MIN..=AUDIO_SPATIAL_FALLOFF_POWER_MAX,
            )
            .text("Spatial Falloff (1/r^N)")
            .step_by(0.05),
        );
        self.audio.spatial_falloff_power = self.audio.spatial_falloff_power.clamp(
            AUDIO_SPATIAL_FALLOFF_POWER_MIN,
            AUDIO_SPATIAL_FALLOFF_POWER_MAX,
        );
    }

    fn draw_settings_rendering(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(
                &mut self.focal_length_xy,
                FOCAL_LENGTH_MIN..=FOCAL_LENGTH_MAX,
            )
            .text("Focal Length XY"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.focal_length_zw,
                FOCAL_LENGTH_MIN..=FOCAL_LENGTH_MAX,
            )
            .text("Focal Length ZW"),
        );
        ui.checkbox(
            &mut self.zw_angle_color_shift_enabled,
            "ZW Angle Red/Blue Shift",
        );
        ui.add(
            egui::Slider::new(
                &mut self.zw_angle_color_shift_strength,
                ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN..=ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
            )
            .text("ZW Shift Strength"),
        );
        self.zw_angle_color_shift_strength = self.zw_angle_color_shift_strength.clamp(
            ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN,
            ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
        );

        ui.separator();

        ui.add(
            egui::Slider::new(
                &mut self.vte_max_trace_steps,
                VTE_TRACE_STEPS_MIN..=VTE_TRACE_STEPS_MAX,
            )
            .logarithmic(true)
            .text("Max Trace Steps"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.vte_max_trace_distance,
                VTE_TRACE_DISTANCE_MIN..=VTE_TRACE_DISTANCE_MAX,
            )
            .text("Max Trace Distance"),
        );
        self.vte_max_trace_distance = self
            .vte_max_trace_distance
            .clamp(VTE_TRACE_DISTANCE_MIN, VTE_TRACE_DISTANCE_MAX);

        ui.separator();
        ui.label("Render Resolution");
        ui.horizontal(|ui| {
            ui.label("Width:");
            ui.add(
                egui::DragValue::new(&mut self.pending_render_width)
                    .range(128..=3840)
                    .speed(16),
            );
            ui.label("Height:");
            ui.add(
                egui::DragValue::new(&mut self.pending_render_height)
                    .range(128..=2160)
                    .speed(16),
            );
            ui.label("Layers:");
            ui.add(
                egui::DragValue::new(&mut self.pending_render_layers)
                    .range(1..=512)
                    .speed(1),
            );
        });
        let dims_changed = self.pending_render_width != self.args.width
            || self.pending_render_height != self.args.height
            || self.pending_render_layers != self.args.layers;
        ui.horizontal(|ui| {
            let apply_btn = ui.add_enabled(dims_changed, egui::Button::new("Apply Resolution"));
            if apply_btn.clicked() {
                self.args.width = self.pending_render_width;
                self.args.height = self.pending_render_height;
                self.args.layers = self.pending_render_layers;
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.recreate_sized_buffers(
                        [self.args.width, self.args.height, self.args.layers],
                        None,
                    );
                }
            }
            if dims_changed {
                ui.label(format!(
                    "(current: {}x{}x{})",
                    self.args.width, self.args.height, self.args.layers
                ));
            }
        });
    }

    fn draw_settings_advanced(&mut self, ui: &mut egui::Ui) {
        let mut sky_emissive_tweak = self.vte_integral_sky_emissive_enabled;
        if ui
            .checkbox(&mut sky_emissive_tweak, "Integral Sky+Emissive Tweak")
            .changed()
        {
            self.toggle_vte_integral_sky_emissive();
        }
        ui.add(
            egui::Slider::new(
                &mut self.vte_integral_sky_scale,
                VTE_INTEGRAL_SKY_SCALE_MIN..=VTE_INTEGRAL_SKY_SCALE_MAX,
            )
            .text("Sky Scale"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.vte_integral_hit_emissive_boost,
                VTE_INTEGRAL_HIT_EMISSIVE_MIN..=VTE_INTEGRAL_HIT_EMISSIVE_MAX,
            )
            .text("Hit Emissive"),
        );
        let mut log_merge_tweak = self.vte_integral_log_merge_enabled;
        if ui
            .checkbox(&mut log_merge_tweak, "Integral Log Merge")
            .changed()
        {
            self.toggle_vte_integral_log_merge();
        }
        ui.add(
            egui::Slider::new(
                &mut self.vte_integral_log_merge_k,
                VTE_INTEGRAL_LOG_MERGE_K_MIN..=VTE_INTEGRAL_LOG_MERGE_K_MAX,
            )
            .text("Log-Merge K"),
        );
    }

    fn draw_settings_debug(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Region-Tree Bounds").strong());
        ui.checkbox(
            &mut self.multiplayer_stream_tree_diag_enabled,
            "Render stream tree bounds",
        );
        ui.checkbox(
            &mut self.multiplayer_stream_tree_compare_diag_enabled,
            "Render stream/world mismatch bounds",
        );
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut self.multiplayer_stream_tree_diag_labels_enabled,
                "Labels",
            );
            ui.checkbox(
                &mut self.multiplayer_stream_tree_diag_non_empty_only,
                "Non-empty only",
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(
                &mut self.multiplayer_stream_tree_diag_show_branch_bounds,
                "Branch",
            );
            ui.checkbox(
                &mut self.multiplayer_stream_tree_diag_show_uniform_bounds,
                "Uniform",
            );
            ui.checkbox(
                &mut self.multiplayer_stream_tree_diag_show_chunk_array_bounds,
                "ChunkArray",
            );
            ui.checkbox(
                &mut self.multiplayer_stream_tree_diag_show_procedural_bounds,
                "Procedural",
            );
            ui.checkbox(
                &mut self.multiplayer_stream_tree_diag_show_empty_bounds,
                "Empty",
            );
        });
        ui.checkbox(
            &mut self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled,
            "Render sample-ray BVH node bounds",
        );
        ui.add(
            egui::Slider::new(&mut self.multiplayer_stream_tree_diag_max_nodes, 1..=4096)
                .text("Bounds max nodes"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.multiplayer_stream_tree_diag_sample_ray_max_nodes,
                1..=512,
            )
            .text("Sample-ray max nodes"),
        );
        ui.add(
            egui::Slider::new(&mut self.multiplayer_stream_tree_diag_max_labels, 1..=512)
                .text("Label max count"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.multiplayer_stream_tree_compare_diag_max_chunks,
                1..=4096,
            )
            .text("Mismatch sample cap"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.multiplayer_stream_tree_compare_diag_log_interval,
                1..=1200,
            )
            .text("Mismatch log interval (frames)"),
        );

        ui.add_space(8.0);
        ui.label(RichText::new("Tree Dumps").strong());
        if ui.button("Dump world + render trees to stderr").clicked() {
            self.scene.dump_world_tree();
            self.scene.dump_render_trees();
            eprintln!("--- tree dump complete ---");
        }
    }

    #[allow(dead_code)]
    pub(super) fn draw_egui_hotbar(&self, ctx: &egui::Context) {
        let screen_rect = ctx.content_rect();
        let slot_size = 80.0;
        let gap = 6.5;
        let total_width = 9.0 * slot_size + 8.0 * gap;
        let start_x = (screen_rect.width() - total_width) / 2.0;
        let start_y = screen_rect.height() - slot_size - 10.0;

        egui::Area::new(egui::Id::new("hotbar"))
            .fixed_pos(egui::pos2(start_x, start_y))
            .interactable(false)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(gap, 0.0);
                    for i in 0..9 {
                        let is_selected = i == self.hotbar_selected_index;
                        let slot = self.inventory.slot(i);

                        let (rect, _response) = ui.allocate_exact_size(
                            egui::vec2(slot_size, slot_size),
                            egui::Sense::hover(),
                        );

                        // Background
                        let bg_color = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160);
                        ui.painter().rect_filled(rect, 3.0, bg_color);

                        let icon_rect = rect.shrink(5.0);

                        // Render icon and resolve item name
                        let item_name = if let Some(stack) = slot {
                            let tex = self
                                .content_registry
                                .resolve_item_thumbnail_texture(&stack.item);
                            let fallback = self
                                .content_registry
                                .item_color(stack.item.namespace, stack.item.item_type);
                            self.paint_icon(ui.painter(), icon_rect, tex, fallback);

                            // Scale badge (bottom-left) for block items
                            if let Some(block) = stack.to_block_data() {
                                if block.scale_exp != 0 {
                                    let badge_pos = rect.left_bottom() + egui::vec2(4.0, -3.0);
                                    ui.painter().text(
                                        badge_pos,
                                        egui::Align2::LEFT_BOTTOM,
                                        format!("s{}", block.scale_exp),
                                        egui::FontId::proportional(12.0),
                                        egui::Color32::from_rgb(140, 200, 255),
                                    );
                                }
                                self.content_registry
                                    .block_name(block.namespace, block.block_type)
                            } else if let Some((ens, etype)) = stack.spawn_egg_entity_key() {
                                self.content_registry
                                    .entity_lookup(ens, etype)
                                    .map(|e| e.canonical_name.as_str())
                                    .unwrap_or("???")
                            } else {
                                self.content_registry
                                    .item_name(stack.item.namespace, stack.item.item_type)
                            }
                        } else {
                            ""
                        };

                        // Selection border
                        if is_selected {
                            ui.painter().rect_stroke(
                                rect,
                                3.0,
                                egui::Stroke::new(3.0, egui::Color32::from_rgb(255, 255, 100)),
                                egui::epaint::StrokeKind::Outside,
                            );
                        } else {
                            ui.painter().rect_stroke(
                                rect,
                                3.0,
                                egui::Stroke::new(
                                    1.3,
                                    egui::Color32::from_rgba_unmultiplied(200, 200, 200, 80),
                                ),
                                egui::epaint::StrokeKind::Outside,
                            );
                        }

                        // Slot number label (top-left corner)
                        let label_pos = rect.left_top() + egui::vec2(4.0, 1.3);
                        ui.painter().text(
                            label_pos,
                            egui::Align2::LEFT_TOP,
                            format!("{}", i + 1),
                            egui::FontId::proportional(13.0),
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180),
                        );

                        // Stack count badge (bottom-right)
                        if let Some(stack) = slot {
                            if stack.count > 1 {
                                let badge_pos = rect.right_bottom() + egui::vec2(-4.0, -3.0);
                                ui.painter().text(
                                    badge_pos,
                                    egui::Align2::RIGHT_BOTTOM,
                                    format!("{}", stack.count),
                                    egui::FontId::proportional(12.0),
                                    egui::Color32::WHITE,
                                );
                            }
                        }

                        // Item name (bottom center, small text)
                        let label_pos = rect.center_bottom() + egui::vec2(0.0, -3.0);
                        ui.painter().text(
                            label_pos,
                            egui::Align2::CENTER_BOTTOM,
                            item_name,
                            egui::FontId::proportional(10.0),
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200),
                        );
                    }
                });
            });
    }

    pub(super) fn draw_egui_orientation_indicator(&mut self, ctx: &egui::Context) {
        use polychora::shared::voxel::TesseractOrientation;

        let screen_rect = ctx.content_rect();
        let slot_size = 80.0;
        let gap = 6.5;
        let total_hotbar_width = 9.0 * slot_size + 8.0 * gap;
        let hotbar_start_x = (screen_rect.width() - total_hotbar_width) / 2.0;
        let hotbar_start_y = screen_rect.height() - slot_size - 10.0;

        // Prefer the left edge of the hotbar on wide windows. On narrower
        // windows, lift the controls above the hotbar so they remain usable
        // and do not collide with the Aetna hotbar.
        let widget_width = 148.0;
        let widget_height = 52.0;
        let left_of_hotbar_x = hotbar_start_x - widget_width - 8.0;
        let (widget_x, widget_y) = if left_of_hotbar_x >= 10.0 {
            (
                left_of_hotbar_x,
                hotbar_start_y + (slot_size - widget_height) / 2.0,
            )
        } else {
            (
                ((screen_rect.width() - widget_width) * 0.5).max(10.0),
                (hotbar_start_y - widget_height - 8.0).max(10.0),
            )
        };

        let is_rotated = self.placement_orientation != TesseractOrientation::IDENTITY;

        egui::Area::new(egui::Id::new("orientation_indicator"))
            .fixed_pos(egui::pos2(widget_x, widget_y))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(widget_width, widget_height),
                    egui::Sense::hover(),
                );

                // Background — tinted when orientation is non-identity
                let bg = if is_rotated {
                    egui::Color32::from_rgba_unmultiplied(40, 30, 80, 160)
                } else {
                    egui::Color32::from_rgba_unmultiplied(0, 0, 0, 140)
                };
                ui.painter().rect_filled(rect, 4.0, bg);

                let btn_size = egui::vec2(28.0, 20.0);
                let btn_font = egui::FontId::proportional(11.0);
                let text_color = egui::Color32::from_rgba_unmultiplied(200, 200, 200, 200);
                let hover_color = egui::Color32::from_rgb(255, 255, 100);

                // Row layout constants
                let row0_y = rect.top() + 2.0;
                let row1_y = rect.top() + 26.0;
                let col_start = rect.left() + 3.0;
                let col_gap = 30.0;

                // Top row: XZ, YZ, XW (primary — matching Z/X/C keys)
                let top_buttons: [(&str, &str, TesseractOrientation); 3] = [
                    (
                        "XZ",
                        "Z key: rotate in XZ plane (yaw)",
                        TesseractOrientation::ROT_XZ,
                    ),
                    (
                        "YZ",
                        "X key: rotate in YZ plane (pitch)",
                        TesseractOrientation::ROT_YZ,
                    ),
                    (
                        "XW",
                        "C key: rotate in XW plane (4D)",
                        TesseractOrientation::ROT_XW,
                    ),
                ];
                for (i, (label, tooltip, rot)) in top_buttons.iter().enumerate() {
                    let btn_rect = egui::Rect::from_min_size(
                        egui::pos2(col_start + i as f32 * col_gap, row0_y),
                        btn_size,
                    );
                    let resp = ui.interact(
                        btn_rect,
                        egui::Id::new(format!("rot_{}", label)),
                        egui::Sense::click(),
                    );
                    let color = if resp.hovered() {
                        hover_color
                    } else {
                        text_color
                    };
                    ui.painter().text(
                        btn_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        btn_font.clone(),
                        color,
                    );
                    if resp.clicked() {
                        self.placement_orientation = rot.compose(self.placement_orientation);
                    }
                    resp.on_hover_text(*tooltip);
                }

                // Bottom row: XY, YW, ZW (secondary — UI only)
                let bottom_buttons: [(&str, &str, TesseractOrientation); 3] = [
                    ("XY", "Rotate in XY plane", TesseractOrientation::ROT_XY),
                    ("YW", "Rotate in YW plane", TesseractOrientation::ROT_YW),
                    ("ZW", "Rotate in ZW plane", TesseractOrientation::ROT_ZW),
                ];
                for (i, (label, tooltip, rot)) in bottom_buttons.iter().enumerate() {
                    let btn_rect = egui::Rect::from_min_size(
                        egui::pos2(col_start + i as f32 * col_gap, row1_y),
                        btn_size,
                    );
                    let resp = ui.interact(
                        btn_rect,
                        egui::Id::new(format!("rot_{}", label)),
                        egui::Sense::click(),
                    );
                    let color = if resp.hovered() {
                        hover_color
                    } else {
                        text_color
                    };
                    ui.painter().text(
                        btn_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        btn_font.clone(),
                        color,
                    );
                    if resp.clicked() {
                        self.placement_orientation = rot.compose(self.placement_orientation);
                    }
                    resp.on_hover_text(*tooltip);
                }

                // Reset button
                let reset_rect = egui::Rect::from_min_size(
                    egui::pos2(col_start + 3.0 * col_gap, row0_y),
                    egui::vec2(28.0, 20.0),
                );
                let reset_resp =
                    ui.interact(reset_rect, egui::Id::new("rot_reset"), egui::Sense::click());
                let reset_color = if reset_resp.hovered() {
                    hover_color
                } else {
                    text_color
                };
                ui.painter().text(
                    reset_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "\u{21ba}",
                    egui::FontId::proportional(16.0),
                    reset_color,
                );
                if reset_resp.clicked() {
                    self.placement_orientation = TesseractOrientation::IDENTITY;
                }
                reset_resp.on_hover_text("Reset orientation to identity");

                // Orientation value display
                let label_text = if is_rotated {
                    format!("Ori: {}", self.placement_orientation.0)
                } else {
                    "Ori: 0".to_string()
                };
                let label_color = if is_rotated {
                    egui::Color32::from_rgb(180, 160, 255)
                } else {
                    egui::Color32::from_rgba_unmultiplied(160, 160, 160, 180)
                };
                ui.painter().text(
                    egui::pos2(col_start + 3.0 * col_gap + 14.0, row1_y + 10.0),
                    egui::Align2::CENTER_CENTER,
                    &label_text,
                    egui::FontId::proportional(10.0),
                    label_color,
                );
            });
    }

    pub(super) fn draw_egui_teleport_dialog(
        &mut self,
        ctx: &egui::Context,
        teleport_target: &mut Option<[f32; 4]>,
        close_teleport: &mut bool,
    ) {
        let mut open = true;
        egui::Window::new("Teleport")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.label("Coordinates:");
                let labels = ["X:", "Y:", "Z:", "W:"];
                for (i, label) in labels.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(*label);
                        ui.add(
                            egui::TextEdit::singleline(&mut self.teleport_coords[i])
                                .desired_width(120.0),
                        );
                    });
                }

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    if ui.button("Teleport").clicked() {
                        let parsed: Option<[f32; 4]> = (|| {
                            Some([
                                self.teleport_coords[0].parse().ok()?,
                                self.teleport_coords[1].parse().ok()?,
                                self.teleport_coords[2].parse().ok()?,
                                self.teleport_coords[3].parse().ok()?,
                            ])
                        })();
                        if let Some(pos) = parsed {
                            *teleport_target = Some(pos);
                        }
                    }

                    if ui.button("Go to Origin").clicked() {
                        *teleport_target = Some([0.0, 0.0, 0.0, 0.0]);
                    }
                });

                if self.multiplayer.is_some() && !self.remote_players.is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label("Players:");
                    let mut sorted_ids: Vec<u64> = self.remote_players.keys().copied().collect();
                    sorted_ids.sort();
                    for entity_id in sorted_ids {
                        if let Some(player) = self.remote_players.get(&entity_id) {
                            let name = if player.name.is_empty() {
                                player
                                    .owner_client_id
                                    .map(|id| format!("Player {}", id))
                                    .unwrap_or_else(|| format!("Entity {}", entity_id))
                            } else {
                                player.name.clone()
                            };
                            let pos = player.position;
                            let label = format!(
                                "{} ({:.1}, {:.1}, {:.1}, {:.1})",
                                name, pos[0], pos[1], pos[2], pos[3],
                            );
                            if ui.button(label).clicked() {
                                *teleport_target = Some(pos);
                            }
                        }
                    }
                }
            });
        if !open {
            *close_teleport = true;
        }
    }

    pub(super) fn draw_egui_inventory(
        &mut self,
        ctx: &egui::Context,
        close_inventory: &mut bool,
        inventory_pick: &mut Option<polychora::shared::protocol::ItemStack>,
    ) {
        use polychora::shared::inventory::InventoryTab;

        let mut open = true;
        egui::Window::new("Inventory")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .open(&mut open)
            .default_width(676.0)
            .show(ctx, |ui| {
                // Tab bar: Creative | Survival
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.inventory_tab == InventoryTab::Creative, "Creative")
                        .clicked()
                    {
                        self.inventory_tab = InventoryTab::Creative;
                    }
                    if ui
                        .selectable_label(self.inventory_tab == InventoryTab::Survival, "Survival")
                        .clicked()
                    {
                        self.inventory_tab = InventoryTab::Survival;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new("Tab to switch")
                                .small()
                                .color(egui::Color32::from_rgb(140, 140, 140)),
                        );
                    });
                });
                ui.separator();

                match self.inventory_tab {
                    InventoryTab::Creative => {
                        self.draw_creative_inventory_tab(ui, inventory_pick);
                    }
                    InventoryTab::Survival => {
                        self.draw_survival_inventory_tab(ui);
                    }
                }
            });

        if !open {
            *close_inventory = true;
        }
    }

    fn draw_creative_inventory_tab(
        &self,
        ui: &mut egui::Ui,
        inventory_pick: &mut Option<polychora::shared::protocol::ItemStack>,
    ) {
        // Category tabs
        ui.horizontal(|ui| {
            for cat in polychora_plugin_api::block::BlockCategory::ALL {
                if ui.selectable_label(false, cat.label()).clicked() {
                    // Future: could filter by category.
                }
            }
            ui.label("|");
            ui.label("All");
        });
        ui.separator();

        // Material grid
        let items_per_row = 10;
        let cell_size = 74.0;
        let cell_gap = 5.0;

        egui::ScrollArea::vertical()
            .max_height(416.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(cell_gap, cell_gap);
                    for (idx, entry) in self.content_registry.all_blocks_ordered().enumerate() {
                        if idx > 0 && idx % items_per_row == 0 {
                            ui.end_row();
                        }
                        let block_key = (entry.namespace, entry.block_type);

                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(cell_size, cell_size),
                            egui::Sense::click(),
                        );

                        // Background
                        let bg = if response.hovered() {
                            egui::Color32::from_rgba_unmultiplied(80, 80, 80, 200)
                        } else {
                            egui::Color32::from_rgba_unmultiplied(40, 40, 40, 200)
                        };
                        ui.painter().rect_filled(rect, 3.0, bg);

                        // Material icon (tesseract image or color fallback)
                        let icon_rect = rect.shrink(4.0);
                        self.paint_icon(ui.painter(), icon_rect, Some(entry.texture), entry.color);

                        // Material name
                        let text_pos = egui::pos2(rect.center().x, rect.bottom() - 3.0);
                        ui.painter().text(
                            text_pos,
                            egui::Align2::CENTER_BOTTOM,
                            &entry.name,
                            egui::FontId::proportional(10.0),
                            egui::Color32::from_rgba_unmultiplied(220, 220, 220, 255),
                        );
                        let category_pos = egui::pos2(rect.center().x, rect.top() + 4.0);
                        ui.painter().text(
                            category_pos,
                            egui::Align2::CENTER_TOP,
                            entry.category.label(),
                            egui::FontId::proportional(8.0),
                            egui::Color32::from_rgba_unmultiplied(180, 180, 180, 220),
                        );

                        if response.hovered() {
                            ui.painter().rect_stroke(
                                rect,
                                3.0,
                                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 255, 100)),
                                egui::epaint::StrokeKind::Outside,
                            );
                        }

                        if response.clicked() {
                            *inventory_pick = Some(polychora::shared::protocol::ItemStack::block(
                                block_key.0,
                                block_key.1,
                                1,
                                0,
                            ));
                        }
                    }
                });

                // Spawn Eggs separator
                ui.add_space(8.0);
                ui.separator();
                ui.label("Spawn Eggs");
                ui.add_space(4.0);

                let mut entities: Vec<_> = self.content_registry.spawnable_entities().collect();
                entities.sort_by(|a, b| a.canonical_name.cmp(&b.canonical_name));

                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(cell_gap, cell_gap);
                    for (idx, entity) in entities.iter().enumerate() {
                        if idx > 0 && idx % items_per_row == 0 {
                            ui.end_row();
                        }
                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(cell_size, cell_size),
                            egui::Sense::click(),
                        );

                        // Background
                        let bg = if response.hovered() {
                            egui::Color32::from_rgba_unmultiplied(80, 80, 80, 200)
                        } else {
                            egui::Color32::from_rgba_unmultiplied(40, 40, 40, 200)
                        };
                        ui.painter().rect_filled(rect, 3.0, bg);

                        // Egg icon
                        let egg_tex = if entity.spawn_egg_texture_id != 0 {
                            Some(TextureRef {
                                namespace: 0,
                                texture_id: entity.spawn_egg_texture_id,
                            })
                        } else {
                            None
                        };
                        self.paint_icon(ui.painter(), rect.shrink(8.0), egg_tex, entity.base_color);

                        // Entity name label
                        let text_pos = egui::pos2(rect.center().x, rect.bottom() - 3.0);
                        ui.painter().text(
                            text_pos,
                            egui::Align2::CENTER_BOTTOM,
                            &entity.canonical_name,
                            egui::FontId::proportional(10.0),
                            egui::Color32::from_rgba_unmultiplied(220, 220, 220, 255),
                        );
                        // Category label
                        let category_pos = egui::pos2(rect.center().x, rect.top() + 4.0);
                        ui.painter().text(
                            category_pos,
                            egui::Align2::CENTER_TOP,
                            entity.category.label(),
                            egui::FontId::proportional(8.0),
                            egui::Color32::from_rgba_unmultiplied(180, 180, 180, 220),
                        );

                        if response.hovered() {
                            ui.painter().rect_stroke(
                                rect,
                                3.0,
                                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 255, 100)),
                                egui::epaint::StrokeKind::Outside,
                            );
                        }

                        if response.clicked() {
                            *inventory_pick =
                                Some(polychora::shared::protocol::ItemStack::spawn_egg(
                                    entity.namespace,
                                    entity.entity_type,
                                ));
                        }
                    }
                });
            });

        ui.separator();
        ui.label("Click a material or spawn egg to place it in the selected hotbar slot. Tab or Esc to close.");
    }

    fn draw_survival_inventory_tab(&mut self, ui: &mut egui::Ui) {
        use polychora::shared::inventory::{HOTBAR_SIZE, INVENTORY_COLS};

        let cell_size = 60.0;
        let cell_gap = 4.0;
        let mut clicked_slot: Option<usize> = None;
        let mut right_clicked_slot: Option<usize> = None;

        // Draw main inventory rows 3, 2, 1 (slots 27..36, 18..27, 9..18) top-to-bottom
        for row in (1..4).rev() {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(cell_gap, cell_gap);
                for col in 0..INVENTORY_COLS {
                    let slot_idx = row * INVENTORY_COLS + col;
                    let (left, right) = self.draw_inventory_cell(ui, slot_idx, cell_size, false);
                    if left {
                        clicked_slot = Some(slot_idx);
                    }
                    if right {
                        right_clicked_slot = Some(slot_idx);
                    }
                }
            });
        }

        ui.separator();

        // Draw hotbar row (row 0, slots 0..9) — highlighted
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(cell_gap, cell_gap);
            for col in 0..HOTBAR_SIZE {
                let (left, right) =
                    self.draw_inventory_cell(ui, col, cell_size, col == self.hotbar_selected_index);
                if left {
                    clicked_slot = Some(col);
                }
                if right {
                    right_clicked_slot = Some(col);
                }
            }
        });

        // Handle left-click: swap clicked slot with selected hotbar slot
        if let Some(slot_idx) = clicked_slot {
            if slot_idx != self.hotbar_selected_index {
                self.inventory
                    .swap_slots(slot_idx, self.hotbar_selected_index);
                self.inventory_dirty = true;
            }
            self.selected_block =
                block_data_from_slot(self.inventory.hotbar_slot(self.hotbar_selected_index));
        }

        // Handle right-click: drop one item from clicked slot
        if let Some(slot_idx) = right_clicked_slot {
            if self.inventory.slot(slot_idx).is_some() {
                self.inventory.decrement_slot(slot_idx);
                self.send_drop_item(slot_idx as u8);
                self.send_inventory_sync();
                self.inventory_dirty = true;
                if slot_idx == self.hotbar_selected_index {
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                }
            }
        }

        ui.separator();
        ui.label("Left-click to swap with hotbar. Right-click to drop. Tab or Esc to close.");
    }

    /// Returns (left_clicked, right_clicked).
    fn draw_inventory_cell(
        &self,
        ui: &mut egui::Ui,
        slot_idx: usize,
        cell_size: f32,
        is_selected: bool,
    ) -> (bool, bool) {
        let slot = self.inventory.slot(slot_idx);
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(cell_size, cell_size), egui::Sense::click());
        let clicked = response.clicked();
        let right_clicked = response.secondary_clicked();

        // Background
        let bg = if is_selected {
            egui::Color32::from_rgba_unmultiplied(60, 60, 40, 200)
        } else {
            egui::Color32::from_rgba_unmultiplied(40, 40, 40, 200)
        };
        ui.painter().rect_filled(rect, 3.0, bg);

        if let Some(stack) = slot {
            let icon_rect = rect.shrink(4.0);

            // Unified icon rendering via TextureRef
            let tex = self
                .content_registry
                .resolve_item_thumbnail_texture(&stack.item);
            let fallback = self
                .content_registry
                .item_color(stack.item.namespace, stack.item.item_type);
            self.paint_icon(ui.painter(), icon_rect, tex, fallback);

            // Scale badge (bottom-left) for block items
            if let Some(block_data) = stack.to_block_data() {
                if block_data.scale_exp != 0 {
                    let badge_pos = rect.left_bottom() + egui::vec2(4.0, -3.0);
                    ui.painter().text(
                        badge_pos,
                        egui::Align2::LEFT_BOTTOM,
                        format!("s{}", block_data.scale_exp),
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_rgb(140, 200, 255),
                    );
                }
            }

            // Count badge (bottom-right)
            if stack.count > 1 {
                let badge_pos = rect.right_bottom() + egui::vec2(-4.0, -3.0);
                ui.painter().text(
                    badge_pos,
                    egui::Align2::RIGHT_BOTTOM,
                    format!("{}", stack.count),
                    egui::FontId::proportional(11.0),
                    egui::Color32::WHITE,
                );
            }
        }

        // Border
        if is_selected {
            ui.painter().rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 255, 100)),
                egui::epaint::StrokeKind::Outside,
            );
        } else if response.hovered() {
            ui.painter().rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(200, 200, 100)),
                egui::epaint::StrokeKind::Outside,
            );
        } else {
            ui.painter().rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(200, 200, 200, 60),
                ),
                egui::epaint::StrokeKind::Outside,
            );
        }
        (clicked, right_clicked)
    }

    pub(super) fn draw_egui_block_gui(&mut self, ctx: &egui::Context, close_gui: &mut bool) {
        use polychora_plugin_api::gui_abi::{GuiAction, ItemSlot};

        let Some(session) = &self.block_gui_session else {
            return;
        };

        let cell_size = 54.0;
        let cell_gap = 4.0;
        let mut clicked_slot: Option<u32> = None;
        let held = session.held_slot;
        let mut open = true;

        egui::Window::new(&session.title)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .open(&mut open)
            .show(ctx, |ui| {
                let session = self.block_gui_session.as_ref().unwrap();

                // Draw block GUI slot groups
                for group in &session.slot_groups {
                    if let Some(label) = &group.label {
                        ui.label(label.as_str());
                    }
                    let start = group.slot_start as usize;
                    let end = group.slot_end as usize;
                    let cols = group.columns as usize;

                    let group_slots = &session.slots[start..end.min(session.slots.len())];
                    for row in group_slots.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(cell_gap, cell_gap);
                            for (i, slot) in row.iter().enumerate() {
                                let global_idx = start
                                    + (row.as_ptr() as usize - group_slots.as_ptr() as usize)
                                        / std::mem::size_of::<ItemSlot>()
                                    + i;
                                let is_held = held == Some(global_idx as u32);
                                if self
                                    .draw_block_gui_slot(ui, slot, global_idx, cell_size, is_held)
                                {
                                    clicked_slot = Some(global_idx as u32);
                                }
                            }
                        });
                    }
                }

                // Draw player inventory if requested
                if session.show_player_inventory {
                    ui.separator();
                    ui.label("Inventory");
                    let player_start = session.block_slot_count as usize;
                    let player_slots = &session.slots[player_start..];
                    let cols = 9;
                    for row in player_slots.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(cell_gap, cell_gap);
                            for (i, slot) in row.iter().enumerate() {
                                let global_idx = player_start
                                    + (row.as_ptr() as usize - player_slots.as_ptr() as usize)
                                        / std::mem::size_of::<ItemSlot>()
                                    + i;
                                let is_held = held == Some(global_idx as u32);
                                if self
                                    .draw_block_gui_slot(ui, slot, global_idx, cell_size, is_held)
                                {
                                    clicked_slot = Some(global_idx as u32);
                                }
                            }
                        });
                    }
                }

                ui.separator();
                if held.is_some() {
                    ui.label(
                        egui::RichText::new("Click a slot to place the item. Esc to close.")
                            .small()
                            .color(egui::Color32::from_rgb(200, 200, 100)),
                    );
                } else {
                    ui.label(
                        egui::RichText::new("Click a slot to pick up items. Esc to close.")
                            .small()
                            .color(egui::Color32::from_rgb(140, 140, 140)),
                    );
                }
            });

        if !open {
            *close_gui = true;
        }

        // Handle click logic: pick up / place
        if let Some(slot_idx) = clicked_slot {
            if let Some(held_idx) = held {
                if held_idx == slot_idx {
                    // Clicked the held slot again — deselect
                    if let Some(session) = self.block_gui_session.as_mut() {
                        session.held_slot = None;
                    }
                } else {
                    // Move from held slot to clicked slot
                    let from = held_idx;
                    let count = self
                        .block_gui_session
                        .as_ref()
                        .and_then(|s| s.slots.get(from as usize))
                        .map(|s| s.count)
                        .unwrap_or(0);
                    if count > 0 {
                        let action = GuiAction::MoveStack {
                            from_slot: from,
                            to_slot: slot_idx,
                            count,
                        };
                        if let Some(wasm) = self.wasm_model_manager.as_mut() {
                            if let Some(session) = self.block_gui_session.as_mut() {
                                let accepted =
                                    polychora::block_gui::send_gui_action(wasm, session, action);
                                if accepted {
                                    session.held_slot = None;
                                }
                            }
                        }
                    }
                }
            } else {
                // Pick up from this slot (if non-empty)
                let non_empty = self
                    .block_gui_session
                    .as_ref()
                    .and_then(|s| s.slots.get(slot_idx as usize))
                    .is_some_and(|s| !s.is_empty());
                if non_empty {
                    if let Some(session) = self.block_gui_session.as_mut() {
                        session.held_slot = Some(slot_idx);
                    }
                }
            }
        }
    }

    /// Draw a single slot from the block GUI. Returns true if clicked.
    fn draw_block_gui_slot(
        &self,
        ui: &mut egui::Ui,
        slot: &polychora_plugin_api::gui_abi::ItemSlot,
        _global_idx: usize,
        cell_size: f32,
        is_held: bool,
    ) -> bool {
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(cell_size, cell_size), egui::Sense::click());
        let clicked = response.clicked();

        // Background
        let bg = if is_held {
            egui::Color32::from_rgba_unmultiplied(60, 60, 40, 200)
        } else {
            egui::Color32::from_rgba_unmultiplied(40, 40, 40, 200)
        };
        ui.painter().rect_filled(rect, 3.0, bg);

        if !slot.is_empty() {
            let icon_rect = rect.shrink(4.0);

            // Convert ItemSlot to a temporary Item for thumbnail resolution
            let item = polychora::shared::protocol::Item {
                namespace: slot.item_ns,
                item_type: slot.item_type,
                data: slot.data.clone(),
            };
            let tex = self.content_registry.resolve_item_thumbnail_texture(&item);
            let fallback = self
                .content_registry
                .item_color(slot.item_ns, slot.item_type);
            self.paint_icon(ui.painter(), icon_rect, tex, fallback);

            // Count badge
            if slot.count > 1 {
                let badge_pos = rect.right_bottom() + egui::vec2(-4.0, -3.0);
                ui.painter().text(
                    badge_pos,
                    egui::Align2::RIGHT_BOTTOM,
                    format!("{}", slot.count),
                    egui::FontId::proportional(11.0),
                    egui::Color32::WHITE,
                );
            }
        }

        // Border
        if is_held {
            ui.painter().rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 255, 100)),
                egui::epaint::StrokeKind::Outside,
            );
        } else if response.hovered() {
            ui.painter().rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(200, 200, 100)),
                egui::epaint::StrokeKind::Outside,
            );
        } else {
            ui.painter().rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(200, 200, 200, 60),
                ),
                egui::epaint::StrokeKind::Outside,
            );
        }

        clicked
    }

    pub(super) fn draw_egui_controls_dialog(&mut self, ctx: &egui::Context) {
        let mut open = true;
        egui::Window::new("Controls")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Movement");
                egui::Grid::new("controls_movement")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("W / A / S / D").strong());
                        ui.label("Move forward / left / backward / right");
                        ui.end_row();

                        ui.label(egui::RichText::new("Space").strong());
                        ui.label("Jump (double-tap to toggle fly mode)");
                        ui.end_row();

                        ui.label(egui::RichText::new("Shift").strong());
                        ui.label("Descend / Crouch");
                        ui.end_row();

                        ui.label(egui::RichText::new("Q / E").strong());
                        ui.label("Move in 4D (W-axis negative / positive)");
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.heading("Camera");
                egui::Grid::new("controls_camera")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Mouse").strong());
                        ui.label("Look around");
                        ui.end_row();

                        ui.label(egui::RichText::new("R (hold)").strong());
                        ui.label("Reset orientation");
                        ui.end_row();

                        ui.label(egui::RichText::new("F (hold)").strong());
                        ui.label("Pull to 3D");
                        ui.end_row();

                        ui.label(egui::RichText::new("G").strong());
                        ui.label("Look at nearest block");
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.heading("Building");
                egui::Grid::new("controls_building")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Left Click").strong());
                        ui.label("Break block");
                        ui.end_row();

                        ui.label(egui::RichText::new("Right Click").strong());
                        ui.label("Place block");
                        ui.end_row();

                        ui.label(egui::RichText::new("Middle Click").strong());
                        ui.label("Pick material (copies orientation & scale)");
                        ui.end_row();

                        ui.label(egui::RichText::new("[ / ]").strong());
                        ui.label("Scale down / up");
                        ui.end_row();

                        ui.label(egui::RichText::new("Z / X / C").strong());
                        ui.label("Rotate block: XZ (yaw) / YZ (pitch) / XW (4D)");
                        ui.end_row();

                        ui.label(egui::RichText::new("Scroll Wheel").strong());
                        ui.label("Cycle hotbar slot");
                        ui.end_row();

                        ui.label(egui::RichText::new("1-9, 0").strong());
                        ui.label("Select hotbar slot");
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.heading("UI");
                egui::Grid::new("controls_ui")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Escape").strong());
                        ui.label("Open / close menu");
                        ui.end_row();

                        ui.label(egui::RichText::new("Tab / I").strong());
                        ui.label("Toggle inventory");
                        ui.end_row();

                        ui.label(egui::RichText::new("T").strong());
                        ui.label("Toggle teleport dialog");
                        ui.end_row();

                        ui.label(egui::RichText::new("`").strong());
                        ui.label("Toggle developer console");
                        ui.end_row();
                    });
            });

        if !open {
            self.controls_dialog_open = false;
        }
    }

    pub(super) fn run_egui_frame(&mut self) -> Option<EguiPaintData> {
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone())?;
        let raw_input = self.egui_winit_state.as_mut()?.take_egui_input(&window);

        let egui_ctx = self.egui_ctx.clone();
        let mut close_menu = false;
        let mut close_inventory = false;
        let mut inventory_pick: Option<polychora::shared::protocol::ItemStack> = None;
        let mut teleport_target: Option<[f32; 4]> = None;
        let mut close_teleport = false;
        let mut close_console = false;
        let mut close_block_gui = false;
        let mut console_command: Option<String> = None;
        let mut transition_to_playing: Option<MainMenuTransition> = None;
        let mut return_to_main_menu = false;
        let full_output = egui_ctx.run_ui(raw_input, |ui| {
            if self.app_state == AppState::MainMenu {
                self.draw_egui_main_menu(ui.ctx(), &mut transition_to_playing);
            } else if !self.world_ready {
                self.draw_egui_loading_screen(ui);
            } else {
                let ctx = ui.ctx().clone();
                if self.menu_open {
                    self.draw_egui_pause_menu(&ctx, &mut close_menu, &mut return_to_main_menu);
                }
                if self.inventory_open {
                    self.draw_egui_inventory(&ctx, &mut close_inventory, &mut inventory_pick);
                }
                if self.teleport_dialog_open {
                    self.draw_egui_teleport_dialog(&ctx, &mut teleport_target, &mut close_teleport);
                }
                if self.controls_dialog_open {
                    self.draw_egui_controls_dialog(&ctx);
                }
                if self.dev_console_open {
                    self.draw_egui_dev_console(&ctx, &mut console_command, &mut close_console);
                }
                if self.block_gui_session.is_some() {
                    self.draw_egui_block_gui(&ctx, &mut close_block_gui);
                }
                self.draw_egui_orientation_indicator(&ctx);
            }
        });

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = full_output;
        if let Some(egui_state) = self.egui_winit_state.as_mut() {
            egui_state.handle_platform_output(&window, platform_output);
        }

        if let Some(transition) = transition_to_playing {
            self.handle_main_menu_transition(transition, &window);
        }
        if return_to_main_menu {
            self.transition_to_main_menu(&window);
        }
        if close_menu {
            self.menu_open = false;
            self.controls_dialog_open = false;
            self.grab_mouse(&window);
        }
        if close_inventory {
            self.inventory_open = false;
            self.grab_mouse(&window);
        }
        if close_block_gui {
            self.close_block_gui(&window);
        }
        if let Some(stack) = inventory_pick {
            self.inventory
                .set_slot(self.hotbar_selected_index, Some(stack));
            self.inventory_dirty = true;
            self.selected_block =
                block_data_from_slot(self.inventory.hotbar_slot(self.hotbar_selected_index));
        }
        if close_teleport {
            self.teleport_dialog_open = false;
            if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
                self.grab_mouse(&window);
            }
        }
        if let Some(pos) = teleport_target {
            self.camera.position = pos;
            self.teleport_dialog_open = false;
            if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
                self.grab_mouse(&window);
            }
            eprintln!(
                "Teleported to ({:.1}, {:.1}, {:.1}, {:.1})",
                pos[0], pos[1], pos[2], pos[3],
            );
        }
        if close_console {
            self.close_dev_console();
        }
        if let Some(command) = console_command {
            self.execute_dev_console_command(&command);
        }

        let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

        let mut texture_updates = Vec::new();
        for (texture_id, delta) in textures_delta.set {
            if !matches!(texture_id, egui::TextureId::Managed(0)) {
                continue;
            }
            let (size, pixels) = match delta.image {
                egui::ImageData::Color(image) => {
                    let size = [image.size[0] as u32, image.size[1] as u32];
                    let mut pixels = Vec::with_capacity(image.pixels.len() * 4);
                    for pixel in image.pixels.iter() {
                        let [r, g, b, a] = pixel.to_srgba_unmultiplied();
                        pixels.push(r);
                        pixels.push(g);
                        pixels.push(b);
                        pixels.push(a);
                    }
                    (size, pixels)
                }
            };
            texture_updates.push(EguiTextureUpdate {
                size,
                pos: delta.pos.map(|[x, y]| [x as u32, y as u32]),
                pixels,
            });
        }

        let material_icons_tid = self.material_icons_texture_id;
        let mut meshes = Vec::new();
        for clipped in clipped_primitives {
            let egui::epaint::Primitive::Mesh(mesh) = clipped.primitive else {
                continue;
            };
            let texture_slot = if matches!(mesh.texture_id, egui::TextureId::Managed(0)) {
                EguiTextureSlot::EguiAtlas
            } else if Some(mesh.texture_id) == material_icons_tid {
                EguiTextureSlot::MaterialIcons
            } else {
                continue;
            };

            let mut vertices = Vec::with_capacity(mesh.indices.len());
            for &index in &mesh.indices {
                let Some(vertex) = mesh.vertices.get(index as usize) else {
                    continue;
                };
                let [r, g, b, a] = vertex.color.to_srgba_unmultiplied();
                vertices.push(EguiPaintVertex {
                    position_px: [
                        vertex.pos.x * pixels_per_point,
                        vertex.pos.y * pixels_per_point,
                    ],
                    uv: [vertex.uv.x, vertex.uv.y],
                    color: [
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    ],
                });
            }
            if vertices.is_empty() {
                continue;
            }

            meshes.push(EguiPaintMesh {
                clip_rect_px: [
                    clipped.clip_rect.min.x * pixels_per_point,
                    clipped.clip_rect.min.y * pixels_per_point,
                    clipped.clip_rect.max.x * pixels_per_point,
                    clipped.clip_rect.max.y * pixels_per_point,
                ],
                vertices,
                texture_slot,
            });
        }

        Some(EguiPaintData {
            texture_updates,
            meshes,
        })
    }

    #[allow(dead_code)]
    fn draw_egui_waila(&self, ctx: &egui::Context) {
        let target = match &self.waila_target {
            Some(t) => t,
            None => return,
        };

        let screen_rect = ctx.content_rect();
        let panel_x = screen_rect.width() / 2.0;
        let panel_y = 30.0;

        egui::Area::new(egui::Id::new("waila_panel"))
            .fixed_pos(egui::pos2(panel_x, panel_y))
            .pivot(egui::Align2::CENTER_TOP)
            .interactable(false)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_unmultiplied(15, 15, 22, 190))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::symmetric(12, 8))
                    .show(ui, |ui| match target {
                        WailaTarget::Block { coords, block } => {
                            self.draw_waila_block(ui, *coords, block);
                        }
                        WailaTarget::Entity {
                            entity_id,
                            entity_type_ns,
                            entity_type,
                            position,
                            orientation,
                            scale,
                            data,
                            distance,
                        } => {
                            self.draw_waila_entity(
                                ui,
                                *entity_id,
                                *entity_type_ns,
                                *entity_type,
                                *position,
                                *orientation,
                                *scale,
                                data,
                                *distance,
                            );
                        }
                    });
            });
    }

    #[allow(dead_code)]
    fn draw_waila_block(
        &self,
        ui: &mut egui::Ui,
        coords: [i32; 4],
        block: &polychora::shared::voxel::BlockData,
    ) {
        let entry = self
            .content_registry
            .block_entry(block.namespace, block.block_type);
        let name = entry.map(|e| e.name.as_str()).unwrap_or("Unknown");
        let category = entry.map(|e| e.category.label()).unwrap_or("Unknown");
        let [r, g, b] = entry.map(|e| e.color).unwrap_or([128, 128, 128]);

        // Header row: icon + name + category + ID
        ui.horizontal(|ui| {
            let icon_size = 28.0;
            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(icon_size, icon_size), egui::Sense::hover());

            let tex = self
                .content_registry
                .block_icon_texture(block.namespace, block.block_type);
            self.paint_icon(ui.painter(), icon_rect, tex, [r, g, b]);

            ui.label(
                egui::RichText::new(name)
                    .strong()
                    .size(15.0)
                    .color(egui::Color32::from_rgb(240, 240, 240)),
            );
            ui.label(
                egui::RichText::new(format!(
                    "{} ({:#x}:{:#x})",
                    category, block.namespace, block.block_type
                ))
                .size(12.0)
                .color(egui::Color32::from_rgb(160, 160, 170)),
            );
        });

        let info_color = egui::Color32::from_rgb(140, 145, 160);
        let info_size = 11.0;

        // Namespace and type ID (raw values from world data)
        let ns_label = self.content_registry.namespace_label(block.namespace);
        ui.label(
            egui::RichText::new(format!(
                "ns: {:#010x} ({})  type: {:#010x}",
                block.namespace, ns_label, block.block_type
            ))
            .monospace()
            .size(info_size)
            .color(info_color),
        );

        // Coordinates and scale
        let scale_label = if block.scale_exp != 0 {
            format!("  scale: {}", block.scale_exp)
        } else {
            String::new()
        };
        ui.label(
            egui::RichText::new(format!(
                "[{}, {}, {}, {}]{}",
                coords[0], coords[1], coords[2], coords[3], scale_label
            ))
            .monospace()
            .size(info_size)
            .color(info_color),
        );
    }

    #[allow(dead_code)]
    fn draw_waila_entity(
        &self,
        ui: &mut egui::Ui,
        entity_id: u64,
        entity_type_ns: u32,
        entity_type: u32,
        position: [f32; 4],
        orientation: [f32; 4],
        scale: f32,
        data: &[u8],
        distance: f32,
    ) {
        let entry = self
            .content_registry
            .entity_lookup(entity_type_ns, entity_type);
        let canonical_name = entry
            .map(|e| e.canonical_name.as_str())
            .unwrap_or("unknown");
        let category = entry
            .map(|e| format!("{:?}", e.category))
            .unwrap_or_else(|| "Unknown".to_string());
        // Resolve entity's primary texture and color for the icon
        let first_tex = entry.and_then(|e| e.model_textures.first().copied());
        let [r, g, b] = entry.map(|e| e.base_color).unwrap_or([128, 128, 128]);

        // Check if this is a player
        let player_name = self.remote_players.get(&entity_id).map(|p| p.name.clone());

        // Header row: icon + name + category + distance
        ui.horizontal(|ui| {
            let icon_size = 28.0;
            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(icon_size, icon_size), egui::Sense::hover());

            self.paint_icon(ui.painter(), icon_rect, first_tex, [r, g, b]);

            let display_name = if let Some(ref pname) = player_name {
                format!("{} ({})", canonical_name, pname)
            } else {
                canonical_name.to_string()
            };

            ui.label(
                egui::RichText::new(display_name)
                    .strong()
                    .size(15.0)
                    .color(egui::Color32::from_rgb(240, 240, 240)),
            );
            ui.label(
                egui::RichText::new(format!("{} {:.1}m", category, distance))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(160, 160, 170)),
            );
        });

        let info_color = egui::Color32::from_rgb(140, 145, 160);
        let info_size = 11.0;

        // Entity ID + namespace/type (raw values from wire)
        let ns_label = self.content_registry.namespace_label(entity_type_ns);
        ui.label(
            egui::RichText::new(format!(
                "id: {}  ns: {:#010x} ({})  type: {:#010x}",
                entity_id, entity_type_ns, ns_label, entity_type
            ))
            .monospace()
            .size(info_size)
            .color(info_color),
        );

        // Position
        ui.label(
            egui::RichText::new(format!(
                "pos: [{:.1}, {:.1}, {:.1}, {:.1}]",
                position[0], position[1], position[2], position[3]
            ))
            .monospace()
            .size(info_size)
            .color(info_color),
        );

        // Orientation + scale
        ui.label(
            egui::RichText::new(format!(
                "ori: [{:.2}, {:.2}, {:.2}, {:.2}]  scale: {:.2}",
                orientation[0], orientation[1], orientation[2], orientation[3], scale
            ))
            .monospace()
            .size(info_size)
            .color(info_color),
        );

        // Sim config summary
        if let Some(entry) = entry {
            if let Some(ref config) = entry.sim_config {
                ui.label(
                    egui::RichText::new(format!(
                        "{:?}: {:?} spd={:.1}",
                        config.mode, config.locomotion, config.move_speed
                    ))
                    .monospace()
                    .size(info_size)
                    .color(info_color),
                );
            }
        }

        // CBOR data decode
        if let Some(decoded) = format_cbor_for_display(data) {
            ui.label(
                egui::RichText::new(format!("data: {}", decoded))
                    .monospace()
                    .size(info_size)
                    .color(egui::Color32::from_rgb(160, 170, 140)),
            );
        }
    }
}
