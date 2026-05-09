use super::*;
use higher_dimension_playground::render::{
    OVERLAY_EDGE_TAG_PLACE, OVERLAY_EDGE_TAG_REGION_BRANCH, OVERLAY_EDGE_TAG_REGION_CHUNK_ARRAY,
    OVERLAY_EDGE_TAG_REGION_UNIFORM, OVERLAY_EDGE_TAG_TARGET,
};
use polychora::shared::render_tree::{DebugRayBvhNodeHit, DebugRayBvhNodeKind};

fn describe_sample_ray_hit_for_hud(hit: &DebugRayBvhNodeHit) -> String {
    let span = [
        (hit.bounds.max[0].saturating_sub(hit.bounds.min[0])).to_num::<i32>(),
        (hit.bounds.max[1].saturating_sub(hit.bounds.min[1])).to_num::<i32>(),
        (hit.bounds.max[2].saturating_sub(hit.bounds.min[2])).to_num::<i32>(),
        (hit.bounds.max[3].saturating_sub(hit.bounds.min[3])).to_num::<i32>(),
    ];
    match &hit.kind {
        DebugRayBvhNodeKind::Internal => format!(
            "kind=Internal bounds=({:+},{:+},{:+},{:+})->({:+},{:+},{:+},{:+}) span={}x{}x{}x{} t={:.3}",
            hit.bounds.min[0],
            hit.bounds.min[1],
            hit.bounds.min[2],
            hit.bounds.min[3],
            hit.bounds.max[0],
            hit.bounds.max[1],
            hit.bounds.max[2],
            hit.bounds.max[3],
            span[0],
            span[1],
            span[2],
            span[3],
            hit.t_enter,
        ),
        DebugRayBvhNodeKind::LeafUniform { block } => format!(
            "kind=LeafUniform block={}:{} scale={} bounds=({:+},{:+},{:+},{:+})->({:+},{:+},{:+},{:+}) span={}x{}x{}x{} t={:.3}",
            block.namespace, block.block_type,
            block.scale_exp,
            hit.bounds.min[0],
            hit.bounds.min[1],
            hit.bounds.min[2],
            hit.bounds.min[3],
            hit.bounds.max[0],
            hit.bounds.max[1],
            hit.bounds.max[2],
            hit.bounds.max[3],
            span[0],
            span[1],
            span[2],
            span[3],
            hit.t_enter,
        ),
        DebugRayBvhNodeKind::LeafChunkArray { scale_exp } => format!(
            "kind=LeafChunkArray scale={} bounds=({:+},{:+},{:+},{:+})->({:+},{:+},{:+},{:+}) span={}x{}x{}x{} t={:.3}",
            scale_exp,
            hit.bounds.min[0],
            hit.bounds.min[1],
            hit.bounds.min[2],
            hit.bounds.min[3],
            hit.bounds.max[0],
            hit.bounds.max[1],
            hit.bounds.max[2],
            hit.bounds.max[3],
            span[0],
            span[1],
            span[2],
            span[3],
            hit.t_enter,
        ),
    }
}

fn overlay_edge_tag_for_sample_ray_hit(hit: &DebugRayBvhNodeHit) -> u32 {
    match &hit.kind {
        DebugRayBvhNodeKind::Internal => OVERLAY_EDGE_TAG_REGION_BRANCH,
        DebugRayBvhNodeKind::LeafUniform { .. } => OVERLAY_EDGE_TAG_REGION_UNIFORM,
        DebugRayBvhNodeKind::LeafChunkArray { .. } => OVERLAY_EDGE_TAG_REGION_CHUNK_ARRAY,
    }
}

/// Data produced by the scene-targeting phase of the gameplay loop, consumed
/// when building `RenderOptions` and dispatching the frame.
struct SceneTargetingData {
    targets: Option<scene::BlockEditTargets>,
    hud_player_tags: Vec<HudPlayerTag>,
    hud_target_hit_voxel: Option<[i32; 4]>,
    hud_target_hit_face: Option<[i32; 4]>,
    hud_stream_first_node_desc: Option<String>,
    hud_stream_final_solid_leaf_desc: Option<String>,
    custom_overlay_edge_instances: Vec<common::ModelInstance>,
    vte_highlight_hit_min: Option<[f32; 4]>,
    vte_highlight_hit_max: [f32; 4],
    vte_highlight_face_axis: u32,
    vte_highlight_face_sign: i32,
}

impl App {
    /// Check whether the placement preview (ghost or wireframe) should be
    /// suppressed this frame based on user-configured conditions.
    fn should_suppress_placement_preview(
        &self,
        place: &scene::ScaleAwareBlockTarget,
        targets: &scene::BlockEditTargets,
    ) -> bool {
        if self.placement_preview_hide_camera_intersect {
            let wmin = place.world_min();
            let wmax = place.world_max();
            let cam = self.camera.position;
            let inside = (0..4).all(|i| cam[i] >= wmin[i] && cam[i] <= wmax[i]);
            if inside {
                return true;
            }
        }
        if self.placement_preview_hide_same_scale {
            if let Some(hit) = &targets.hit {
                if hit.scale_exp == place.scale_exp {
                    return true;
                }
            }
        }
        false
    }

    /// Process the auto-command queue (press/wait/screenshot commands injected
    /// via CLI `--commands`).  Returns `true` when a screenshot command was
    /// consumed this frame.
    fn process_command_queue(&mut self) -> bool {
        if self.perf_suite_active() {
            return false;
        }
        if self.command_wait_frames > 0 {
            self.command_wait_frames -= 1;
            return false;
        }
        if let Some(cmd) = self.command_queue.pop_front() {
            match cmd {
                AutoCommand::Press(keycode) => {
                    self.inject_key_press(keycode);
                }
                AutoCommand::Wait(n) => {
                    self.command_wait_frames = n;
                }
                AutoCommand::Screenshot => {
                    return true;
                }
            }
        } else if !self.command_queue.is_empty() || self.command_wait_frames > 0 {
            // Still processing commands
        } else if self.args.commands.is_some() && self.args.gpu_screenshot {
            // Commands finished and we're in screenshot mode, exit
            self.should_exit_after_render = true;
        }
        false
    }

    /// Handle keyboard/mouse input when menus are closed: mouse look, scroll
    /// wheel, look-at, orientation resets, hotbar selection, flight/sprint
    /// toggles, inventory, and teleport dialog.  When a menu is open, drains
    /// all gameplay inputs instead.
    fn apply_input_and_camera(&mut self, dt: f32) {
        if self.menu_open
            || self.inventory_open
            || self.teleport_dialog_open
            || self.dev_console_open
            || self.block_gui_session.is_some()
        {
            self.drain_gameplay_inputs_while_menu_open();
            return;
        }

        self.input.take_menu_left();
        self.input.take_menu_right();
        self.input.take_menu_up();
        self.input.take_menu_down();
        self.input.take_menu_activate();

        // VTE sweep / integral toggles
        if self.input.take_vte_sweep() {
            self.toggle_vte_runtime_sweep();
        }
        if self.input.take_vte_integral_sky_emissive_toggle() {
            self.toggle_vte_integral_sky_emissive();
        }
        if self.input.take_vte_integral_log_merge_toggle() {
            self.toggle_vte_integral_log_merge();
        }

        // Scroll wheel
        let scroll_steps = self.input.take_scroll_steps();
        if scroll_steps != 0 {
            if self.control_scheme.uses_scroll_pair_cycle() {
                for _ in 0..scroll_steps.abs() {
                    if scroll_steps > 0 {
                        self.scroll_cycle_pair = self.scroll_cycle_pair.next();
                    } else {
                        self.scroll_cycle_pair = match self.scroll_cycle_pair {
                            RotationPair::Standard => RotationPair::FourD,
                            RotationPair::FourD => RotationPair::Standard,
                            RotationPair::DoubleRotation => RotationPair::Standard,
                        };
                    }
                }
            } else {
                // Scroll wheel cycles hotbar selection.
                for _ in 0..scroll_steps.abs() {
                    if scroll_steps > 0 {
                        self.hotbar_selected_index = (self.hotbar_selected_index + 8) % 9;
                    } else {
                        self.hotbar_selected_index = (self.hotbar_selected_index + 1) % 9;
                    }
                }
                self.selected_block =
                    block_data_from_slot(self.inventory.hotbar_slot(self.hotbar_selected_index));
                eprintln!(
                    "Hotbar slot {} selected: {} ({})",
                    self.hotbar_selected_index + 1,
                    self.selected_block.block_type,
                    self.content_registry.block_name(
                        self.selected_block.namespace,
                        self.selected_block.block_type
                    ),
                );
            }
        }

        // Mouse look
        if self.mouse_grabbed {
            let pair = self.active_rotation_pair();
            let (dx, dy) = self.input.take_mouse_delta();
            if dx.abs() > 0.5 || dy.abs() > 0.5 {
                self.look_at_target = None;
            }
            match self.control_scheme {
                ControlScheme::LookTransport => {
                    if self.input.mouse_forward_held() {
                        self.camera.apply_mouse_look_transport_with_modifiers(
                            dx,
                            dy,
                            MOUSE_SENSITIVITY,
                            self.input.mouse_back_held(),
                            true,
                        );
                    } else {
                        self.camera.apply_mouse_look_transport(
                            dx,
                            dy,
                            MOUSE_SENSITIVITY,
                            self.input.mouse_back_held(),
                        );
                    }
                }
                ControlScheme::TransportUniform => {
                    self.camera.apply_mouse_look_transport_uniform(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        self.input.mouse_back_held(),
                        self.input.mouse_forward_held(),
                    );
                }
                ControlScheme::TransportDecoupled => {
                    self.camera.apply_mouse_look_transport_decoupled(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        self.input.mouse_back_held(),
                        self.input.mouse_forward_held(),
                    );
                }
                ControlScheme::TransportScaled => {
                    self.camera.apply_mouse_look_transport_scaled(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        self.input.mouse_back_held(),
                        self.input.mouse_forward_held(),
                    );
                }
                ControlScheme::RotorFree => {
                    self.camera.apply_mouse_look_rotor(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        self.input.mouse_back_held(),
                        self.input.mouse_forward_held(),
                    );
                }
                ControlScheme::IntuitiveUpright
                | ControlScheme::LegacySideButtonLayers
                | ControlScheme::LegacyScrollCycle => {
                    self.camera.apply_mouse_look_on(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        pair.h_target(),
                        pair.v_target(),
                    );
                }
            }
        } else {
            self.input.take_mouse_delta();
        }

        // Orientation reset / pull-to-3D
        if self.input.reset_orientation_held() || self.input.pull_to_3d_held() {
            let pull_home = self.input.reset_orientation_held();
            self.look_at_target = None;
            match self.control_scheme {
                ControlScheme::LookTransport
                | ControlScheme::TransportUniform
                | ControlScheme::TransportDecoupled
                | ControlScheme::TransportScaled
                | ControlScheme::RotorFree => {
                    if pull_home {
                        self.camera.pull_toward_home_look_frame(dt);
                    } else {
                        self.camera.pull_toward_nearest_3d_look_frame(dt);
                    }
                }
                ControlScheme::IntuitiveUpright
                | ControlScheme::LegacySideButtonLayers
                | ControlScheme::LegacyScrollCycle => {
                    if pull_home {
                        self.camera.pull_toward_home_angles(dt);
                    } else {
                        self.camera.pull_toward_nearest_3d_angles(dt);
                    }
                }
            }
        }

        // Look-at: on G press, fan-cast across the ZW viewing wedge to
        // find the nearest solid block and smoothly rotate toward it.
        if self.input.take_look_at() && self.mouse_grabbed {
            let edit_reach = self
                .args
                .edit_reach
                .clamp(BLOCK_EDIT_REACH_MIN, BLOCK_EDIT_REACH_MAX);
            let (_right, _up, view_z, view_w) = self.current_view_basis();
            let hit = self.scene.fan_cast_nearest_block(
                self.camera.position,
                view_z,
                view_w,
                self.focal_length_zw,
                edit_reach,
                32,
            );
            if let Some([x, y, z, w]) = hit {
                let target_pos = [
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    z as f32 + 0.5,
                    w as f32 + 0.5,
                ];
                let dir = [
                    target_pos[0] - self.camera.position[0],
                    target_pos[1] - self.camera.position[1],
                    target_pos[2] - self.camera.position[2],
                    target_pos[3] - self.camera.position[3],
                ];
                match self.control_scheme {
                    ControlScheme::LookTransport
                    | ControlScheme::TransportUniform
                    | ControlScheme::TransportDecoupled
                    | ControlScheme::TransportScaled
                    | ControlScheme::RotorFree => {
                        self.look_at_target = Some(LookAtTarget::Direction(dir));
                    }
                    _ => {
                        let (ty, tp, txw, tzw) = Camera4D::angles_for_direction_upright(dir);
                        self.look_at_target = Some(LookAtTarget::Angles {
                            yaw: ty,
                            pitch: tp,
                            xw_angle: txw,
                            zw_angle: tzw,
                        });
                    }
                }
            }
        }

        // Apply smooth pull toward look-at target
        if let Some(target) = self.look_at_target {
            let converged = match target {
                LookAtTarget::Angles {
                    yaw,
                    pitch,
                    xw_angle,
                    zw_angle,
                } => self
                    .camera
                    .pull_toward_target_angles(yaw, pitch, xw_angle, zw_angle, dt),
                LookAtTarget::Direction(dir) => {
                    self.camera.pull_toward_target_direction_look_frame(dir, dt)
                }
            };
            if converged {
                self.look_at_target = None;
            }
        }

        // Auto-level / upright constraints
        let pair = self.active_rotation_pair();
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => {
                self.camera.enforce_upright_constraints();
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                if pair != RotationPair::DoubleRotation {
                    self.camera.auto_level(dt);
                }
            }
            ControlScheme::LookTransport
            | ControlScheme::TransportUniform
            | ControlScheme::TransportDecoupled
            | ControlScheme::TransportScaled
            | ControlScheme::RotorFree => {}
        }

        // Toggle flight mode on double-tap space
        if self.input.take_fly_toggle() {
            self.camera.toggle_flying();
        }
        if self.input.take_sprint_toggle() && !self.sprint_enabled {
            self.sprint_enabled = true;
            eprintln!("Sprint: on");
        }

        // Bracket keys adjust placement scale.
        if self.input.take_place_material_prev() {
            let new_scale = (self.selected_block.scale_exp - 1).max(-3);
            self.selected_block.scale_exp = new_scale;
            self.inventory
                .update_slot_scale(self.hotbar_selected_index, new_scale);
            self.inventory_dirty = true;
        }
        if self.input.take_place_material_next() {
            let new_scale = (self.selected_block.scale_exp + 1).min(3);
            self.selected_block.scale_exp = new_scale;
            self.inventory
                .update_slot_scale(self.hotbar_selected_index, new_scale);
            self.inventory_dirty = true;
        }
        // Number keys 1-9 select hotbar slot.
        if let Some(digit) = self.input.take_place_material_digit() {
            if (1..=9).contains(&digit) {
                self.hotbar_selected_index = (digit - 1) as usize;
                self.selected_block =
                    block_data_from_slot(self.inventory.hotbar_slot(self.hotbar_selected_index));
                eprintln!(
                    "Hotbar slot {} selected: {} ({})",
                    digit,
                    self.selected_block.block_type,
                    self.content_registry.block_name(
                        self.selected_block.namespace,
                        self.selected_block.block_type
                    ),
                );
            }
        }
        // I key toggles inventory.
        if self.input.take_inventory_toggle() {
            self.toggle_inventory();
        }
        // Tab toggles inventory open/closed.
        if self.input.take_inventory_tab_cycle() {
            self.toggle_inventory();
        }
        // T key toggles teleport dialog.
        if self.input.take_teleport_dialog() {
            self.toggle_teleport_dialog();
        }
    }

    /// Apply movement, gravity, collision, footstep audio, placement
    /// orientation, and block/entity edit actions.  When menus are open,
    /// drains all gameplay-related inputs instead.
    fn apply_physics_and_editing(&mut self, dt: f32, edit_reach: f32, now: Instant) {
        if self.menu_open
            || self.inventory_open
            || self.teleport_dialog_open
            || self.dev_console_open
        {
            self.input.take_jump();
            self.input.take_remove_block();
            self.input.take_place_block();
            self.input.take_pick_material();
            self.input.take_rotate_xz();
            self.input.take_rotate_yz();
            self.input.take_rotate_xw();
            self.footstep_distance_accum = 0.0;
            self.was_grounded_last_frame = self.camera.is_grounded;
        } else {
            // Jump when in gravity mode, consume jump either way.
            if self.camera.is_flying {
                self.input.take_jump();
            } else if self.input.take_jump() {
                let was_grounded_for_jump = self.camera.is_grounded;
                self.camera.jump();
                if was_grounded_for_jump && !self.camera.is_grounded {
                    self.audio.play(SoundEffect::Jump);
                }
            }

            // Movement (vertical zeroed in gravity mode internally).
            let prev_position = self.camera.position;
            let (forward, strafe, vertical, w_axis) = self.input.movement_axes();
            let has_movement_input = forward.abs() > 1e-6
                || strafe.abs() > 1e-6
                || vertical.abs() > 1e-6
                || w_axis.abs() > 1e-6;
            if self.sprint_enabled && !has_movement_input {
                self.sprint_enabled = false;
                eprintln!("Sprint: off");
            }
            let move_speed = if self.sprint_enabled && forward > 0.0 {
                self.move_speed * SPRINT_SPEED_MULTIPLIER
            } else {
                self.move_speed
            };
            match self.control_scheme {
                ControlScheme::IntuitiveUpright => {
                    self.camera
                        .apply_movement_upright(forward, strafe, vertical, w_axis, dt, move_speed);
                }
                ControlScheme::LookTransport
                | ControlScheme::TransportUniform
                | ControlScheme::TransportDecoupled
                | ControlScheme::TransportScaled
                | ControlScheme::RotorFree => {
                    self.camera.apply_movement_look_frame(
                        forward, strafe, vertical, w_axis, dt, move_speed,
                    );
                }
                ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                    self.camera
                        .apply_movement(forward, strafe, vertical, w_axis, dt, move_speed);
                }
            }

            // Apply gravity physics (no-op while flying), then always resolve voxel collisions.
            self.camera.update_physics(dt);
            if dt > 0.0 {
                let external_velocity = self.player_modifier_external_velocity;
                self.camera.position[0] += external_velocity[0] * dt;
                self.camera.position[2] += external_velocity[2] * dt;
                self.camera.position[3] += external_velocity[3] * dt;
                let decay = (-MULTIPLAYER_PLAYER_MODIFIER_DECAY_HZ * dt.clamp(0.0, 0.25)).exp();
                self.player_modifier_external_velocity[0] *= decay;
                self.player_modifier_external_velocity[2] *= decay;
                self.player_modifier_external_velocity[3] *= decay;
                if self.player_modifier_external_velocity[0].abs() < 1e-3 {
                    self.player_modifier_external_velocity[0] = 0.0;
                }
                if self.player_modifier_external_velocity[2].abs() < 1e-3 {
                    self.player_modifier_external_velocity[2] = 0.0;
                }
                if self.player_modifier_external_velocity[3].abs() < 1e-3 {
                    self.player_modifier_external_velocity[3] = 0.0;
                }
            }
            let (resolved_pos, grounded) = self.scene.resolve_player_collision(
                prev_position,
                self.camera.position,
                &mut self.camera.velocity_y,
            );
            self.camera.position = resolved_pos;
            self.camera.is_grounded = grounded;

            if self.camera.is_grounded && !self.was_grounded_last_frame && !self.camera.is_flying {
                self.audio.play(SoundEffect::Land);
            }
            let moved_dx = self.camera.position[0] - prev_position[0];
            let moved_dz = self.camera.position[2] - prev_position[2];
            let moved_dw = self.camera.position[3] - prev_position[3];
            let moved_xzw =
                (moved_dx * moved_dx + moved_dz * moved_dz + moved_dw * moved_dw).sqrt();
            let moved_speed_xzw = if dt > 1e-5 { moved_xzw / dt } else { 0.0 };
            if self.camera.is_grounded
                && !self.camera.is_flying
                && has_movement_input
                && moved_speed_xzw > FOOTSTEP_MIN_XZW_SPEED
            {
                self.footstep_distance_accum += moved_xzw;
                let stride = if self.sprint_enabled {
                    FOOTSTEP_DISTANCE_SPRINT
                } else {
                    FOOTSTEP_DISTANCE_WALK
                };
                while self.footstep_distance_accum >= stride {
                    let intensity = (moved_speed_xzw / (self.move_speed * SPRINT_SPEED_MULTIPLIER))
                        .clamp(0.65, 1.25);
                    self.audio.play_scaled(SoundEffect::Footstep, intensity);
                    self.footstep_distance_accum -= stride;
                }
            } else {
                self.footstep_distance_accum = 0.0;
            }
            self.was_grounded_last_frame = self.camera.is_grounded;

            // Placement orientation rotation.
            {
                use polychora::shared::voxel::TesseractOrientation;
                if self.input.take_rotate_xz() {
                    self.placement_orientation =
                        TesseractOrientation::ROT_XZ.compose(self.placement_orientation);
                }
                if self.input.take_rotate_yz() {
                    self.placement_orientation =
                        TesseractOrientation::ROT_YZ.compose(self.placement_orientation);
                }
                if self.input.take_rotate_xw() {
                    self.placement_orientation =
                        TesseractOrientation::ROT_XW.compose(self.placement_orientation);
                }
            }

            // Block edit actions.
            let look_dir_for_edit = self.current_look_direction();
            if self.mouse_grabbed {
                let pick_requested = self.input.take_pick_material();
                let remove_requested = self.input.take_remove_block();
                let place_requested = self.input.take_place_block();
                if pick_requested {
                    let pick_targets = self.scene.block_edit_targets(
                        self.camera.position,
                        look_dir_for_edit,
                        edit_reach,
                        self.selected_block.scale_exp,
                    );
                    if let Some(picked) = pick_targets.hit_block {
                        if !picked.is_air() {
                            let origin = pick_targets.hit.unwrap().origin_i32();
                            self.placement_orientation = picked.orientation;
                            self.inventory.set_slot(
                                self.hotbar_selected_index,
                                Some(polychora::shared::protocol::ItemStack::block(
                                    picked.namespace,
                                    picked.block_type,
                                    1,
                                    picked.scale_exp,
                                )),
                            );
                            self.inventory_dirty = true;
                            self.selected_block = picked;
                            eprintln!(
                                "Picked voxel {} ({}) from ({}, {}, {}, {})",
                                self.selected_block.block_type,
                                self.content_registry.block_name(
                                    self.selected_block.namespace,
                                    self.selected_block.block_type
                                ),
                                origin[0],
                                origin[1],
                                origin[2],
                                origin[3],
                            );
                        }
                    }
                }
                if remove_requested || place_requested {
                    let edit_targets = self.scene.block_edit_targets(
                        self.camera.position,
                        look_dir_for_edit,
                        edit_reach,
                        self.selected_block.scale_exp,
                    );
                    if remove_requested {
                        if let Some(hit) = &edit_targets.hit {
                            let [x, y, z, w] = hit.origin_i32();
                            let hit_block = edit_targets.hit_block;
                            let air =
                                polychora::shared::voxel::BlockData::AIR.at_scale(hit.scale_exp);
                            eprintln!(
                                "Removed block at ({x}, {y}, {z}, {w}) scale={}",
                                hit.scale_exp
                            );
                            self.play_spatial_sound_voxel(
                                SoundEffect::Break,
                                hit.origin_i32(),
                                1.0,
                            );
                            self.send_multiplayer_voxel_update(now, hit.origin, air);
                            // Survival: add broken block to inventory
                            if self.game_mode == polychora::shared::inventory::GameMode::Survival {
                                if let Some(broken) = hit_block {
                                    if !broken.is_air() {
                                        let stack = polychora::shared::protocol::ItemStack::block(
                                            broken.namespace,
                                            broken.block_type,
                                            1,
                                            broken.scale_exp,
                                        );
                                        self.inventory.try_add(stack);
                                        self.inventory_dirty = true;
                                    }
                                }
                            }
                        }
                    } else if place_requested {
                        // Check if the hit block is interactable (e.g. chest, spawner).
                        let mut handled_interact = false;
                        if let Some(hit_block) = &edit_targets.hit_block {
                            if self
                                .content_registry
                                .is_block_interactable(hit_block.namespace, hit_block.block_type)
                            {
                                if let Some(hit) = &edit_targets.hit {
                                    let position = hit.origin_i32().map(|c| c as i64);
                                    if let Some(wasm) = self.wasm_model_manager.as_mut() {
                                        let result = polychora::block_gui::try_block_interact(
                                            wasm,
                                            hit_block,
                                            position,
                                            &self.inventory,
                                            self.hotbar_selected_index as u32,
                                        );
                                        match result {
                                            polychora::block_gui::BlockInteractResult::OpenGui(
                                                session,
                                            ) => {
                                                eprintln!(
                                                    "Opened block GUI: {} at ({}, {}, {}, {})",
                                                    session.title,
                                                    position[0],
                                                    position[1],
                                                    position[2],
                                                    position[3],
                                                );
                                                self.block_gui_session = Some(session);
                                                if let Some(window) = self
                                                    .rcx
                                                    .as_ref()
                                                    .and_then(|rcx| rcx.window.clone())
                                                {
                                                    self.release_mouse(&window);
                                                }
                                                handled_interact = true;
                                            }
                                            polychora::block_gui::BlockInteractResult::Handled(
                                                effects,
                                            ) => {
                                                self.process_block_interact_side_effects(
                                                    &effects.side_effects,
                                                    hit.origin,
                                                );
                                                handled_interact = true;
                                            }
                                            polychora::block_gui::BlockInteractResult::Nothing => {}
                                        }
                                    }
                                }
                            }
                        }
                        if handled_interact {
                            // Interaction consumed the right-click — skip placement.
                        } else {
                            // Check if the selected hotbar item is a spawn egg.
                            let egg_key = self
                                .inventory
                                .slot(self.hotbar_selected_index)
                                .and_then(|s| s.spawn_egg_entity_key());
                            if let Some((ens, etype)) = egg_key {
                                let spawn_pos = if let Some(place) = &edit_targets.place {
                                    let o = place.origin_i32();
                                    [
                                        o[0] as f32 + 0.5,
                                        o[1] as f32 + 0.5,
                                        o[2] as f32 + 0.5,
                                        o[3] as f32 + 0.5,
                                    ]
                                } else {
                                    let look = self.current_look_direction();
                                    let p = self.camera.position;
                                    [
                                        p[0] + look[0] * 3.0,
                                        p[1] + look[1] * 3.0,
                                        p[2] + look[2] * 3.0,
                                        p[3] + look[3] * 3.0,
                                    ]
                                };
                                let scale = self
                                    .content_registry
                                    .entity_lookup(ens, etype)
                                    .map(|e| e.default_scale)
                                    .unwrap_or(1.0);
                                let look = self.current_look_direction();
                                self.send_multiplayer_spawn_entity(
                                    ens, etype, spawn_pos, look, scale,
                                );
                                eprintln!(
                                "Spawn egg used: entity ({:#x}, {:#x}) at ({:.1}, {:.1}, {:.1}, {:.1})",
                                ens, etype, spawn_pos[0], spawn_pos[1], spawn_pos[2], spawn_pos[3],
                            );
                            } else if let Some(bp_meta) = self
                                .inventory
                                .slot(self.hotbar_selected_index)
                                .and_then(|s| s.blueprint_meta())
                            {
                                if let Some(place) = &edit_targets.place {
                                    self.place_blueprint(&bp_meta, place.origin);
                                }
                            } else if !self.selected_block.is_air() {
                                if let Some(place) = &edit_targets.place {
                                    let [x, y, z, w] = place.origin_i32();
                                    eprintln!(
                                        "Placed voxel {} ({}) at ({x}, {y}, {z}, {w}) scale={}",
                                        self.selected_block.block_type,
                                        self.content_registry.block_name(
                                            self.selected_block.namespace,
                                            self.selected_block.block_type
                                        ),
                                        place.scale_exp,
                                    );
                                    self.play_spatial_sound_voxel(
                                        SoundEffect::Place,
                                        place.origin_i32(),
                                        1.0,
                                    );
                                    let mut placed_block = self.selected_block.clone();
                                    placed_block.orientation = self.placement_orientation;
                                    self.send_multiplayer_voxel_update(
                                        now,
                                        place.origin,
                                        placed_block,
                                    );
                                    if self.game_mode
                                        == polychora::shared::inventory::GameMode::Survival
                                    {
                                        self.inventory.decrement_slot(self.hotbar_selected_index);
                                        self.inventory_dirty = true;
                                        self.selected_block = block_data_from_slot(
                                            self.inventory.hotbar_slot(self.hotbar_selected_index),
                                        );
                                    }
                                }
                            }
                        }
                    } // close `else { ... }` for handled_interact
                }
            } else {
                self.input.take_remove_block();
                self.input.take_place_block();
                self.input.take_pick_material();
            }
        }

        // Drop item from selected hotbar slot (B key)
        if self.input.take_drop_item() {
            let idx = self.hotbar_selected_index;
            if self.inventory.slot(idx).is_some() {
                self.inventory.decrement_slot(idx);
                self.send_drop_item(idx as u8);
                self.send_inventory_sync();
                self.inventory_dirty = true;
                self.selected_block =
                    block_data_from_slot(self.inventory.hotbar_slot(self.hotbar_selected_index));
            }
        }

        if let Some(scenario_index) = self
            .perf_suite_state
            .as_ref()
            .map(|state| state.scenario_index)
        {
            self.set_perf_suite_camera_pose(scenario_index);
        }
        self.apply_pending_player_movement_modifiers();
    }

    /// Process side effects from OP_BLOCK_INTERACT (client-side).
    ///
    /// Allowed effects: UpdateBlockMetadata, ConsumeHeldItem.
    fn process_block_interact_side_effects(
        &mut self,
        side_effects: &[polychora_plugin_api::side_effects::SideEffect],
        block_position: [polychora::shared::spatial::ChunkCoord; 4],
    ) {
        use polychora_plugin_api::side_effects::SideEffect;
        for effect in side_effects {
            match effect {
                SideEffect::UpdateBlockMetadata { metadata } => {
                    let mut block = self.scene.get_block_data(
                        block_position[0].to_num::<i32>(),
                        block_position[1].to_num::<i32>(),
                        block_position[2].to_num::<i32>(),
                        block_position[3].to_num::<i32>(),
                    );
                    block.extra_data = metadata.clone();
                    self.send_multiplayer_voxel_update(
                        std::time::Instant::now(),
                        block_position,
                        block,
                    );
                }
                SideEffect::ConsumeHeldItem { count } => {
                    let idx = self.hotbar_selected_index;
                    for _ in 0..*count {
                        self.inventory.decrement_slot(idx);
                    }
                    self.inventory_dirty = true;
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                }
                SideEffect::GiveItem {
                    item_ns,
                    item_type,
                    item_data,
                    count,
                } => {
                    use polychora::shared::protocol::{Item, ItemStack};
                    let stack = ItemStack {
                        item: Item {
                            namespace: *item_ns,
                            item_type: *item_type,
                            data: item_data.clone(),
                        },
                        count: *count,
                    };
                    let _remainder = self.inventory.try_add(stack);
                    self.inventory_dirty = true;
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                }
                SideEffect::SpawnEntity { .. } => {
                    eprintln!("Warning: SpawnEntity side effect not allowed for OP_BLOCK_INTERACT");
                }
            }
        }
    }

    /// Place a blueprint structure at the given position.
    fn place_blueprint(
        &mut self,
        bp_meta: &polychora::shared::item_types::BlueprintMeta,
        position: [polychora::shared::spatial::ChunkCoord; 4],
    ) {
        let origin = position.map(|c| c.to_num::<i64>());
        let host_tree = bp_meta.to_host_tree();
        eprintln!(
            "Blueprint: placing tree at ({}, {}, {}, {})",
            origin[0], origin[1], origin[2], origin[3],
        );

        // TODO: orient the tree when player orientation is wired up
        let tree_data = postcard::to_allocvec(&host_tree).expect("postcard serialize tree");
        self.send_set_tree_core(origin, tree_data);
    }

    /// Update WAILA (What Am I Looking At) target based on block and entity
    /// ray-cast results.  Writes `self.waila_target`.
    fn update_waila_target(&mut self, look_dir: [f32; 4], edit_reach: f32) {
        self.waila_target = if !self.menu_open && self.mouse_grabbed && !self.args.no_hud {
            let entity_hit = find_targeted_entity(
                self.camera.position,
                look_dir,
                edit_reach,
                &self.remote_entities,
                &self.remote_players,
            );
            let waila_targets = self.scene.block_edit_targets(
                self.camera.position,
                look_dir,
                edit_reach,
                self.selected_block.scale_exp,
            );
            let block_target = waila_targets.hit.and_then(|hit| {
                let block = waila_targets.hit_block?;
                if block.is_air() {
                    return None;
                }
                let half = hit.size().to_num::<f32>() * 0.5;
                let wmin = hit.world_min();
                let block_center = [
                    wmin[0] + half,
                    wmin[1] + half,
                    wmin[2] + half,
                    wmin[3] + half,
                ];
                let block_dist = distance4(self.camera.position, block_center);
                Some((
                    WailaTarget::Block {
                        coords: hit.origin_i32(),
                        block,
                    },
                    block_dist,
                ))
            });

            match (&entity_hit, &block_target) {
                (Some(eh), Some((_, bd))) if eh.distance < *bd => {
                    if let Some(ent) = self.remote_entities.get(&eh.entity_id) {
                        Some(WailaTarget::Entity {
                            entity_id: eh.entity_id,
                            entity_type_ns: ent.entity_type_ns,
                            entity_type: ent.entity_type,
                            position: ent.render_position,
                            orientation: ent.render_orientation,
                            scale: ent.scale,
                            data: ent.data.clone(),
                            distance: eh.distance,
                        })
                    } else if let Some(player) = self.remote_players.get(&eh.entity_id) {
                        Some(WailaTarget::Entity {
                            entity_id: eh.entity_id,
                            entity_type_ns: 0,
                            entity_type: 0,
                            position: player.render_position,
                            orientation: player.render_look,
                            scale: 1.0,
                            data: Vec::new(),
                            distance: eh.distance,
                        })
                    } else {
                        block_target.map(|(t, _)| t)
                    }
                }
                (Some(eh), None) => {
                    if let Some(ent) = self.remote_entities.get(&eh.entity_id) {
                        Some(WailaTarget::Entity {
                            entity_id: eh.entity_id,
                            entity_type_ns: ent.entity_type_ns,
                            entity_type: ent.entity_type,
                            position: ent.render_position,
                            orientation: ent.render_orientation,
                            scale: ent.scale,
                            data: ent.data.clone(),
                            distance: eh.distance,
                        })
                    } else {
                        self.remote_players
                            .get(&eh.entity_id)
                            .map(|player| WailaTarget::Entity {
                                entity_id: eh.entity_id,
                                entity_type_ns: 0,
                                entity_type: 0,
                                position: player.render_position,
                                orientation: player.render_look,
                                scale: 1.0,
                                data: Vec::new(),
                                distance: eh.distance,
                            })
                    }
                }
                (_, Some((target, _))) => Some(target.clone()),
                (None, None) => None,
            }
        } else {
            None
        };
    }

    /// Build overlay edge instances for block targeting, wireframe placement
    /// preview, BVH debug visualisation, and stream-tree diagnostics.
    fn build_overlay_edges(
        &mut self,
        targets: Option<&scene::BlockEditTargets>,
        highlight_mode: EditHighlightModeArg,
        sample_ray_node_hits: &[DebugRayBvhNodeHit],
    ) -> Vec<common::ModelInstance> {
        let overlay_edge_capacity = 2usize
            .saturating_add(if self.multiplayer_stream_tree_diag_enabled {
                self.multiplayer_stream_tree_diag_max_nodes.max(1)
            } else {
                0
            })
            .saturating_add(
                if self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled {
                    self.multiplayer_stream_tree_diag_sample_ray_max_nodes
                        .max(1)
                } else {
                    0
                },
            )
            .saturating_add(if self.multiplayer_stream_tree_compare_diag_enabled {
                self.multiplayer_stream_tree_compare_diag_max_chunks
                    .saturating_mul(2)
            } else {
                0
            });
        let mut out = Vec::with_capacity(overlay_edge_capacity);
        if highlight_mode.uses_edges() {
            if let Some(targets) = targets {
                if let Some(hit) = &targets.hit {
                    append_axis_aligned_outline_edge_instance(
                        &mut out,
                        hit.world_min(),
                        hit.world_max(),
                        OVERLAY_EDGE_TAG_TARGET,
                    );
                }
                if let Some(place) = &targets.place {
                    append_axis_aligned_outline_edge_instance(
                        &mut out,
                        place.world_min(),
                        place.world_max(),
                        OVERLAY_EDGE_TAG_PLACE,
                    );
                }
            }
        }
        // Wireframe placement preview: add outline edges for the placement
        // target independently of the edit-highlight edge mode.
        // Only show when holding a block item.
        if self.placement_preview_mode == PlacementPreviewMode::Wireframe {
            let has_block = self
                .inventory
                .hotbar_slot(self.hotbar_selected_index)
                .as_ref()
                .and_then(|s| s.to_block_data())
                .is_some();
            if has_block {
                if let Some(targets) = targets {
                    if let Some(place) = &targets.place {
                        if !self.should_suppress_placement_preview(place, targets) {
                            append_axis_aligned_outline_edge_instance(
                                &mut out,
                                place.world_min(),
                                place.world_max(),
                                OVERLAY_EDGE_TAG_PLACE,
                            );
                        }
                    }
                }
            }
        }
        if self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled {
            for hit in sample_ray_node_hits {
                append_chunk_bounds_outline_edge_instance(
                    &mut out,
                    hit.bounds.min,
                    hit.bounds.max,
                    overlay_edge_tag_for_sample_ray_hit(hit),
                );
            }
        }
        self.append_multiplayer_stream_tree_diag_overlay_instances(&mut out);
        self.append_multiplayer_stream_tree_compare_overlay_instances(&mut out);
        if self.args.no_hud {
            out.clear();
        }
        out
    }

    /// Compute the VTE face-highlight bounds from the current block targets.
    fn compute_vte_highlight(
        backend: RenderBackend,
        highlight_mode: EditHighlightModeArg,
        targets: Option<&scene::BlockEditTargets>,
    ) -> (Option<[f32; 4]>, [f32; 4], u32, i32) {
        let mut hit_min = None;
        let mut hit_max = [0.0f32; 4];
        let mut face_axis = 0u32;
        let mut face_sign = 0i32;
        if backend == RenderBackend::VoxelTraversal && highlight_mode.uses_faces() {
            if let Some(targets) = targets {
                if let Some(hit) = &targets.hit {
                    hit_min = Some(hit.world_min());
                    hit_max = hit.world_max();
                    face_axis = targets.face_axis as u32;
                    face_sign = targets.face_sign as i32;
                }
            }
        }
        (hit_min, hit_max, face_axis, face_sign)
    }

    /// Prepare all scene-targeting data needed for the render frame: block
    /// edit targets, HUD tag/hit data, sample-ray BVH diagnostics, WAILA,
    /// overlay edges, and VTE face-highlight bounds.
    fn prepare_scene_targeting_data(
        &mut self,
        look_dir: [f32; 4],
        edit_reach: f32,
        view_matrix: &ndarray::Array2<f32>,
        aspect: f32,
        backend: RenderBackend,
        highlight_mode: EditHighlightModeArg,
    ) -> SceneTargetingData {
        let mut hud_player_tags =
            self.remote_player_tags(view_matrix, look_dir, self.focal_length_xy, aspect);
        self.append_multiplayer_stream_tree_diag_hud_tags(
            &mut hud_player_tags,
            view_matrix,
            self.focal_length_xy,
            aspect,
        );
        let targets = if !self.menu_open
            && self.mouse_grabbed
            && (highlight_mode.uses_faces()
                || highlight_mode.uses_edges()
                || self.placement_preview_mode.needs_targets())
        {
            Some(self.scene.block_edit_targets(
                self.camera.position,
                look_dir,
                edit_reach,
                self.selected_block.scale_exp,
            ))
        } else {
            None
        };
        let mut hud_target_hit_voxel = None;
        let mut hud_target_hit_face = None;
        if let Some(targets) = &targets {
            hud_target_hit_voxel = targets.hit.map(|h| h.origin_i32());
            if targets.face_sign != 0 && targets.hit.is_some() {
                let mut face = [0i32; 4];
                face[targets.face_axis as usize] = targets.face_sign as i32;
                hud_target_hit_face = Some(face);
            }
        }
        let sample_ray_node_hits = if !self.menu_open && self.mouse_grabbed {
            self.scene.debug_render_bvh_ray_node_hits(
                self.camera.position,
                look_dir,
                edit_reach,
                self.multiplayer_stream_tree_diag_sample_ray_max_nodes
                    .max(1),
            )
        } else {
            Vec::new()
        };
        let hud_stream_first_node_desc = sample_ray_node_hits
            .first()
            .map(describe_sample_ray_hit_for_hud);
        let hud_stream_final_solid_leaf_desc = sample_ray_node_hits
            .iter()
            .rev()
            .find(|hit| {
                matches!(
                    hit.kind,
                    DebugRayBvhNodeKind::LeafUniform { .. }
                        | DebugRayBvhNodeKind::LeafChunkArray { .. }
                )
            })
            .map(describe_sample_ray_hit_for_hud);

        self.update_waila_target(look_dir, edit_reach);

        let custom_overlay_edge_instances =
            self.build_overlay_edges(targets.as_ref(), highlight_mode, &sample_ray_node_hits);
        let (
            vte_highlight_hit_min,
            vte_highlight_hit_max,
            vte_highlight_face_axis,
            vte_highlight_face_sign,
        ) = Self::compute_vte_highlight(backend, highlight_mode, targets.as_ref());

        SceneTargetingData {
            targets,
            hud_player_tags,
            hud_target_hit_voxel,
            hud_target_hit_face,
            hud_stream_first_node_desc,
            hud_stream_final_solid_leaf_desc,
            custom_overlay_edge_instances,
            vte_highlight_hit_min,
            vte_highlight_hit_max,
            vte_highlight_face_axis,
            vte_highlight_face_sign,
        }
    }

    /// Resolve screenshot request state: decrement countdown, check manual
    /// trigger, determine whether to take a screenshot this frame and whether
    /// it is an auto-screenshot (exit after render).
    ///
    /// Returns `(take_screenshot, auto_screenshot)`.
    fn resolve_screenshot_request(&mut self, command_screenshot_requested: bool) -> (bool, bool) {
        let countdown_triggered = if self.gpu_screenshot_countdown > 1 {
            self.gpu_screenshot_countdown -= 1;
            false
        } else if self.gpu_screenshot_countdown == 1 {
            self.gpu_screenshot_countdown = 0;
            true
        } else {
            false
        };
        let manual_screenshot = self.input.take_screenshot();
        let mut take_screenshot = manual_screenshot || command_screenshot_requested;
        if countdown_triggered
            && self.args.gpu_screenshot_source == GpuScreenshotSourceArg::Framebuffer
        {
            take_screenshot = true;
        }
        let auto_screenshot = countdown_triggered || command_screenshot_requested;
        if take_screenshot {
            if let Some(parent) = self.args.screenshot_output.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            if auto_screenshot {
                self.should_exit_after_render = true;
            }
        }
        (take_screenshot, auto_screenshot)
    }

    /// Format the VTE sweep status string for the HUD.
    fn vte_sweep_status_string(&self) -> String {
        if let Some(state) = self.vte_sweep_state {
            let profiles = self.vte_sweep_profiles();
            let profile = profiles[state.profile_index];
            format!(
                "#{} {}/{}:{} {}f",
                state.run_id,
                state.profile_index + 1,
                profiles.len(),
                profile.label,
                state.frames_remaining
            )
        } else {
            "off".to_string()
        }
    }

    /// Submit the frame to the VTE or tetra rasterisation backend.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_render_frame(
        &mut self,
        frame_params: FrameParams,
        backend: RenderBackend,
        look_dir: [f32; 4],
        preview_time_s: f32,
        preview_instance: Option<common::ModelInstance>,
        holding_block: bool,
        targets: Option<&scene::BlockEditTargets>,
        disable_remote_non_voxel: bool,
        vte_disable_entities: bool,
    ) {
        if backend == RenderBackend::VoxelTraversal {
            let mut vte_non_voxel_instances = if vte_disable_entities || disable_remote_non_voxel {
                Vec::new()
            } else {
                let remote_instances = self.remote_player_instances(preview_time_s);
                let entity_instances = self.remote_entity_instances();
                let mut instances =
                    Vec::with_capacity(remote_instances.len() + entity_instances.len() + 1);
                instances.extend(remote_instances);
                instances.extend(entity_instances);
                instances
            };
            let preview_instance_storage;
            let mut preview_overlay_instances: &[common::ModelInstance] = &[];
            if let Some(pi) = preview_instance {
                preview_instance_storage = pi;
                if self.vte_overlay_raster_enabled {
                    preview_overlay_instances = std::slice::from_ref(&preview_instance_storage);
                } else if !vte_disable_entities {
                    vte_non_voxel_instances.push(preview_instance_storage);
                }
            }
            // Ghost placement preview (only when holding a block)
            if !vte_disable_entities
                && holding_block
                && self.placement_preview_mode == PlacementPreviewMode::Ghost
            {
                if let Some(targets) = targets {
                    if let Some(place) = &targets.place {
                        if !self.should_suppress_placement_preview(place, targets) {
                            vte_non_voxel_instances.push(build_ghost_placement_instance(
                                place,
                                &self.selected_block,
                                &self.material_resolver,
                                self.placement_orientation,
                            ));
                        }
                    }
                }
            }
            let voxel_build_start = Instant::now();
            let voxel_result = self.scene.build_voxel_frame_data(
                self.camera.position,
                look_dir,
                self.vte_max_trace_distance,
                &self.material_resolver,
            );
            let voxel_build_elapsed_ms = voxel_build_start.elapsed().as_secs_f64() * 1000.0;

            // Install pre-built GPU buffers from background rebuild.
            if let Some(gpu_buffers) = voxel_result.new_gpu_buffers {
                let gen = voxel_result
                    .gpu_buffers_generation
                    .unwrap_or(voxel_result.frame_data.metadata_generation);
                self.rcx
                    .as_mut()
                    .unwrap()
                    .install_new_voxel_gpu_buffers(gpu_buffers, gen);
            }

            // World-ready gate
            if !self.world_ready
                && self.app_state == AppState::Playing
                && self.multiplayer.is_none()
            {
                self.world_ready = true;
                eprintln!("World ready: no multiplayer connection");
            }
            if !self.world_ready
                && self.app_state == AppState::Playing
                && self.multiplayer.is_some()
            {
                if let Some(wait_since) = self.multiplayer_initial_world_wait_since {
                    const MULTIPLAYER_WORLD_READY_FALLBACK_SECS: f32 = 3.0;
                    if wait_since.elapsed().as_secs_f32() >= MULTIPLAYER_WORLD_READY_FALLBACK_SECS {
                        self.world_ready = true;
                        eprintln!(
                            "World ready fallback: no subtree patch after {:.1}s (continuing with async world streaming)",
                            MULTIPLAYER_WORLD_READY_FALLBACK_SECS
                        );
                    }
                }
            }
            let render_submit_start = Instant::now();
            self.rcx.as_mut().unwrap().render_voxel_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                voxel_result.frame_data.as_input(),
                &vte_non_voxel_instances,
                preview_overlay_instances,
            );
            self.set_runtime_profile_voxel_build_ms(voxel_build_elapsed_ms);
            self.set_runtime_profile_render_submit_ms(
                render_submit_start.elapsed().as_secs_f64() * 1000.0,
            );
        } else {
            let remote_instances = if vte_disable_entities || disable_remote_non_voxel {
                Vec::new()
            } else {
                self.remote_player_instances(preview_time_s)
            };
            let entity_instances = if vte_disable_entities || disable_remote_non_voxel {
                Vec::new()
            } else {
                self.remote_entity_instances()
            };
            let mut render_instances =
                Vec::with_capacity(remote_instances.len() + entity_instances.len() + 1);
            render_instances.extend(remote_instances);
            render_instances.extend(entity_instances);
            if !vte_disable_entities {
                if let Some(pi) = preview_instance {
                    render_instances.push(pi);
                }
            }
            let render_submit_start = Instant::now();
            self.rcx.as_mut().unwrap().render_tetra_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                TetraFrameInput {
                    model_instances: &render_instances,
                },
            );
            self.set_runtime_profile_render_submit_ms(
                render_submit_start.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    /// Save an auto-screenshot (render-buffer or framebuffer capture) and
    /// write the JSON sidecar metadata.
    fn save_auto_screenshot(&mut self) {
        if let Some(parent) = self.args.screenshot_output.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match self.args.gpu_screenshot_source {
            GpuScreenshotSourceArg::RenderBuffer => {
                self.rcx.as_mut().unwrap().save_rendered_frame_png(
                    self.args
                        .screenshot_output
                        .to_str()
                        .unwrap_or("frames/gpu_render.png"),
                );
            }
            GpuScreenshotSourceArg::Framebuffer => {
                if let Some(src_path) = latest_framebuffer_screenshot_path() {
                    let webp_path = self.args.screenshot_output.with_extension("webp");
                    if let Err(err) = std::fs::copy(&src_path, &webp_path) {
                        eprintln!(
                            "Failed to copy framebuffer screenshot {} -> {}: {}",
                            src_path.display(),
                            webp_path.display(),
                            err
                        );
                    }
                    match image::open(&src_path) {
                        Ok(img) => {
                            if let Err(err) = img.save(&self.args.screenshot_output) {
                                eprintln!(
                                    "Failed to save {}: {err}",
                                    self.args.screenshot_output.display()
                                );
                            } else {
                                println!("Saved PNG to {}", self.args.screenshot_output.display());
                            }
                        }
                        Err(err) => {
                            eprintln!(
                                "Failed to decode framebuffer screenshot {}: {}",
                                src_path.display(),
                                err
                            );
                        }
                    }
                } else {
                    eprintln!("No framebuffer screenshot found under frames/ after auto-capture.");
                }
            }
        }
        // Write JSON sidecar metadata
        let json_path = self.args.screenshot_output.with_extension("json");
        let window_size = self
            .rcx
            .as_ref()
            .and_then(|rcx| rcx.window.as_ref())
            .map(|w| {
                let size = w.inner_size();
                [size.width, size.height]
            })
            .unwrap_or([0, 0]);
        let metadata = serde_json::json!({
            "render_width": self.args.width,
            "render_height": self.args.height,
            "render_layers": self.args.layers,
            "window_width": window_size[0],
            "window_height": window_size[1],
            "camera_position": [
                self.camera.position[0],
                self.camera.position[1],
                self.camera.position[2],
                self.camera.position[3],
            ],
            "camera_angles_rad": [
                self.camera.yaw,
                self.camera.pitch,
                self.camera.xw_angle,
                self.camera.zw_angle,
            ],
            "backend": format!("{:?}", self.args.backend),
            "vte_display_mode": format!("{:?}", self.args.vte_display_mode),
            "gpu_screenshot_source": format!("{:?}", self.args.gpu_screenshot_source),
            "world_file": if self.args.load_world {
                Some(self.args.world_file.to_string_lossy().into_owned())
            } else {
                None
            },
            "scene": format!("{:?}", self.args.scene),
            "no_hud": self.args.no_hud,
        });
        match std::fs::write(&json_path, serde_json::to_string_pretty(&metadata).unwrap()) {
            Ok(()) => println!("Saved metadata to {}", json_path.display()),
            Err(err) => {
                eprintln!("Failed to save metadata {}: {}", json_path.display(), err)
            }
        }
        self.should_exit_after_render = true;
    }

    pub(super) fn update_and_render(&mut self) {
        let frame_start = Instant::now();
        self.begin_runtime_profile_frame();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        if self.app_state == AppState::MainMenu {
            let main_menu_update_start = Instant::now();
            self.update_and_render_main_menu(dt);
            self.set_runtime_profile_update_ms(
                main_menu_update_start.elapsed().as_secs_f64() * 1000.0,
            );
            if self.perf_suite_active() {
                self.advance_perf_suite_after_frame(frame_start);
            } else {
                self.record_runtime_profile_sample(frame_start);
            }
            self.persist_settings_if_needed(false);
            return;
        }

        let gameplay_update_start = Instant::now();
        self.poll_multiplayer_events();
        self.smooth_remote_players(dt, now);
        self.smooth_remote_entities(dt);
        if let Some(scenario_index) = self
            .perf_suite_state
            .as_ref()
            .map(|state| state.scenario_index)
        {
            self.set_perf_suite_camera_pose(scenario_index);
        }

        let command_screenshot_requested = self.process_command_queue();

        self.apply_input_and_camera(dt);

        // Determine active rotation pair
        let pair = self.active_rotation_pair();

        let edit_reach = self
            .args
            .edit_reach
            .clamp(BLOCK_EDIT_REACH_MIN, BLOCK_EDIT_REACH_MAX);

        self.apply_physics_and_editing(dt, edit_reach, now);

        let look_dir = self.current_look_direction();
        self.send_multiplayer_player_update(now, look_dir);
        if self.inventory_dirty {
            self.inventory_dirty = false;
            self.send_inventory_sync();
        }
        self.send_multiplayer_chunk_sample_diag_request();
        let preview_elapsed = now - self.start_time;
        let preview_time_s = preview_elapsed.as_secs_f32();
        let preview_time_ticks_ms = preview_elapsed.as_millis() as u32;
        let aspect = self
            .rcx
            .as_ref()
            .and_then(|rcx| rcx.window.as_ref())
            .map(|window| {
                let size = window.inner_size();
                size.width.max(1) as f32 / size.height.max(1) as f32
            })
            .unwrap_or_else(|| self.args.width.max(1) as f32 / self.args.height.max(1) as f32);
        let held_item = self
            .inventory
            .hotbar_slot(self.hotbar_selected_index)
            .clone();
        let holding_block = held_item.as_ref().and_then(|s| s.to_block_data()).is_some();
        let preview_instance = held_item.as_ref().map(|stack| {
            build_held_item_preview_instance(
                &self.camera,
                stack,
                preview_time_s,
                self.control_scheme,
                aspect,
                &self.content_registry,
                &self.material_resolver,
            )
        });

        let view_matrix = self.current_view_matrix();
        let backend = self.args.backend.to_render_backend();
        let disable_remote_non_voxel = env_flag_enabled("R4D_DISABLE_REMOTE_NON_VOXEL");
        let vte_disable_entities = env_flag_enabled("R4D_VTE_DISABLE_ENTITIES");
        let highlight_mode = self.args.edit_highlight_mode;

        let scene_data = self.prepare_scene_targeting_data(
            look_dir,
            edit_reach,
            &view_matrix,
            aspect,
            backend,
            highlight_mode,
        );

        let (take_screenshot, auto_screenshot) =
            self.resolve_screenshot_request(command_screenshot_requested);
        let vte_sweep_status = self.vte_sweep_status_string();
        let needs_egui_paint = self.app_state == AppState::MainMenu
            || !self.world_ready
            || self.menu_open
            || self.inventory_open
            || self.teleport_dialog_open
            || self.controls_dialog_open
            || self.dev_console_open
            || self.block_gui_session.is_some();
        let egui_paint = if needs_egui_paint {
            self.run_egui_frame()
        } else {
            None
        };
        let aetna_ui = if self.args.no_hud {
            None
        } else {
            self.build_aetna_overlay()
        };
        let mut do_navigation_hud =
            !self.menu_open && !self.dev_console_open && self.info_panel_mode != InfoPanelMode::Off;
        if self.args.no_hud {
            do_navigation_hud = false;
        }
        let hud_readout_mode = if !self.menu_open
            && !self.dev_console_open
            && matches!(
                self.info_panel_mode,
                InfoPanelMode::VectorTable | InfoPanelMode::VectorTable2
            ) {
            HudReadoutMode::CompactVectors
        } else {
            HudReadoutMode::Full
        };
        let hud_rotation_label = self.current_info_hud_text(
            pair,
            look_dir,
            edit_reach,
            highlight_mode,
            &vte_sweep_status,
            scene_data.hud_target_hit_voxel,
            scene_data.hud_target_hit_face,
            scene_data.hud_stream_first_node_desc.as_deref(),
            scene_data.hud_stream_final_solid_leaf_desc.as_deref(),
        );
        let hud_rotation_label = if self.args.no_hud {
            None
        } else {
            hud_rotation_label
        };

        let render_options = RenderOptions {
            do_raster: true,
            render_backend: backend,
            vte_max_trace_steps: self.vte_max_trace_steps,
            vte_max_trace_distance: self.vte_max_trace_distance,
            vte_display_mode: self.args.vte_display_mode.to_render_mode(),
            vte_slice_layer: self.args.vte_slice_layer,
            vte_thick_half_width: self.args.vte_thick_half_width,
            vte_reference_compare: self.vte_reference_compare_enabled,
            vte_reference_mismatch_only: self.vte_reference_mismatch_only_enabled,
            vte_compare_slice_only: self.vte_compare_slice_only_enabled,
            vte_integral_sky_emissive_tweak: self.vte_integral_sky_emissive_enabled,
            vte_integral_sky_scale: self.vte_integral_sky_scale,
            vte_integral_hit_emissive_boost: self.vte_integral_hit_emissive_boost,
            vte_integral_log_merge_tweak: self.vte_integral_log_merge_enabled,
            vte_integral_log_merge_k: self.vte_integral_log_merge_k,
            zw_angle_color_shift_enabled: self.zw_angle_color_shift_enabled,
            zw_angle_color_shift_strength: self.zw_angle_color_shift_strength,
            vte_highlight_hit_min: if self.args.no_hud {
                None
            } else {
                scene_data.vte_highlight_hit_min
            },
            vte_highlight_hit_max: if self.args.no_hud {
                [0.0; 4]
            } else {
                scene_data.vte_highlight_hit_max
            },
            vte_highlight_face_axis: if self.args.no_hud {
                0
            } else {
                scene_data.vte_highlight_face_axis
            },
            vte_highlight_face_sign: if self.args.no_hud {
                0
            } else {
                scene_data.vte_highlight_face_sign
            },
            do_navigation_hud,
            custom_overlay_lines: Vec::new(),
            custom_overlay_edge_instances: scene_data.custom_overlay_edge_instances,
            take_framebuffer_screenshot: take_screenshot,
            prepare_render_screenshot: auto_screenshot,
            hud_readout_mode,
            hud_rotation_label,
            hud_target_hit_voxel: scene_data.hud_target_hit_voxel,
            hud_target_hit_face: scene_data.hud_target_hit_face,
            hud_player_tags: if self.args.no_hud {
                Vec::new()
            } else {
                scene_data.hud_player_tags
            },
            egui_paint,
            aetna_ui,
            ..Default::default()
        };

        let frame_params = FrameParams {
            view_matrix,
            time_ticks_ms: preview_time_ticks_ms,
            focal_length_xy: self.focal_length_xy,
            focal_length_zw: self.focal_length_zw,
            render_options,
        };
        self.set_runtime_profile_update_ms(gameplay_update_start.elapsed().as_secs_f64() * 1000.0);

        self.dispatch_render_frame(
            frame_params,
            backend,
            look_dir,
            preview_time_s,
            preview_instance,
            holding_block,
            scene_data.targets.as_ref(),
            disable_remote_non_voxel,
            vte_disable_entities,
        );

        let post_render_start = Instant::now();
        if auto_screenshot {
            self.save_auto_screenshot();
        }

        self.advance_vte_runtime_sweep_after_frame();
        self.set_runtime_profile_post_render_ms(post_render_start.elapsed().as_secs_f64() * 1000.0);
        if self.perf_suite_active() {
            self.advance_perf_suite_after_frame(frame_start);
        } else {
            self.record_runtime_profile_sample(frame_start);
        }
        self.persist_settings_if_needed(false);
    }
}
