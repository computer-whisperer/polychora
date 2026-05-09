use aetna_core::prelude::{
    badge, card, card_content, card_header, card_title, column, mono, row, spacer, stack, text,
    tokens, Align, Axis, Color, Cursor, El, Justify, Rect, Size,
};

use super::{block_data_from_slot, format_cbor_for_display, App, WailaTarget};

const HOTBAR_SLOT_KEY_PREFIX: &str = "aetna_hotbar_slot_";

impl App {
    pub(super) fn build_aetna_overlay(&self) -> Option<El> {
        let mut children = vec![self.build_aetna_hotbar()];
        if let Some(waila) = self.build_aetna_waila_panel() {
            children.push(waila);
        }

        Some(stack(children).fill_size().layout(|cx| {
            let mut rects = Vec::with_capacity(cx.children.len());
            for (index, child) in cx.children.iter().enumerate() {
                let (measured_w, measured_h) = (cx.measure)(child);
                let width = measured_w.min((cx.container.w - 24.0).max(260.0));
                let x = cx.container.x + ((cx.container.w - width) * 0.5).max(12.0);
                let y = if index == 0 {
                    cx.container.bottom() - measured_h - 12.0
                } else {
                    cx.container.y + 30.0
                };
                rects.push(Rect::new(x, y, width, measured_h));
            }
            rects
        }))
    }

    fn build_aetna_hotbar(&self) -> El {
        let slots = (0..9).map(|i| self.build_aetna_hotbar_slot(i));
        row(slots)
            .gap(tokens::SPACE_2)
            .align(Align::Center)
            .height(Size::Hug)
            .width(Size::Hug)
    }

    fn build_aetna_hotbar_slot(&self, index: usize) -> El {
        let stack = self.inventory.hotbar_slot(index);
        let (name, count, color) = stack
            .as_ref()
            .and_then(|stack| {
                let block = stack.to_block_data()?;
                let entry = self
                    .content_registry
                    .block_entry(block.namespace, block.block_type);
                let name = entry
                    .map(|entry| entry.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());
                let [r, g, b] = entry.map(|entry| entry.color).unwrap_or([128, 128, 128]);
                Some((name, stack.count, Color::rgb(r, g, b)))
            })
            .unwrap_or_else(|| ("Empty".to_string(), 0, Color::rgb(44, 48, 58)));
        let selected = index == self.hotbar_selected_index;
        let label = short_label(name);

        column([
            row([
                text(format!("{}", index + 1)).caption().muted(),
                spacer(),
                if count > 1 {
                    badge(format!("{}", count)).muted()
                } else {
                    text("")
                },
            ])
            .width(Size::Fill(1.0))
            .align(Align::Center),
            column(std::iter::empty::<El>())
                .width(Size::Fixed(36.0))
                .height(Size::Fixed(28.0))
                .fill(color)
                .radius(4.0),
            text(label)
                .caption()
                .center_text()
                .ellipsis()
                .max_lines(1)
                .width(Size::Fill(1.0)),
        ])
        .key(format!("aetna_hotbar_slot_{index}"))
        .focusable()
        .cursor(Cursor::Pointer)
        .width(Size::Fixed(76.0))
        .height(Size::Fixed(82.0))
        .padding(tokens::SPACE_2)
        .gap(tokens::SPACE_1)
        .align(Align::Center)
        .fill(tokens::CARD.with_alpha(205))
        .stroke(if selected {
            Color::rgb(250, 246, 140)
        } else {
            tokens::BORDER.with_alpha(180)
        })
        .stroke_width(if selected { 2.5 } else { 1.0 })
        .radius(6.0)
        .shadow(if selected { tokens::SHADOW_MD } else { 0.0 })
    }

    fn build_aetna_waila_panel(&self) -> Option<El> {
        let target = self.waila_target.as_ref()?;
        let panel = match target {
            WailaTarget::Block { coords, block } => {
                let entry = self
                    .content_registry
                    .block_entry(block.namespace, block.block_type);
                let name = entry.map(|e| e.name.as_str()).unwrap_or("Unknown");
                let category = entry.map(|e| e.category.label()).unwrap_or("Unknown");
                let ns_label = self.content_registry.namespace_label(block.namespace);
                let scale_label = if block.scale_exp != 0 {
                    format!("  scale: {}", block.scale_exp)
                } else {
                    String::new()
                };

                waila_panel(
                    name,
                    category,
                    [
                        format!(
                            "ns: {:#010x} ({})  type: {:#010x}",
                            block.namespace, ns_label, block.block_type
                        ),
                        format!(
                            "[{}, {}, {}, {}]{}",
                            coords[0], coords[1], coords[2], coords[3], scale_label
                        ),
                    ],
                )
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
                let entry = self
                    .content_registry
                    .entity_lookup(*entity_type_ns, *entity_type);
                let canonical_name = entry
                    .map(|e| e.canonical_name.as_str())
                    .unwrap_or("unknown");
                let category = entry
                    .map(|e| format!("{:?}", e.category))
                    .unwrap_or_else(|| "Unknown".to_string());
                let player_name = self.remote_players.get(entity_id).map(|p| p.name.clone());
                let display_name = if let Some(name) = player_name {
                    format!("{} ({})", canonical_name, name)
                } else {
                    canonical_name.to_string()
                };
                let ns_label = self.content_registry.namespace_label(*entity_type_ns);
                let mut lines = vec![
                    format!(
                        "id: {}  ns: {:#010x} ({})  type: {:#010x}",
                        entity_id, entity_type_ns, ns_label, entity_type
                    ),
                    format!(
                        "pos: [{:.1}, {:.1}, {:.1}, {:.1}]",
                        position[0], position[1], position[2], position[3]
                    ),
                    format!(
                        "ori: [{:.2}, {:.2}, {:.2}, {:.2}]  scale: {:.2}",
                        orientation[0], orientation[1], orientation[2], orientation[3], scale
                    ),
                ];
                if let Some(entry) = entry {
                    if let Some(config) = &entry.sim_config {
                        lines.push(format!(
                            "{:?}: {:?} spd={:.1}",
                            config.mode, config.locomotion, config.move_speed
                        ));
                    }
                }
                if let Some(decoded) = format_cbor_for_display(data) {
                    lines.push(format!("data: {}", decoded));
                }

                waila_panel(
                    display_name,
                    format!("{} {:.1}m", category, distance),
                    lines,
                )
            }
        };

        Some(panel)
    }
}

fn short_label(name: String) -> String {
    if name.chars().count() > 14 {
        let prefix = name.chars().take(11).collect::<String>();
        format!("{prefix}...")
    } else {
        name
    }
}

fn waila_panel(
    title: impl Into<String>,
    badge_label: impl Into<String>,
    detail_lines: impl IntoIterator<Item = String>,
) -> El {
    let details: Vec<El> = detail_lines
        .into_iter()
        .map(|line| mono(line).caption().muted().width(Size::Fill(1.0)))
        .collect();

    card([
        card_header([row([
            card_title(title).line_height(tokens::TEXT_BASE.size),
            spacer(),
            badge(badge_label).info(),
        ])
        .align(Align::Center)
        .gap(tokens::SPACE_3)]),
        card_content([column(details).gap(tokens::SPACE_1).width(Size::Fill(1.0))]).pt(0.0),
    ])
    .width(Size::Fixed(500.0))
    .height(Size::Hug)
    .axis(Axis::Column)
    .justify(Justify::Start)
    .shadow(tokens::SHADOW_LG)
}

impl App {
    pub(super) fn handle_aetna_ui_events(&mut self, events: Vec<aetna_core::UiEvent>) -> bool {
        let mut consumed = false;
        for event in events {
            let Some(route) = event.route() else {
                continue;
            };
            let Some(slot_index) = aetna_hotbar_slot_index(route) else {
                continue;
            };

            consumed = true;
            if event.is_click_or_activate(route) {
                self.hotbar_selected_index = slot_index;
                self.selected_block =
                    block_data_from_slot(self.inventory.hotbar_slot(self.hotbar_selected_index));
            }
        }
        consumed
    }
}

fn aetna_hotbar_slot_index(route: &str) -> Option<usize> {
    let suffix = route.strip_prefix(HOTBAR_SLOT_KEY_PREFIX)?;
    let index = suffix.parse::<usize>().ok()?;
    (index < 9).then_some(index)
}
