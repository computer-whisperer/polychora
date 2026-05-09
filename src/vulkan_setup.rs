use std::sync::Arc;
use vulkano::device::{
    physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
    Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::event_loop::EventLoop;

pub fn vulkan_setup(
    event_loop: Option<&EventLoop<()>>,
) -> (Arc<Instance>, Arc<Device>, Arc<Queue>) {
    let library = VulkanLibrary::new().unwrap();

    let required_extensions = match event_loop {
        Some(event_loop) => Surface::required_extensions(event_loop).unwrap(),
        None => Default::default(),
    };

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: event_loop.is_some(),
        ..DeviceExtensions::empty()
    };

    let mut device_features = DeviceFeatures {
        fill_mode_non_solid: true,
        vulkan_memory_model: true,
        vulkan_memory_model_device_scope: true,
        variable_pointers: true,
        variable_pointers_storage_buffer: true,
        // Required by our Slang compile flags (`-fvk-use-scalar-layout`) and
        // validated by `spirv-val --scalar-block-layout`.
        scalar_block_layout: true,
        shader_int64: true,
        shader_int8: true,
        shader_draw_parameters: true,
        // Required for texture pool: partially-bound descriptor arrays of sampled 3D textures.
        descriptor_binding_partially_bound: true,
        runtime_descriptor_array: true,
        shader_sampled_image_array_non_uniform_indexing: true,
        ..Default::default()
    };
    device_features |= aetna_vulkano::required_device_features();

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
                && p.supported_features().contains(&device_features)
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::COMPUTE)
                        && match event_loop {
                            Some(event_loop) => {
                                q.queue_flags.intersects(QueueFlags::GRAPHICS)
                                    && p.presentation_support(i as u32, event_loop).unwrap()
                            }
                            None => true,
                        }
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: device_features,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    (instance, device, queue)
}
