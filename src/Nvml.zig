const Self = @This();
const nv = @import("cimports.zig").c;
const std = @import("std");

pub fn check(errcode: nv.nvmlReturn_t) void {
    // TODO be smarter about what error we handle.
    if (errcode == 0) return;
    const msg = switch (errcode) {
        1 => "NVML_ERROR_UNINITIALIZED: NVML was not first initialized with nvmlInit()",
        2 => "NVML_ERROR_INVALID_ARGUMENT: A supplied argument is invalid",
        3 => "NVML_ERROR_NOT_SUPPORTED: The requested operation is not available on target device",
        4 => "NVML_ERROR_NO_PERMISSION: The current user does not have permission for operation",
        5 => "NVML_ERROR_ALREADY_INITIALIZED: Deprecated: Multiple initializations are now allowed through ref counting",
        6 => "NVML_ERROR_NOT_FOUND: A query to find an object was unsuccessful",
        7 => "NVML_ERROR_INSUFFICIENT_SIZE: An input argument is not large enough",
        8 => "NVML_ERROR_INSUFFICIENT_POWER: A device's external power cables are not properly attached",
        9 => "NVML_ERROR_DRIVER_NOT_LOADED: NVIDIA driver is not loaded",
        10 => "NVML_ERROR_TIMEOUT: User provided timeout passed",
        11 => "NVML_ERROR_IRQ_ISSUE: NVIDIA Kernel detected an interrupt issue with a GPU",
        12 => "NVML_ERROR_LIBRARY_NOT_FOUND: NVML Shared Library couldn't be found or loaded",
        13 => "NVML_ERROR_FUNCTION_NOT_FOUND: Local version of NVML doesn't implement this function",
        14 => "NVML_ERROR_CORRUPTED_INFOROM: infoROM is corrupted",
        15 => "NVML_ERROR_GPU_IS_LOST: The GPU has fallen off the bus or has otherwise become inaccessible",
        16 => "NVML_ERROR_RESET_REQUIRED: The GPU requires a reset before it can be used again",
        17 => "NVML_ERROR_OPERATING_SYSTEM: The GPU control device has been blocked by the operating system/cgroups",
        18 => "NVML_ERROR_LIB_RM_VERSION_MISMATCH: RM detects a driver/library version mismatch",
        19 => "NVML_ERROR_IN_USE: An operation cannot be performed because the GPU is currently in use",
        20 => "NVML_ERROR_MEMORY: Insufficient memory",
        21 => "NVML_ERROR_NO_DATA: No data",
        22 => "NVML_ERROR_VGPU_ECC_NOT_SUPPORTED: The requested vgpu operation is not available on target device, becasue ECC is enabled",
        23 => "NVML_ERROR_INSUFFICIENT_RESOURCES: Ran out of critical resources, other than memory",
        24 => "NVML_ERROR_FREQ_NOT_SUPPORTED: Ran out of critical resources, other than memory",
        else => "NVML_ERROR_UNKNOWN: An internal driver error occurred",
    };
    std.debug.panic("NVML error {}: {s}", .{ errcode, msg });
}

pub fn init() Self {
    check(nv.nvmlInit());
    return .{};
}

pub fn deinit(self: Self) void {
    _ = self;
    _ = nv.nvmlShutdown();
}

pub fn getDeviceCount(self: Self) u8 {
    _ = self;
    var res: c_uint = undefined;
    check(nv.nvmlDeviceGetCount(&res));
    return @intCast(u8, res);
}

const Memory = nv.nvmlMemory_t;

const Device = struct {
    id: u8,
    handle: *nv.nvmlDevice_st,

    /// Returns the number of "compute" processes using the GPU.
    pub fn getComputeRunningProcesses(self: Device) u32 {
        // TODO: also look at GraphicsRunningProcesses
        // Passing count=0 indicates we want to get the actual count
        var count: c_uint = 0;
        const errcode = nv.nvmlDeviceGetComputeRunningProcesses(self.handle, &count, null);
        switch (errcode) {
            // NVML_ERROR_INSUFFICIENT_SIZE indicates than more than "count" process were found
            // Since we aren't reading the infos, this is not an error for us.
            0, 7 => return @intCast(u32, count),
            else => {
                // Check will panic.
                check(errcode);
                unreachable;
            },
        }
    }

    pub fn getMemoryInfo(self: Device) Memory {
        var memory: Memory = undefined;
        // TODO: use nvmlDeviceGetMemoryInfo_v2 if needed
        check(nv.nvmlDeviceGetMemoryInfo(self.handle, &memory));
        return memory;
    }
};

pub fn getDevice(self: Self, id: u8) Device {
    _ = self;
    var dev: Device = .{ .id = id, .handle = undefined };
    check(nv.nvmlDeviceGetHandleByIndex(@as(c_uint, id), @ptrCast([*c]?*nv.nvmlDevice_st, &dev.handle)));
    return dev;
}
