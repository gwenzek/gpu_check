const std = @import("std");
const log = std.log;

// TODO: we only use a small part of Cuda API, inline the relevant code in a local file
const cuda = @import("cudaz");
const CUDA_SUCCESS = cuda.cu.CUDA_SUCCESS;

const Nvml = @import("Nvml.zig");

const Result = enum(u8) {
    SUCCESS = 0,
    TRANSIENT = 111,
    RUNNING_PROCESSES = 101,
    MEMORY_USED = 102,
    MEM_TEST = 103,
};

pub fn main() u8 {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = general_purpose_allocator.allocator();
    const args = std.process.argsAlloc(alloc) catch {
        log.err("gpu_check crashed: Failed to read args.", .{});
        return @enumToInt(Result.TRANSIENT);
    };
    defer std.process.argsFree(alloc, args);
    // TODO: parse args and run for each GPU
    return @enumToInt(checkGPU(0));
}

/// Simple GPU checks
/// From: https://github.com/fairinternal/ResearchSuperCluster/blob/main/slurm/scripts/check_gpu.py
pub fn checkGPU(deviceId: u3) Result {
    log.info("GPU {}, running various checks", .{deviceId});
    const nvml = Nvml.init();
    defer nvml.deinit();
    const dev0 = nvml.getDevice(deviceId);
    var retcode: Result = .SUCCESS;

    const num_proc = dev0.getComputeRunningProcesses();
    if (num_proc > 0) {
        log.warn("Found {} processes running on GPU {}.", .{ num_proc, dev0.id });
        retcode = .RUNNING_PROCESSES;
    }

    const mem = dev0.getMemoryInfo();
    log.info("Memory usage: {} used, {} free out of {} total.", .{ mem.used, mem.free, mem.total });
    if (mem.used > 100_000_000) {
        // Note: we can't expect 0 mem used,
        // because Cuda driver always reserve some memory for itself.
        const mem_ratio = @intToFloat(f32, mem.used) / @intToFloat(f32, mem.total) * 100;
        log.warn("Memory usage is pretty high ! {d:.2} %", .{mem_ratio});
        retcode = .MEMORY_USED;
    }

    if (!memtest(deviceId, 0.5)) {
        retcode = .MEM_TEST;
    }
    if (retcode == .SUCCESS) {
        log.info("GPU {} looks healthy !", .{deviceId});
    }
    return retcode;
}

pub fn memtest(device: u3, mem_gb: f32) bool {
    var stream = cuda.Stream.init(device) catch |stream_err| {
        log.warn("Failed to start Cuda stream {}", .{stream_err});
        return false;
    };
    defer stream.deinit();
    var d_A = stream.alloc(u8, @floatToInt(usize, mem_gb * 1e9)) catch {
        log.warn("Failed to allocated {}Gb.", .{mem_gb});
        return false;
    };
    var memset = stream._memset(u8, d_A, 0xf);
    if (memset != CUDA_SUCCESS) {
        log.warn("Failed to memcopy with {s}({})", .{ cuda.errorName(memset), memset });
        return false;
    }

    var free = stream._free(d_A);
    if (free != CUDA_SUCCESS) {
        log.warn("Failed to free vector A {s}({})", .{ cuda.errorName(free), free });
        return false;
    }

    var sync = stream._synchronize();
    if (sync != CUDA_SUCCESS) {
        log.warn("Error in memset {s}({})", .{ cuda.errorName(sync), sync });
    }

    return true;
}
