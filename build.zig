const std = @import("std");
const cuda_sdk = @import("cudaz/sdk.zig");

const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;

const CUDA_PATH = "/usr/local/cuda";
const SDK_ROOT = sdk_root() ++ "/";

pub fn addCudaDeps(
    b: *Builder,
    exe: *LibExeObjStep,
    cuda_dir: []const u8,
) void {
    // Add libc and cuda headers / lib
    exe.linkLibC();
    const cuda_lib64 = std.fs.path.join(b.allocator, &[_][]const u8{ cuda_dir, "lib64" }) catch unreachable;
    defer b.allocator.free(cuda_lib64);
    exe.addLibPath(cuda_lib64);
    exe.linkSystemLibraryNeeded("nvidia-ml");
    exe.linkSystemLibraryNeeded("cuda");
    // exe.linkSystemLibraryNeeded("nvidia-ptxjitcompiler");
    exe.addIncludeDir(SDK_ROOT ++ "src");

    const cuda_include = std.fs.path.join(b.allocator, &[_][]const u8{ cuda_dir, "include" }) catch unreachable;
    defer b.allocator.free(cuda_include);
    exe.addIncludeDir(cuda_include);
}

fn sdk_root() []const u8 {
    return std.fs.path.dirname(@src().file).?;
}

pub fn build(b: *Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();
    var gpu_check = b.addExecutable("gpu_check", "src/main.zig");
    gpu_check.setTarget(target);
    gpu_check.setBuildMode(mode);

    var cuda_path = std.os.getenv("CUDA_HOME");
    if (cuda_path == null) cuda_path = CUDA_PATH;
    addCudaDeps(b, gpu_check, cuda_path.?);
    cuda_sdk.addCudazDeps(b, gpu_check, "/cuda_dir/", "./nope.zig", "./nope.ptx");
    gpu_check.install();

    var run = b.step("run", "Run gpu_check");
    const gpu_check_run = gpu_check.run();
    run.dependOn(&gpu_check_run.step);
    run.dependOn(b.getInstallStep());
    // var tests = b.step("test", "Tests");
}
