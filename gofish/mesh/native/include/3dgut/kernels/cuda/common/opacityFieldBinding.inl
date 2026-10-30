// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

// SplatRaster members for the matched camera-min opacity field used by mesh
// extraction. Included by splatRaster.cpp.
//
// Two entry points share one context:
//  - the opacity field: min over training cameras of the per-camera opacity
//    at each query point (begin / integrate camera / finish / release);
//  - the bisection table: the same per-camera evaluation staged once into a
//    persistent camera table and evaluated as a sign test (O >= 1/2) for each
//    of the fixed bisection rounds (begin / stage camera / seal / round /
//    reset / release).
// Cameras must be integrated in training order. All work is issued on the
// current CUDA stream of the query device.

#include <3dgut/kernels/cuda/common/bisectionTable.h>

#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace persistent_table = threedgut::fast_sof_persistent_camera_table;

struct SplatRaster::OpacityFieldContext {
    int64_t token            = 0;
    int device               = -1;
    cudaStream_t stream      = nullptr;
    bool streamSynchronized  = true;
    bool finished            = false;
    uint32_t nextCamera      = 0;

    torch::Tensor particleDensity;
    torch::Tensor particleSigmaMax;
    torch::Tensor queryPoints;
    torch::Tensor minimumOpacity;
    torch::Tensor observed;
    torch::Tensor fallbackPathCount;

    torch::Tensor currentRayOrigin;
    torch::Tensor currentRayDirection;
    torch::Tensor currentPhysicalMask;

    // Bisection table.
    persistent_table::Table* table = nullptr;
    bool tableSealed               = false;
    uint32_t round                 = 0;
    std::vector<torch::Tensor> tableRayOrigins;
    std::vector<torch::Tensor> tablePhysicalMasks;
    torch::Tensor roundActive;
    torch::Tensor roundObserved;
    torch::Tensor roundEqual;
    torch::Tensor roundReport;
    torch::Tensor packedRoundDevice;
    torch::Tensor packedRoundHost;
    size_t reportOffset     = 0;
    size_t packedRoundBytes = 0;

    ~OpacityFieldContext() noexcept {
        if (table != nullptr) {
            const auto destroy = persistent_table::destroyTable(table);
            if (destroy.freed) table = nullptr;
            if (!destroy.ok()) std::terminate();
            streamSynchronized = true;
        }
        if (streamSynchronized || device < 0) return;
        int priorDevice      = -1;
        const bool havePrior = cudaGetDevice(&priorDevice) == cudaSuccess;
        if (cudaSetDevice(device) == cudaSuccess) (void)cudaStreamSynchronize(stream);
        if (havePrior && priorDevice != device) (void)cudaSetDevice(priorDevice);
    }
};

namespace {

void opacityFieldRequireTensor(const torch::Tensor& tensor, const char* name,
                               int device, torch::ScalarType type) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.is_contiguous(),
                name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(tensor.get_device() == device, name, " is on the wrong CUDA device");
    TORCH_CHECK(tensor.scalar_type() == type, name, " has the wrong dtype");
}

void opacityFieldRequireCurrentDevice(int expectedDevice, const char* operation) {
    int currentDevice         = -1;
    const cudaError_t status  = cudaGetDevice(&currentDevice);
    TORCH_CHECK(status == cudaSuccess, operation, ": cudaGetDevice failed: ",
                cudaGetErrorString(status));
    TORCH_CHECK(currentDevice == expectedDevice, operation,
                ": expected cuda:", expectedDevice, ", current cuda:", currentDevice);
}

template <typename T>
T& opacityFieldRequireContext(std::unique_ptr<T>& context, int64_t token) {
    TORCH_CHECK(context && context->token == token,
                "missing or stale opacity-field context");
    return *context;
}

template <typename Context>
void opacityFieldRequireStream(const Context& context, const char* operation) {
    opacityFieldRequireCurrentDevice(context.device, operation);
    TORCH_CHECK(at::cuda::getCurrentCUDAStream(context.device) == context.stream, operation,
                ": the opacity field must run on the stream it was created on");
}

template <typename Context>
void opacityFieldSynchronize(Context& context, const char* operation) {
    const cudaError_t status = cudaStreamSynchronize(context.stream);
    TORCH_CHECK(status == cudaSuccess, operation, " synchronization failed: ",
                cudaGetErrorString(status));
    context.streamSynchronized = true;
}

torch::Tensor opacityFieldPoseSnapshot(const torch::Tensor& tensor, const char* name,
                                       int device) {
    TORCH_CHECK(tensor.defined() && tensor.is_contiguous() &&
                    tensor.scalar_type() == torch::kFloat32 && tensor.numel() == 7,
                name, " must be contiguous float32 [7]");
    if (tensor.is_cuda()) {
        TORCH_CHECK(tensor.get_device() == device, name, " is on the wrong CUDA device");
        return tensor.to(torch::kCPU).contiguous();
    }
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be CPU or CUDA");
    return tensor.clone().contiguous();
}

} // namespace

// Prepare one camera: validate and own its rays/mask and build its render
// parameters. Returns whether a physical-domain mask is attached.
bool SplatRaster::prepareOpacityFieldCamera(
    OpacityFieldContext& context,
    uint32_t frameNumber,
    torch::Tensor rayOrigin,
    torch::Tensor rayDirection,
    torch::Tensor physicalMask,
    threedgut::TSensorModel sensorModel,
    TTimestamp startTimestamp,
    TTimestamp endTimestamp,
    torch::Tensor sensorsStartPose,
    torch::Tensor sensorsEndPose,
    RenderParameters& params) {
    opacityFieldRequireTensor(rayOrigin, "ray origin", context.device, torch::kFloat32);
    opacityFieldRequireTensor(rayDirection, "ray direction", context.device, torch::kFloat32);
    TORCH_CHECK(rayOrigin.sizes() == rayDirection.sizes() &&
                    rayOrigin.dim() == 4 && rayOrigin.size(0) == 1 &&
                    rayOrigin.size(3) == 3 && rayOrigin.size(1) > 0 && rayOrigin.size(2) > 0,
                "rays must be nonempty [1,H,W,3]");
    const int64_t height = rayOrigin.size(1), width = rayOrigin.size(2);
    TORCH_CHECK(width <= std::numeric_limits<int>::max() &&
                    height <= std::numeric_limits<int>::max(),
                "camera raster is too large");
    TORCH_CHECK(sensorModel.shutterType == TSensorModel::GlobalShutter,
                "the opacity field supports global-shutter cameras only");

    torch::Tensor ownedRayOrigin    = rayOrigin.clone();
    torch::Tensor ownedRayDirection = rayDirection.clone();
    const torch::Tensor referenceOrigin =
        ownedRayOrigin.select(0, 0).select(0, 0).select(0, 0);
    TORCH_CHECK(torch::equal(
                    ownedRayOrigin,
                    referenceOrigin.view({1, 1, 1, 3}).expand_as(ownedRayOrigin)),
                "the opacity field requires one camera center per view");

    const bool masked = physicalMask.defined() && physicalMask.numel() != 0;
    torch::Tensor ownedPhysicalMask;
    if (masked) {
        opacityFieldRequireTensor(physicalMask, "physical mask", context.device, torch::kUInt8);
        TORCH_CHECK(physicalMask.dim() == 2 && physicalMask.size(0) == height &&
                        physicalMask.size(1) == width,
                    "physical mask must be contiguous uint8 [H,W]");
        ownedPhysicalMask = physicalMask.clone();
    }

    torch::Tensor ownedStartPose = opacityFieldPoseSnapshot(
        sensorsStartPose, "camera start pose", context.device);
    torch::Tensor ownedEndPose = opacityFieldPoseSnapshot(
        sensorsEndPose, "camera end pose", context.device);
    TORCH_CHECK(torch::isfinite(ownedStartPose).all().item<bool>() &&
                    torch::isfinite(ownedEndPose).all().item<bool>() &&
                    torch::equal(ownedStartPose, ownedEndPose),
                "global-shutter camera poses must be finite and identical");

    params.id          = frameNumber;
    params.resolution  = tcnn::ivec2{static_cast<int>(width), static_cast<int>(height)};
    params.sensorModel = sensorModel;
    params.sensorState = toSensorState(
        startTimestamp, ownedStartPose, endTimestamp, ownedEndPose);
    params.objectAABB = BoundingBox{
        tcnn::vec3{-1e06f, -1e06f, -1e06f}, tcnn::vec3{1e06f, 1e06f, 1e06f}};

    context.currentRayOrigin    = std::move(ownedRayOrigin);
    context.currentRayDirection = std::move(ownedRayDirection);
    context.currentPhysicalMask = std::move(ownedPhysicalMask);
    return masked;
}

int64_t SplatRaster::beginOpacityField(
    torch::Tensor particleDensity,
    torch::Tensor particleSigmaMax,
    torch::Tensor queryPoints) {
    TORCH_CHECK(!m_opacityFieldContext,
                "cannot begin a second opacity field while one is active");
    TORCH_CHECK(queryPoints.is_cuda(), "query points must be CUDA");
    const int device = queryPoints.get_device();
    opacityFieldRequireCurrentDevice(device, "begin opacity field");
    const c10::cuda::CUDAGuard deviceGuard(c10::Device(c10::DeviceType::CUDA, device));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
    opacityFieldRequireTensor(queryPoints, "query points", device, torch::kFloat32);
    opacityFieldRequireTensor(particleDensity, "particle density", device, torch::kFloat32);
    opacityFieldRequireTensor(particleSigmaMax, "particle sigma_max", device, torch::kFloat32);
    TORCH_CHECK(queryPoints.dim() == 2 && queryPoints.size(1) == 3 &&
                    queryPoints.size(0) > 0 &&
                    queryPoints.size(0) <= std::numeric_limits<int>::max(),
                "query points must be nonempty float32 [Q,3]");
    TORCH_CHECK(particleDensity.dim() == 2 && particleDensity.size(1) == 32 &&
                    particleDensity.size(0) > 0 &&
                    particleDensity.size(0) <= std::numeric_limits<uint32_t>::max(),
                "particle density must be float32 [N,32]");
    const uint32_t particleCount = static_cast<uint32_t>(particleDensity.size(0));
    TORCH_CHECK((particleSigmaMax.dim() == 1 ||
                 (particleSigmaMax.dim() == 2 && particleSigmaMax.size(1) == 1)) &&
                    particleSigmaMax.numel() == particleCount,
                "particle sigma_max must be float32 [N] or [N,1]");

    auto context    = std::make_unique<OpacityFieldContext>();
    context->device = device;
    context->stream = stream;
    // The field owns its inputs; kernels consume these allocations only.
    context->particleDensity  = particleDensity.clone();
    context->particleSigmaMax = particleSigmaMax.reshape({-1}).clone();
    context->queryPoints      = queryPoints.clone();

    TORCH_CHECK(m_opacityFieldNextToken <=
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "opacity-field token overflow");
    context->token          = static_cast<int64_t>(m_opacityFieldNextToken++);
    context->minimumOpacity = torch::ones(
        {queryPoints.size(0)},
        torch::TensorOptions().dtype(torch::kFloat32).device(queryPoints.device()));
    context->observed = torch::zeros(
        {queryPoints.size(0)},
        torch::TensorOptions().dtype(torch::kUInt8).device(queryPoints.device()));
    context->fallbackPathCount = torch::zeros(
        {1}, torch::TensorOptions().dtype(torch::kInt32).device(queryPoints.device()));

    m_parameters.values.numParticles      = particleCount;
    m_parameters.values.radianceSphDegree = 0;
    m_parameters.parameters.dptrDensityParameters =
        context->particleDensity.data_ptr<float>();
    m_parameters.parameters.dptrRadianceParameters = nullptr;
    m_parameters.valuesBuffer.setFromHost(
        &m_parameters.values, sizeof(m_parameters.values),
        reinterpret_cast<uint64_t>(stream), m_logger);
    m_parameters.parametersBuffer.setFromHost(
        &m_parameters.parameters, sizeof(m_parameters.parameters),
        reinterpret_cast<uint64_t>(stream), m_logger);
    // Parameter uploads and all later uses are ordered on this stream.
    context->streamSynchronized = false;
    const int64_t token         = context->token;
    m_opacityFieldContext       = std::move(context);
    return token;
}

void SplatRaster::integrateOpacityFieldCamera(
    int64_t contextToken,
    uint32_t frameNumber,
    torch::Tensor rayOrigin,
    torch::Tensor rayDirection,
    torch::Tensor physicalMask,
    threedgut::TSensorModel sensorModel,
    TTimestamp startTimestamp,
    TTimestamp endTimestamp,
    torch::Tensor sensorsStartPose,
    torch::Tensor sensorsEndPose) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    TORCH_CHECK(context.table == nullptr, "camera integration rejected a bisection table");
    TORCH_CHECK(!context.finished, "cannot integrate a camera after finish");
    opacityFieldRequireStream(context, "integrate opacity-field camera");
    const c10::cuda::CUDAGuard deviceGuard(c10::Device(c10::DeviceType::CUDA, context.device));

    RenderParameters params;
    const bool masked = prepareOpacityFieldCamera(
        context, frameNumber, std::move(rayOrigin), std::move(rayDirection),
        std::move(physicalMask), sensorModel, startTimestamp, endTimestamp,
        sensorsStartPose, sensorsEndPose, params);

    uint32_t rejectedQueries    = 0;
    uint32_t firstRejectedQuery = UINT32_MAX;
    uint32_t failureTelemetry[4] = {0u, 0u, UINT32_MAX, 0u};
    const Status status = m_renderer->evaluateFastSofTntFullShapeCamera(
        params,
        reinterpret_cast<const tcnn::vec3*>(context.currentRayOrigin.data_ptr<float>()),
        reinterpret_cast<const tcnn::vec3*>(context.currentRayDirection.data_ptr<float>()),
        masked ? context.currentPhysicalMask.data_ptr<uint8_t>() : nullptr,
        reinterpret_cast<const tcnn::vec3*>(context.queryPoints.data_ptr<float>()),
        static_cast<uint32_t>(context.queryPoints.size(0)),
        context.particleSigmaMax.data_ptr<float>(),
        context.minimumOpacity.data_ptr<float>(), context.observed.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(context.fallbackPathCount.data_ptr<int32_t>()),
        &rejectedQueries, &firstRejectedQuery,
        failureTelemetry, false, m_parameters, context.device, context.stream);
    TORCH_CHECK(status && failureTelemetry[0] == 0u,
                "opacity-field camera evaluation failed: error=",
                failureTelemetry[0], " index=", failureTelemetry[2]);
    TORCH_CHECK(rejectedQueries == 0 && firstRejectedQuery == UINT32_MAX,
                "opacity-field query admission rejected a query");
    context.streamSynchronized = false;
    ++context.nextCamera;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SplatRaster::finishOpacityField(int64_t contextToken) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    TORCH_CHECK(context.table == nullptr, "finish rejected a bisection table");
    TORCH_CHECK(!context.finished, "the opacity field was already finished");
    TORCH_CHECK(context.nextCamera > 0, "the opacity field has no camera");
    opacityFieldRequireStream(context, "finish opacity field");
    const c10::cuda::CUDAGuard deviceGuard(c10::Device(c10::DeviceType::CUDA, context.device));
    opacityFieldSynchronize(context, "finish opacity-field cameras");
    // Unobserved points report zero opacity here; callers treat them as opaque.
    const torch::Tensor observedBool = context.observed.to(torch::kBool);
    const torch::Tensor opacity      = torch::where(
        observedBool, context.minimumOpacity, torch::zeros_like(context.minimumOpacity));
    const torch::Tensor above = torch::logical_and(
        observedBool, opacity.ge(0.5)).to(torch::kUInt8);
    context.streamSynchronized = false;
    opacityFieldSynchronize(context, "finish opacity-field outputs");
    TORCH_CHECK(context.fallbackPathCount.item<int32_t>() == 0,
                "opacity-field evaluation reported a fallback path");
    context.finished = true;
    return {opacity, context.observed, above};
}

void SplatRaster::releaseOpacityField(int64_t contextToken) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    opacityFieldRequireStream(context, "release opacity field");
    const c10::cuda::CUDAGuard deviceGuard(c10::Device(c10::DeviceType::CUDA, context.device));
    TORCH_CHECK(context.finished, "cannot release an opacity field before finish");
    opacityFieldSynchronize(context, "release opacity field");
    m_opacityFieldContext.reset();
}

int64_t SplatRaster::beginBisectionTable(
    torch::Tensor particleDensity,
    torch::Tensor particleSigmaMax,
    torch::Tensor probeQueryPoints) {
    const int64_t token = beginOpacityField(
        std::move(particleDensity), std::move(particleSigmaMax), std::move(probeQueryPoints));
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, token);
    // The table keeps per-query state on device; the cross-view minimum is unused.
    context.minimumOpacity = torch::Tensor();
    context.table          = persistent_table::createTable(
        static_cast<uint32_t>(context.particleDensity.size(0)),
        static_cast<uint32_t>(context.queryPoints.size(0)),
        context.particleSigmaMax.data_ptr<float>(),
        m_parameters.m_dptrParametersBuffer,
        context.device, context.stream, m_logger);
    TORCH_CHECK(context.table != nullptr, "bisection table allocation failed");
    context.tableSealed = false;
    context.round       = 0u;
    return token;
}

void SplatRaster::stageBisectionCamera(
    int64_t contextToken,
    uint32_t frameNumber,
    torch::Tensor rayOrigin,
    torch::Tensor rayDirection,
    torch::Tensor physicalMask,
    threedgut::TSensorModel sensorModel,
    TTimestamp startTimestamp,
    TTimestamp endTimestamp,
    torch::Tensor sensorsStartPose,
    torch::Tensor sensorsEndPose) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    TORCH_CHECK(context.table != nullptr, "stage requires a bisection table");
    TORCH_CHECK(!context.tableSealed, "cannot append to a sealed bisection table");
    opacityFieldRequireStream(context, "stage bisection camera");
    const c10::cuda::CUDAGuard deviceGuard(c10::Device(c10::DeviceType::CUDA, context.device));

    RenderParameters prepared;
    (void)prepareOpacityFieldCamera(
        context, frameNumber, std::move(rayOrigin), std::move(rayDirection),
        std::move(physicalMask), sensorModel, startTimestamp, endTimestamp,
        sensorsStartPose, sensorsEndPose, prepared);

    torch::Tensor ownedStartPose = opacityFieldPoseSnapshot(
        sensorsStartPose, "camera start pose", context.device);
    torch::Tensor ownedEndPose = opacityFieldPoseSnapshot(
        sensorsEndPose, "camera end pose", context.device);
    RenderParameters params;
    params.id         = frameNumber;
    params.resolution = tcnn::ivec2{
        static_cast<int>(context.currentRayDirection.size(2)),
        static_cast<int>(context.currentRayDirection.size(1))};
    params.sensorModel = sensorModel;
    params.sensorState = toSensorState(
        startTimestamp, ownedStartPose, endTimestamp, ownedEndPose);
    params.objectAABB = BoundingBox{
        tcnn::vec3{-1e06f, -1e06f, -1e06f},
        tcnn::vec3{1e06f, 1e06f, 1e06f}};
    const TSensorPose worldToSensorPose = interpolatedSensorPose(
        params.sensorState.startPose, params.sensorState.endPose, 0.5f);
    const TSensorPose sensorToWorldPose = sensorPoseInverse(worldToSensorPose);

    // The table keeps raw pointers to the origin and mask; retain them until release.
    context.tableRayOrigins.push_back(context.currentRayOrigin);
    context.tablePhysicalMasks.push_back(context.currentPhysicalMask);
    persistent_table::CameraStageTelemetry telemetry;
    context.streamSynchronized = false;
    const Status status = persistent_table::appendCamera(
        *context.table, params,
        reinterpret_cast<const tcnn::vec3*>(context.tableRayOrigins.back().data_ptr<float>()),
        reinterpret_cast<const tcnn::vec3*>(context.currentRayDirection.data_ptr<float>()),
        context.tablePhysicalMasks.back().defined() &&
                context.tablePhysicalMasks.back().numel() != 0
            ? context.tablePhysicalMasks.back().data_ptr<uint8_t>()
            : nullptr,
        sensorPoseToMat(worldToSensorPose),
        sensorPoseToMat(sensorToWorldPose), telemetry, m_logger);
    TORCH_CHECK(status, "bisection camera staging failed");
    context.streamSynchronized  = true;
    context.currentRayDirection = torch::Tensor();
    ++context.nextCamera;
}

void SplatRaster::sealBisectionTable(int64_t contextToken) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    TORCH_CHECK(context.table != nullptr, "seal requires a bisection table");
    opacityFieldRequireStream(context, "seal bisection table");
    const c10::cuda::CUDAGuard deviceGuard(c10::Device(c10::DeviceType::CUDA, context.device));
    const uint32_t cameraCount = persistent_table::cameraCount(*context.table);
    TORCH_CHECK(!context.tableSealed && cameraCount > 0 &&
                    context.nextCamera == cameraCount &&
                    context.tableRayOrigins.size() == cameraCount,
                "bisection table camera population differs");
    context.streamSynchronized = false;
    const Status status        = persistent_table::sealTable(*context.table, m_logger);
    TORCH_CHECK(status, "bisection table seal failed");
    context.streamSynchronized = true;

    const auto byteOptions = torch::TensorOptions().dtype(torch::kUInt8)
                                 .device(context.queryPoints.device());
    context.roundActive   = torch::empty({context.queryPoints.size(0)}, byteOptions);
    context.roundObserved = torch::empty({context.queryPoints.size(0)}, byteOptions);
    context.roundEqual    = torch::empty({context.queryPoints.size(0)}, byteOptions);
    context.roundReport   = torch::empty(
        {3 + static_cast<int64_t>(cameraCount) * 3},
        torch::TensorOptions().dtype(torch::kInt32).device(context.queryPoints.device()));
    // One packed device buffer per round: above | observed | equal | report.
    const size_t vectorBytes = static_cast<size_t>(context.queryPoints.size(0));
    TORCH_CHECK(vectorBytes <= (std::numeric_limits<size_t>::max() - 3u) / 3u,
                "bisection round buffer size overflow");
    context.reportOffset     = (3u * vectorBytes + 3u) & ~size_t{3u};
    const size_t reportWords = 3u + static_cast<size_t>(cameraCount) * 3u;
    context.packedRoundBytes = context.reportOffset + reportWords * sizeof(uint32_t);
    TORCH_CHECK(context.packedRoundBytes <=
                    static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                "bisection round buffer size overflow");
    context.packedRoundDevice = torch::empty(
        {static_cast<int64_t>(context.packedRoundBytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(context.queryPoints.device()));
    context.packedRoundHost = torch::empty(
        {static_cast<int64_t>(context.packedRoundBytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    context.streamSynchronized = false;
    opacityFieldSynchronize(context, "bisection table seal");
    context.observed    = torch::Tensor();
    context.tableSealed = true;
    context.finished    = false;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SplatRaster::evaluateBisectionRound(
    int64_t contextToken, torch::Tensor queryPoints, uint32_t roundOrdinal) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    TORCH_CHECK(context.table != nullptr, "round requires a bisection table");
    opacityFieldRequireStream(context, "evaluate bisection round");
    TORCH_CHECK(context.tableSealed && !context.finished &&
                    roundOrdinal == context.round && roundOrdinal < 8u,
                "bisection rounds must run in order [0,8)");
    opacityFieldRequireTensor(queryPoints, "round query points", context.device, torch::kFloat32);
    TORCH_CHECK(queryPoints.dim() == 2 && queryPoints.size(1) == 3 &&
                    queryPoints.size(0) == context.queryPoints.size(0),
                "round query shape differs from the table capacity");

    context.streamSynchronized = false;
    const Status status        = persistent_table::enqueueRound(
        *context.table,
        reinterpret_cast<const tcnn::vec3*>(queryPoints.data_ptr<float>()),
        context.roundActive.data_ptr<uint8_t>(),
        context.roundObserved.data_ptr<uint8_t>(),
        context.roundEqual.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(context.roundReport.data_ptr<int32_t>()) + 3,
        reinterpret_cast<uint32_t*>(context.roundReport.data_ptr<int32_t>()),
        context.packedRoundDevice.data_ptr<uint8_t>(),
        context.packedRoundBytes,
        m_logger);
    TORCH_CHECK(status, "bisection round enqueue failed");

    cudaError_t cudaStatus = cudaMemcpyAsync(
        context.packedRoundHost.data_ptr<uint8_t>(),
        context.packedRoundDevice.data_ptr<uint8_t>(),
        context.packedRoundBytes, cudaMemcpyDeviceToHost, context.stream);
    TORCH_CHECK(cudaStatus == cudaSuccess, "bisection round copy failed: ",
                cudaGetErrorString(cudaStatus));
    opacityFieldSynchronize(context, "bisection round");
    const uint32_t* hostFailure = reinterpret_cast<const uint32_t*>(
        context.packedRoundHost.data_ptr<uint8_t>() + context.reportOffset);
    TORCH_CHECK(hostFailure[0] == 0u, "bisection round failed: code=", hostFailure[0],
                " camera=", hostFailure[1], " index=", hostFailure[2]);
    ++context.round;
    if (context.round == 8u) context.finished = true;
    return {context.roundActive, context.roundObserved, context.roundEqual};
}

void SplatRaster::resetBisectionBatch(int64_t contextToken) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    TORCH_CHECK(context.table != nullptr, "reset requires a bisection table");
    opacityFieldRequireStream(context, "reset bisection batch");
    TORCH_CHECK(context.tableSealed && context.finished && context.round == 8u,
                "a query batch can be reset only after all eight rounds");
    context.round    = 0u;
    context.finished = false;
}

void SplatRaster::releaseBisectionTable(int64_t contextToken) {
    auto& context = opacityFieldRequireContext(m_opacityFieldContext, contextToken);
    TORCH_CHECK(context.table != nullptr, "release requires a bisection table");
    opacityFieldRequireStream(context, "release bisection table");
    const c10::cuda::CUDAGuard deviceGuard(c10::Device(c10::DeviceType::CUDA, context.device));
    TORCH_CHECK(context.tableSealed && context.finished && context.round == 8u,
                "the bisection table can be released only after all eight rounds");
    const auto destroy = persistent_table::destroyTable(context.table);
    if (destroy.freed) context.table = nullptr;
    TORCH_CHECK(destroy.ok(), "bisection table destroy failed");
    context.streamSynchronized = true;
    m_opacityFieldContext.reset();
}
