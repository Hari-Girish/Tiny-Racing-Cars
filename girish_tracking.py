"""Object tracking and motion analysis for the toy car races (Person 2 / Hari).

The tracker is covariance tracking from Content/CovarianceTracking.pdf: each car
is described by the covariance matrix of the per-pixel feature vector
[x y R G B] over its window, and the car is found in the next frame by scanning
candidate windows and keeping the one whose covariance matrix is closest to the
model under the Riemannian distance

    rho(C1, C2) = sqrt( sum_i ln(lambda_i(C1, C2))^2 )

where lambda_i are the generalized eigenvalues.  This is the same distance
implemented for HW4 Problem 4, generalized here to take both matrices.

Everything that would otherwise be a per-sequence tuned constant is derived from
the sequence itself:

    pyramid level      round(log2(car height / TARGET_CAR_PIXELS))
    search radius      2x the largest detected displacement in the first frames
    window size        the frame-1 detection bounding box, plus a scale search
    backsub threshold  swept until exactly two car-sized moving regions survive

Supporting techniques are all from the course content: Gaussian pyramids and
coarse-to-fine search (ImagePyramids.pdf), image differencing (Motion.pdf), and
background subtraction / morphology / connected components (RegionExtraction.pdf).
"""

import numpy as np
from scipy import ndimage
from scipy.linalg import eigh
from skimage.measure import label, regionprops

# feature vectors use colours on a 0-255 scale, matching the HW4 convention.
# the covariance distance is invariant to a per-channel rescaling, so this is a
# conditioning choice rather than a meaningful one
COLOR_SCALE = 255.0

# a car is resized to roughly this many pixels tall before searching, which is
# what makes one pyramid rule work for 600 px cars and 200 px cars alike
TARGET_CAR_PIXELS = 90.0

# initialization runs on quarter-size frames; the full 12 MP frames are only
# needed for the final overlays
INIT_SCALE = 0.25

# thresholds tried when looking for the two cars in the first frame
THRESHOLD_SWEEP = np.arange(0.15, 0.80, 0.05)

# a candidate region must have at least this fraction of its pixels moving
# between the first two frames to count as a car rather than a static artifact
MOTION_OVERLAP_MIN = 0.05


# ---------------------------------------------------------------------------
# Gaussian pyramid  (Content/ImagePyramids.pdf)
# ---------------------------------------------------------------------------

# separable 5-tap generating kernel, the standard choice for pyramid blurring
_PYRAMID_KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _blurDecimate(image, axis):
    """Blur with the 5-tap kernel and keep every other sample, in one pass.

    Only the samples that survive decimation are ever computed, and each of the
    five kernel taps is gathered straight into a half-size array.  Filtering the
    full frame first and subsampling afterwards allocates several copies of a
    12 MP frame and is what made this the slowest step in the pipeline.
    """
    length = image.shape[axis]
    kept = np.arange(0, length, 2)

    result = None
    for offset, weight in enumerate(_PYRAMID_KERNEL):
        source = np.clip(kept + offset - 2, 0, length - 1)
        contribution = np.take(image, source, axis=axis) * weight
        result = contribution if result is None else result + contribution
    return result


def pyramidDown(image):
    # separable: rows first, then columns on the already-halved array
    return _blurDecimate(_blurDecimate(image, 0), 1)


def downsample(image, levels):
    # repeat the blur-and-decimate step to reach the requested pyramid level
    out = image
    for _ in range(levels):
        out = pyramidDown(out)
    return out


def chooseLevel(carHeight, target=TARGET_CAR_PIXELS):
    # pick the level that brings the car closest to the target size; halving the
    # image halves the car, so the level is just a log2 of the size ratio
    if carHeight <= target:
        return 0
    return int(max(0, round(np.log2(carHeight / target))))


# ---------------------------------------------------------------------------
# Covariance descriptor and distance  (Content/CovarianceTracking.pdf)
# ---------------------------------------------------------------------------

def covDescriptor(patch):
    # f_k = [x y R G B] per pixel with x, y local to the patch -> 5x5 covariance.
    # frames are stored as float32 to keep memory down; the covariance itself is
    # accumulated in float64 so the generalized eigenproblem stays well behaved
    rows, cols = patch.shape[:2]
    yy, xx = np.mgrid[0:rows, 0:cols]
    colors = patch.astype(np.float64) * COLOR_SCALE
    features = np.column_stack([
        xx.ravel().astype(np.float64),
        yy.ravel().astype(np.float64),
        colors[:, :, 0].ravel(),
        colors[:, :, 1].ravel(),
        colors[:, :, 2].ravel(),
    ])
    # bias=True gives the 1/MN normalization the slides use
    C = np.cov(features, rowvar=False, bias=True)
    # a window of uniform colour has a singular colour block, which makes the
    # generalized eigenproblem fail, so hold the matrix just off singular
    ridge = 1e-8 * np.trace(C) / C.shape[0]
    return C + ridge * np.eye(C.shape[0])


def covDistance(C1, C2):
    # sqrt(sum(ln(generalized eigenvalue)^2)); infinite if the pair is degenerate
    try:
        lam = eigh(C1, C2, eigvals_only=True)
    except np.linalg.LinAlgError:
        return np.inf
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        return np.inf
    return float(np.sqrt(np.sum(np.log(lam) ** 2)))


# a window may hang off the edge of the frame, but only so far: the cars really
# do drive out of shot at the end of several races, and refusing to describe a
# partly visible car means losing the track exactly when it matters
MIN_WINDOW_COVERAGE = 0.6


def windowDescriptor(image, centerRow, centerCol, height, width,
                     minCoverage=MIN_WINDOW_COVERAGE):
    """Covariance of the window centred on the given point.

    The window is clipped to the frame rather than rejected outright, so a car
    leaving the shot is still described, up to the point where too little of it
    is visible for the description to mean anything.
    """
    r0 = int(round(centerRow - height / 2.0))
    c0 = int(round(centerCol - width / 2.0))
    r1, c1 = r0 + max(1, int(round(height))), c0 + max(1, int(round(width)))

    clippedR0, clippedC0 = max(0, r0), max(0, c0)
    clippedR1, clippedC1 = min(image.shape[0], r1), min(image.shape[1], c1)
    if clippedR1 - clippedR0 < 2 or clippedC1 - clippedC0 < 2:
        return None

    coverage = (((clippedR1 - clippedR0) * (clippedC1 - clippedC0))
                / float((r1 - r0) * (c1 - c0)))
    if coverage < minCoverage:
        return None
    return covDescriptor(image[clippedR0:clippedR1, clippedC0:clippedC1, :])


# ---------------------------------------------------------------------------
# Background subtraction and region extraction, used only to start the tracks
# ---------------------------------------------------------------------------

def _cleanAndLabel(binary):
    # closing followed by a hole fill, then connected components; the same chain
    # the detection half uses
    closed = ndimage.binary_dilation(binary, structure=np.ones((3, 3)))
    closed = ndimage.binary_erosion(closed, structure=np.ones((3, 3)))
    closed = ndimage.binary_fill_holes(closed)
    return label(closed)


def _describeRegions(labels, frame, minArea, moving=None):
    """Region records for every labelled component above the area floor.

    Mean colour is taken with ndimage.mean over the whole label image at once.
    Building a boolean mask per region instead costs one full image pass per
    region, and a low threshold on the carpet can leave fifty of them.
    """
    props = [p for p in regionprops(labels) if p.area >= minArea]
    if not props:
        return []
    index = [p.label for p in props]
    channels = [ndimage.mean(frame[:, :, k], labels=labels, index=index) for k in range(3)]
    overlaps = (ndimage.mean(moving.astype(np.float64), labels=labels, index=index)
                if moving is not None else None)

    regions = []
    for position, prop in enumerate(props):
        regions.append({
            "area": int(prop.area),
            "centroid": (float(prop.centroid[0]), float(prop.centroid[1])),
            "bbox": tuple(int(v) for v in prop.bbox),
            "color": np.array([float(channels[k][position]) for k in range(3)]),
            "motionOverlap": float(overlaps[position]) if overlaps is not None else None,
        })
    return sorted(regions, key=lambda r: -r["area"])


def backSubRegions(frame, background, threshold, minArea, moving=None):
    # backsub III over the three colour bands, then the shared cleanup
    difference = np.sqrt(((frame - background) ** 2).sum(axis=2))
    return _describeRegions(_cleanAndLabel(difference > threshold), frame, minArea, moving)


def motionMask(frameA, frameB, threshold):
    # image differencing (Content/Motion.pdf): what changed between two action
    # frames, which is how a moving car is told apart from a static artifact
    difference = np.sqrt(((frameB - frameA) ** 2).sum(axis=2))
    return difference > threshold


def maskRegions(binary, frame, minArea):
    # same cleanup and region records, starting from an already-binary mask
    return _describeRegions(_cleanAndLabel(binary), frame, minArea)


def threeFrameRegions(previous, current, following, threshold, minArea):
    """Where the cars are in `current`, from image differencing alone.

    Differencing two frames marks both the position a car left and the position
    it arrived at.  Intersecting the difference against the previous frame with
    the difference against the next frame keeps only the middle position, which
    is where the car actually is in `current`.  This needs no background image,
    which matters because the shared background frames were shot from a
    different camera pose than distance races 1 and 2.
    """
    before = motionMask(previous, current, threshold)
    after = motionMask(current, following, threshold)
    return maskRegions(before & after, current, minArea)


def estimateCameraShift(previous, current, maxShift, step=1):
    """Global frame-to-frame camera shift, by direct search over translations.

    Several of these races were shot handheld, so the camera itself moves
    between frames.  When it does, the distance a car appears to travel across
    the image is its real motion plus the camera's, and the distance race is
    then scoring the photographer instead of the cars.

    The score is the median absolute difference over the whole overlap rather
    than the sum of squares.  The cars are a small part of the frame, so the
    median is set by the background and is not dragged around by the very
    objects whose motion is being separated out.  This is the still-camera
    assumption from Motion.pdf being checked rather than assumed.

    The returned shift is how far the image content moved between the two
    frames, in the same sense as a car's measured displacement, so it can be
    subtracted from one directly.
    """
    previousGrey = previous.mean(axis=2)
    currentGrey = current.mean(axis=2)
    rows, cols = previousGrey.shape

    best = (np.inf, (0, 0))
    for rowShift in range(-maxShift, maxShift + 1, step):
        for colShift in range(-maxShift, maxShift + 1, step):
            # content at y in `current` came from y - shift in `previous`
            r0, r1 = max(0, rowShift), min(rows, rows + rowShift)
            c0, c1 = max(0, colShift), min(cols, cols + colShift)
            if r1 - r0 < 8 or c1 - c0 < 8:
                continue
            overlapCurrent = currentGrey[r0:r1, c0:c1]
            overlapPrevious = previousGrey[r0 - rowShift:r1 - rowShift,
                                           c0 - colShift:c1 - colShift]
            score = float(np.median(np.abs(overlapCurrent - overlapPrevious)))
            if score < best[0]:
                best = (score, (rowShift, colShift))
    return best[1], best[0]


def cameraTrack(frames, maxShift=12, levels=2):
    """Cumulative camera position over a sequence.

    The search is a brute-force scan over translations, so it runs on a coarser
    pyramid level than the tracker uses and the answer is scaled back up.  At
    two extra levels the estimate is quantised to four pixels of the input
    frames, which is far finer than the drift being detected and cheap enough to
    run over every pair in a sequence.

    Returns the per-frame shift and the running total, in the units of the
    frames handed in.
    """
    coarse = [downsample(frame, levels) for frame in frames]
    scale = float(2 ** levels)

    shifts = [(0.0, 0.0)]
    for index in range(1, len(coarse)):
        shift, _ = estimateCameraShift(coarse[index - 1], coarse[index], maxShift)
        shifts.append((shift[0] * scale, shift[1] * scale))
    shifts = np.array(shifts)
    return shifts, np.cumsum(shifts, axis=0)


def keepDominant(regions, fraction=0.25):
    # drop regions far smaller than the largest one; the persistent corner
    # artifacts on the carpet are real differences but an order of magnitude
    # smaller than a car
    if not regions:
        return []
    largest = regions[0]["area"]
    return [r for r in regions if r["area"] >= fraction * largest]


def isFinishLine(region, imageWidth):
    # the finish line spans nearly the whole frame, no car does
    _, minCol, _, maxCol = region["bbox"]
    return (maxCol - minCol) > 0.8 * imageWidth


# ---------------------------------------------------------------------------
# Identity assignment by colour
# ---------------------------------------------------------------------------

def _normalizedRgb(color):
    # dividing by the total removes the illumination gradient, which matters on
    # the carpet where the left of the frame is far brighter than the right
    total = float(np.sum(color))
    if total <= 0:
        return np.zeros(3)
    return np.asarray(color, dtype=np.float64) / total


def colorScore(color, name):
    # how strongly a mean colour argues for one particular car
    r, g, b = _normalizedRgb(color)
    if name == "red":
        return r - max(g, b)
    if name == "blue":
        return b - max(r, g)
    if name == "yellow":
        return min(r, g) - b
    raise ValueError(f"unknown car name {name!r}")


def assignIdentities(regions, carNames):
    # try both ways of pairing the two regions with the two car names and keep
    # the pairing with the higher total score; this separates yellow from red,
    # which a single colour channel cannot do
    first, second = carNames
    straight = colorScore(regions[0]["color"], first) + colorScore(regions[1]["color"], second)
    swapped = colorScore(regions[1]["color"], first) + colorScore(regions[0]["color"], second)
    if straight >= swapped:
        return {first: regions[0], second: regions[1]}, float(straight - swapped)
    return {first: regions[1], second: regions[0]}, float(swapped - straight)


# a genuine car scores 0.11 to 0.28 for its own colour; a grey shadow or a patch
# of carpet scores under 0.03, so this floor separates the two cleanly
MIN_COLOR_SCORE = 0.08

# the two cars are the same kind of object, so one region being several times
# the area of the other means at least one of them is not a car
MAX_AREA_RATIO = 4.0


def pairQuality(regions, carNames):
    """Score a candidate pair of regions as "these really are the two cars".

    A threshold that leaves exactly two regions is not enough on its own: on the
    speed race a low threshold leaves the blue car plus a large grey shadow, and
    the pair passes a count test.  Requiring each region to actually look like
    the colour it is being assigned rejects that.
    """
    named, margin = assignIdentities(regions, carNames)
    weakest = min(colorScore(named[name]["color"], name) for name in named)
    areas = sorted(region["area"] for region in regions)
    ratio = areas[1] / max(areas[0], 1)
    valid = weakest >= MIN_COLOR_SCORE and ratio <= MAX_AREA_RATIO
    return named, float(margin), float(weakest), float(ratio), bool(valid)


# ---------------------------------------------------------------------------
# Track initialization
# ---------------------------------------------------------------------------

def _bestCandidate(candidates):
    # every threshold that produced a valid pair is a candidate; keep the one
    # whose regions look most convincingly like their assigned colours
    valid = [c for c in candidates if c["valid"]]
    if not valid:
        return None
    return max(valid, key=lambda c: c["weakest"])


def findCarsBySubtraction(frame, nextFrame, background, carNames, minArea,
                          motionThreshold=0.10):
    """Two cars from background subtraction, keeping only the regions that move.

    Works when the reference background was shot from the same camera pose as
    the race, which is true for the speed race and Distance Race New.  The
    finish line is pulled out separately because it is legitimately large and
    legitimately static.
    """
    moving = motionMask(frame, nextFrame, motionThreshold)
    candidates = []
    for threshold in THRESHOLD_SWEEP:
        regions = backSubRegions(frame, background, threshold, minArea, moving=moving)
        finishLine = next((r for r in regions if isFinishLine(r, frame.shape[1])), None)
        cars = [region for region in regions
                if region is not finishLine and region["motionOverlap"] >= MOTION_OVERLAP_MIN]
        cars = keepDominant(cars)
        if len(cars) != 2:
            continue
        _, margin, weakest, ratio, valid = pairQuality(cars, carNames)
        candidates.append({"cars": cars, "finishLine": finishLine,
                           "threshold": float(threshold), "method": "backsub",
                           "weakest": weakest, "margin": margin,
                           "ratio": ratio, "valid": valid})
    return _bestCandidate(candidates)


def findCarsByDifferencing(previous, current, following, carNames, minArea):
    """Two cars from three-frame differencing, with no background image at all."""
    candidates = []
    for threshold in THRESHOLD_SWEEP:
        regions = keepDominant(threeFrameRegions(previous, current, following,
                                                 threshold, minArea))
        if len(regions) != 2:
            continue
        _, margin, weakest, ratio, valid = pairQuality(regions, carNames)
        candidates.append({"cars": regions, "finishLine": None,
                           "threshold": float(threshold), "method": "differencing",
                           "weakest": weakest, "margin": margin,
                           "ratio": ratio, "valid": valid})
    return _bestCandidate(candidates)


def locateCars(frames, background, index, carNames, minArea):
    """Find both cars in frames[index], preferring background subtraction and
    falling back to differencing when no threshold gives a convincing pair."""
    if background is not None and index + 1 < len(frames):
        found = findCarsBySubtraction(frames[index], frames[index + 1], background,
                                      carNames, minArea)
        if found is not None:
            return found
    if 0 < index < len(frames) - 1:
        return findCarsByDifferencing(frames[index - 1], frames[index],
                                      frames[index + 1], carNames, minArea)
    return None


def loadReduced(loadFrame, count, background=None, levels=2):
    """Load a sequence straight to a reduced pyramid level, one frame at a time.

    Each full-resolution frame is downsampled and then dropped, so peak memory
    is one full frame plus the reduced sequence.  Holding all the full frames
    would be over 3 GB for the longer races, which does not fit alongside
    everything else on an 8 GB machine and sends the whole run into swap.
    """
    reduced = []
    fullShape = None
    for index in range(count):
        frame = loadFrame(index)
        if fullShape is None:
            fullShape = frame.shape
        reduced.append(downsample(frame, levels))
        del frame

    reducedBackground = downsample(background, levels) if background is not None else None
    factor = fullShape[0] / reduced[0].shape[0]
    minArea = 0.0005 * reduced[0].shape[0] * reduced[0].shape[1]
    return {"frames": reduced, "background": reducedBackground, "levels": levels,
            "factor": factor, "minArea": minArea, "fullShape": fullShape,
            "count": count}


def prepareSmall(frames, background, levels=2):
    # same thing for callers that already hold the frames, such as the unit tests
    return loadReduced(lambda index: frames[index], len(frames), background, levels)


def initTracks(reduced, carNames, startIndex=1):
    """Locate both cars and build their covariance models.

    Works on the reduced sequence from loadReduced, and returns coordinates
    scaled back up to full resolution so the overlays line up with the original
    frames.

    Initialization happens at `startIndex` rather than at frame 0 because
    three-frame differencing needs a frame on either side.  The tracker then
    searches frame 0 backwards, so no frame is dropped from the results.
    """
    smallBackground = reduced["background"]
    factor = reduced["factor"]
    minArea = reduced["minArea"]
    small = reduced["frames"]

    found = locateCars(small, smallBackground, startIndex, carNames, minArea)
    if found is None:
        raise RuntimeError(f"neither subtraction nor differencing found a convincing "
                           f"pair of cars at frame index {startIndex}")
    cars, finishLine = found["cars"], found["finishLine"]
    threshold, method = found["threshold"], found["method"]

    named, margin = assignIdentities(cars, carNames)

    # the search radius has to cover the jumps the cars actually make, so measure
    # them over the next few frames instead of guessing a constant.  The median
    # is used rather than the maximum because one bad detection would otherwise
    # blow the radius up to the size of the whole frame.
    steps = []
    previousCenters = {name: np.array(region["centroid"]) for name, region in named.items()}
    for index in range(startIndex + 1, min(startIndex + 5, len(small) - 1)):
        later = locateCars(small, smallBackground, index, carNames, minArea)
        if later is None:
            continue
        laterNamed, _ = assignIdentities(later["cars"], carNames)
        for name in named:
            current = np.array(laterNamed[name]["centroid"])
            steps.append(float(np.linalg.norm(current - previousCenters[name])))
            previousCenters[name] = current

    typicalStep = float(np.median(steps)) if steps else 0.0
    # 2.5x the typical step leaves room for a car that accelerates, and the cap
    # keeps the search local even if the measurement went wrong
    searchRadius = float(np.clip(2.5 * typicalStep, 8.0, 0.18 * small[0].shape[0]))
    searchRadius *= factor

    tracks = {}
    for name, region in named.items():
        minRow, minCol, maxRow, maxCol = region["bbox"]
        tracks[name] = {
            "name": name,
            "startIndex": startIndex,
            "centers": [(region["centroid"][0] * factor, region["centroid"][1] * factor)],
            "sizes": [((maxRow - minRow) * factor, (maxCol - minCol) * factor)],
            "distances": [0.0],
            "model": None,
            "modelHistory": [],
            "initColor": region["color"],
        }

    return {
        "tracks": tracks,
        "threshold": threshold,
        "initMethod": method,
        "searchRadius": searchRadius,
        "identityMargin": margin,
        "finishLine": finishLine,
        "initScaleFactor": factor,
        "startIndex": startIndex,
    }


# ---------------------------------------------------------------------------
# Per-frame tracking
# ---------------------------------------------------------------------------

def predictCenter(history, index, step):
    """Constant-velocity prediction for the next frame in the walk direction.

    This is what lets a purely local search cope with jumps of several hundred
    pixels: the search box is centred on where the car should be, not on where
    it last was.
    """
    last = np.array(history[index - step]["center"])
    older = history.get(index - 2 * step)
    if older is None:
        return (float(last[0]), float(last[1]))
    predicted = last + (last - np.array(older["center"]))
    return (float(predicted[0]), float(predicted[1]))


def searchWindow(image, model, center, size, radius, scales, blocked=None):
    """Coarse-to-fine scan around a predicted centre, over a few window scales."""
    height0, width0 = size

    def evaluate(rowOffset, colOffset, scale):
        height = max(4.0, height0 * scale)
        width = max(4.0, width0 * scale)
        centerRow = center[0] + rowOffset
        centerCol = center[1] + colOffset
        if blocked is not None and _overlaps((centerRow, centerCol), (height, width), blocked):
            return np.inf, None
        C = windowDescriptor(image, centerRow, centerCol, height, width)
        if C is None:
            return np.inf, None
        return covDistance(model, C), ((centerRow, centerCol), (height, width))

    best = (np.inf, None)
    step = 4
    span = int(round(radius))
    for rowOffset in range(-span, span + 1, step):
        for colOffset in range(-span, span + 1, step):
            for scale in scales:
                distance, result = evaluate(rowOffset, colOffset, scale)
                if distance < best[0]:
                    best = (distance, result)

    if best[1] is None:
        return best

    # refine at single-pixel steps around the coarse winner
    coarseCenter, coarseSize = best[1]
    baseRow = coarseCenter[0] - center[0]
    baseCol = coarseCenter[1] - center[1]
    coarseScale = coarseSize[0] / max(height0, 1e-9)
    for rowOffset in range(int(baseRow) - step, int(baseRow) + step + 1):
        for colOffset in range(int(baseCol) - step, int(baseCol) + step + 1):
            for scale in scales:
                if abs(scale - coarseScale) > 0.3:
                    continue
                distance, result = evaluate(rowOffset, colOffset, scale)
                if distance < best[0]:
                    best = (distance, result)
    return best


def _overlaps(center, size, blocked):
    # window centres and sizes, in (row, col) / (height, width) form
    blockedCenter, blockedSize = blocked
    rowGap = abs(center[0] - blockedCenter[0]) - (size[0] + blockedSize[0]) / 2.0
    colGap = abs(center[1] - blockedCenter[1]) - (size[1] + blockedSize[1]) / 2.0
    return rowGap < 0 and colGap < 0


def updateModel(state, C, distance, historyLength=3):
    """Slide-11 model update: average the recent models, but only accept a frame
    whose match was at least as good as usual, so a merge or an occlusion cannot
    be absorbed into the model."""
    accepted = state["accepted"]
    if accepted and distance > float(np.median(accepted)):
        return
    accepted.append(distance)
    history = state["modelHistory"]
    history.append(C)
    if len(history) > historyLength:
        history.pop(0)
    state["model"] = np.mean(np.stack(history, axis=0), axis=0)


def _walk(levelFrames, states, histories, startIndex, step, radius, scales, factor):
    # march away from the initialization frame one frame at a time, in whichever
    # direction `step` points; the model and the velocity both carry along
    indices = (range(startIndex + step, len(levelFrames)) if step > 0
               else range(startIndex + step, -1, -1))
    for index in indices:
        image = levelFrames[index]
        claimed = None
        # search the better-matching car first so it owns its window in a clash
        order = sorted(states, key=lambda n: states[n]["lastDistance"])
        for name in order:
            state = states[name]
            history = histories[name]
            predicted = predictCenter(history, index, step)
            size = history[index - step]["size"]
            distance, result = searchWindow(
                image, state["model"],
                (predicted[0] / factor, predicted[1] / factor),
                (size[0] / factor, size[1] / factor),
                radius, scales, blocked=claimed)

            if result is None:
                # nothing valid in range: coast on the constant-velocity prediction
                history[index] = {"center": predicted, "size": size, "distance": np.inf}
                state["lastDistance"] = np.inf
                continue

            center, windowSize = result
            claimed = (center, windowSize)
            fullCenter = (center[0] * factor, center[1] * factor)
            fullSize = (windowSize[0] * factor, windowSize[1] * factor)
            history[index] = {"center": fullCenter, "size": fullSize, "distance": distance}
            state["lastDistance"] = distance

            C = windowDescriptor(image, center[0], center[1], windowSize[0], windowSize[1])
            if C is not None:
                updateModel(state, C, distance)


def runTracker(reduced, setup, scales=(0.85, 1.0, 1.15)):
    """Track both cars across the whole sequence at the derived pyramid level.

    Tracking starts at the initialization frame and walks outward in both
    directions, so initializing at frame 1 (which three-frame differencing
    requires) still produces a result for frame 0.
    """
    tracks = setup["tracks"]
    startIndex = setup["startIndex"]
    frames = reduced["frames"]
    largestCar = max(track["sizes"][0][0] for track in tracks.values())

    # the reduced sequence is already a few levels down, so only the difference
    # has to be taken here; asking for more resolution than was loaded would
    # mean re-reading every frame, so the request is clamped and reported
    wanted = chooseLevel(largestCar)
    level = max(wanted, reduced["levels"])
    extra = level - reduced["levels"]
    factor = float(2 ** level)

    levelFrames = frames if extra == 0 else [downsample(f, extra) for f in frames]
    radius = setup["searchRadius"] / factor

    states, histories = {}, {}
    for name, track in tracks.items():
        centerRow, centerCol = track["centers"][0]
        height, width = track["sizes"][0]
        # the initial model is allowed to be clipped harder than a search
        # candidate: a car that starts half out of shot still has to be modelled
        C = windowDescriptor(levelFrames[startIndex], centerRow / factor, centerCol / factor,
                             height / factor, width / factor, minCoverage=0.25)
        if C is None:
            raise RuntimeError(f"car {name} is too far outside the frame to model "
                               f"at frame {startIndex + 1}")
        states[name] = {"model": C, "modelHistory": [C], "accepted": [], "lastDistance": 0.0}
        histories[name] = {startIndex: {"center": (centerRow, centerCol),
                                        "size": (height, width),
                                        "distance": 0.0}}

    _walk(levelFrames, states, histories, startIndex, +1, radius, scales, factor)

    # reset to the initial model before walking backwards, so the backward pass
    # is not biased by whatever the forward pass learned near the end
    for name, track in tracks.items():
        C = states[name]["modelHistory"][0]
        states[name] = {"model": C, "modelHistory": [C], "accepted": [], "lastDistance": 0.0}
    _walk(levelFrames, states, histories, startIndex, -1, radius, scales, factor)

    for name, track in tracks.items():
        history = histories[name]
        ordered = [history[i] for i in range(len(frames))]
        track["centers"] = [entry["center"] for entry in ordered]
        track["sizes"] = [entry["size"] for entry in ordered]
        track["distances"] = [entry["distance"] for entry in ordered]

    setup["level"] = level
    setup["wantedLevel"] = wanted
    return setup


# ---------------------------------------------------------------------------
# Motion analysis
# ---------------------------------------------------------------------------

def referenceHeight(tracks):
    """One car length shared by both cars in a sequence.

    The two toy cars are the same physical size, so they must be measured
    against the same ruler.  Using each car's own initial box instead lets a
    tight detection box inflate that car's distance for the whole race: in
    Distance Race 3 the initial boxes were 320 px and 508 px tall for cars that
    are really the same length, which alone was a factor of 1.6 on the result.
    """
    return float(np.mean([track["sizes"][0][0] for track in tracks.values()]))


def frameSpeeds(track, cameraShifts=None, reference=None):
    """Per-frame speed in pixels and in car lengths.

    Three corrections are applied, and each of them changes the answer on this
    data:

    Perspective.  The cars recede from the camera, so raw pixel speed falls even
    at constant real speed.  The car's window shrinks by the same factor, so the
    ratio of its current height to its starting height is the scale of that
    frame, and dividing by it undoes the perspective.

    A shared ruler.  The scale factor above is relative to each car's own first
    frame, so it cancels how tightly that car happened to be boxed.  Distances
    are then expressed in one shared car length, which makes the two cars
    comparable to each other and to the other sequences.

    Camera motion.  Where the camera was handheld, part of a car's apparent
    displacement is the camera moving.  Subtracting the estimated global shift
    leaves motion relative to the ground, which is what a distance race asks.
    """
    centers = np.array(track["centers"])
    sizes = np.array(track["sizes"])
    deltas = np.diff(centers, axis=0)

    if cameraShifts is not None:
        # cameraShifts[i] is how far the camera moved into frame i, so removing
        # it converts frame-relative motion into ground-relative motion
        deltas = deltas - np.asarray(cameraShifts)[1:len(centers)]

    steps = np.linalg.norm(deltas, axis=1)

    startHeight = sizes[0, 0]
    ruler = float(reference) if reference is not None else float(startHeight)
    relativeScale = ((sizes[1:, 0] + sizes[:-1, 0]) / 2.0) / startHeight
    normalized = steps / (ruler * relativeScale)

    return {
        "pixels": steps,
        "carLengths": normalized,
        "cumulativePixels": np.concatenate([[0.0], np.cumsum(steps)]),
        "cumulativeCarLengths": np.concatenate([[0.0], np.cumsum(normalized)]),
    }


def leadingEdge(track, index):
    # the cars travel up the rotated frame, so the front of the car is its
    # smallest row value
    centerRow, _ = track["centers"][index]
    height, _ = track["sizes"][index]
    return centerRow - height / 2.0


def gapBetween(tracks, index):
    # signed distance between the two leading edges, plus who is in front
    names = list(tracks)
    edges = {name: leadingEdge(tracks[name], index) for name in names}
    leader = min(names, key=lambda n: edges[n])
    trailer = max(names, key=lambda n: edges[n])
    return leader, trailer, abs(edges[leader] - edges[trailer])


# ---------------------------------------------------------------------------
# Unit checks: every piece is validated against a reference it cannot fake
# ---------------------------------------------------------------------------

def _syntheticFrame(rows, cols, center, size, color, seed):
    # textured background with one solid coloured rectangle on it, so the true
    # answer to a search is known exactly
    rng = np.random.default_rng(seed)
    image = 0.45 + 0.05 * rng.standard_normal((rows, cols, 3))
    r0 = int(center[0] - size[0] / 2)
    c0 = int(center[1] - size[1] / 2)
    image[r0:r0 + size[0], c0:c0 + size[1], :] = np.asarray(color)
    return np.clip(image, 0.0, 1.0)


def _runUnitChecks():
    failures = []

    def check(name, condition, detail=""):
        print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    print("covariance distance")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 5))
    C1 = np.cov(A, rowvar=False, bias=True)
    B = rng.standard_normal((200, 5))
    C2 = np.cov(B, rowvar=False, bias=True)
    check("rho(C, C) == 0", abs(covDistance(C1, C1)) < 1e-9, f"got {covDistance(C1, C1):.3e}")
    check("rho is symmetric", abs(covDistance(C1, C2) - covDistance(C2, C1)) < 1e-9,
          f"{covDistance(C1, C2):.6f} vs {covDistance(C2, C1):.6f}")
    check("rho(C1, C2) > 0 for different matrices", covDistance(C1, C2) > 0.1,
          f"got {covDistance(C1, C2):.4f}")
    # the slides note scale invariance; a uniform rescale of the colour channels
    # must not change the distance
    D = np.diag([1.0, 1.0, 7.0, 7.0, 7.0])
    check("rho is invariant to per-channel rescaling",
          abs(covDistance(C1, C2) - covDistance(D @ C1 @ D, D @ C2 @ D)) < 1e-6)

    print("pyramid")
    image = _syntheticFrame(240, 320, (120, 160), (60, 40), (0.9, 0.1, 0.1), 1)
    check("one level halves both dimensions", pyramidDown(image).shape[:2] == (120, 160),
          str(pyramidDown(image).shape))
    check("three levels give an eighth", downsample(image, 3).shape[:2] == (30, 40),
          str(downsample(image, 3).shape))
    check("chooseLevel(600) == 3", chooseLevel(600) == 3, f"got {chooseLevel(600)}")
    check("chooseLevel(200) == 1", chooseLevel(200) == 1, f"got {chooseLevel(200)}")
    check("chooseLevel(80) == 0", chooseLevel(80) == 0, f"got {chooseLevel(80)}")

    print("search recovers a known shift")
    size = (60, 40)
    start = (120.0, 160.0)
    for shift in ((0, 0), (17, -23), (-31, 12)):
        first = _syntheticFrame(300, 380, start, size, (0.9, 0.1, 0.1), 2)
        truth = (start[0] + shift[0], start[1] + shift[1])
        second = _syntheticFrame(300, 380, truth, size, (0.9, 0.1, 0.1), 3)
        model = windowDescriptor(first, start[0], start[1], size[0], size[1])
        distance, result = searchWindow(second, model, start, size, 40, (1.0,))
        found = result[0]
        error = np.hypot(found[0] - truth[0], found[1] - truth[1])
        check(f"shift {shift} recovered", error <= 2.0,
              f"found ({found[0]:.1f}, {found[1]:.1f}) truth {truth}, error {error:.2f} px, "
              f"rho {distance:.4f}")

    print("colour identity assignment")
    red = np.array([0.75, 0.18, 0.20])
    blue = np.array([0.15, 0.30, 0.70])
    yellow = np.array([0.88, 0.80, 0.22])
    for names, colors in ((("red", "blue"), (red, blue)),
                          (("red", "yellow"), (red, yellow))):
        regions = [{"color": colors[0]}, {"color": colors[1]}]
        named, margin = assignIdentities(regions, names)
        straight = named[names[0]] is regions[0] and named[names[1]] is regions[1]
        check(f"{names[0]} vs {names[1]} assigned correctly", straight,
              f"margin {margin:.4f}")
        swapped = [{"color": colors[1]}, {"color": colors[0]}]
        named, margin = assignIdentities(swapped, names)
        check(f"{names[0]} vs {names[1]} assigned correctly when swapped",
              named[names[0]] is swapped[1] and named[names[1]] is swapped[0],
              f"margin {margin:.4f}")

    print("three-frame differencing isolates the middle position")
    a = _syntheticFrame(300, 380, (200.0, 160.0), size, (0.9, 0.1, 0.1), 4)
    b = _syntheticFrame(300, 380, (140.0, 160.0), size, (0.9, 0.1, 0.1), 4)
    c = _syntheticFrame(300, 380, (80.0, 160.0), size, (0.9, 0.1, 0.1), 4)
    found = keepDominant(threeFrameRegions(a, b, c, 0.30, 200))
    ok = len(found) == 1 and abs(found[0]["centroid"][0] - 140.0) < 5
    check("one region at the middle frame's position", ok,
          f"{len(found)} region(s), centroid "
          f"{tuple(round(v, 1) for v in found[0]['centroid']) if found else 'none'}")

    print("camera shift estimation")
    # a whole scene translated by a known amount, with a car moving inside it:
    # the estimator must report the camera's shift, not the car's
    rng2 = np.random.default_rng(11)
    scene = 0.4 + 0.15 * rng2.standard_normal((200, 260, 3))
    scene = np.clip(ndimage.gaussian_filter(scene, (2, 2, 0)), 0, 1)
    for trueShift in ((0, 0), (5, -3), (-7, 4)):
        shifted = np.roll(np.roll(scene, trueShift[0], axis=0), trueShift[1], axis=1)
        moved = shifted.copy()
        moved[40:70, 60:85, :] = [0.95, 0.1, 0.1] # a "car" that moved somewhere else
        base = scene.copy()
        base[95:125, 60:85, :] = [0.95, 0.1, 0.1]
        found, _ = estimateCameraShift(base, moved, 12)
        check(f"camera shift {trueShift} recovered", found == trueShift, f"got {found}")

    print("speed normalization cancels a pure scale change")
    track = {
        # a car receding: constant real speed, so pixel steps and car height
        # shrink together and car-lengths per frame must stay constant
        "centers": [(0.0, 0.0), (100.0, 0.0), (180.0, 0.0), (244.0, 0.0)],
        "sizes": [(100.0, 50.0), (80.0, 40.0), (64.0, 32.0), (51.2, 25.6)],
    }
    speeds = frameSpeeds(track)
    spread = float(np.ptp(speeds["carLengths"]))
    check("car-lengths per frame stays flat while pixels per frame falls",
          spread < 0.02 and speeds["pixels"][0] > speeds["pixels"][-1],
          f"pixels {np.round(speeds['pixels'], 1)}, "
          f"carLengths {np.round(speeds['carLengths'], 4)}")

    # two identical journeys boxed at different tightness must score the same
    tight = {"centers": [(0.0, 0.0), (100.0, 0.0), (180.0, 0.0)],
             "sizes": [(50.0, 25.0), (40.0, 20.0), (32.0, 16.0)]}
    loose = {"centers": [(0.0, 0.0), (100.0, 0.0), (180.0, 0.0)],
             "sizes": [(100.0, 50.0), (80.0, 40.0), (64.0, 32.0)]}
    shared = referenceHeight({"tight": tight, "loose": loose})
    tightTotal = frameSpeeds(tight, reference=shared)["cumulativeCarLengths"][-1]
    looseTotal = frameSpeeds(loose, reference=shared)["cumulativeCarLengths"][-1]
    check("box tightness does not change the distance",
          abs(tightTotal - looseTotal) < 1e-9,
          f"tight {tightTotal:.4f} vs loose {looseTotal:.4f}")
    # and the shared ruler has to be doing real work: without it the same
    # journey scores differently purely because one box was tighter
    tightAlone = frameSpeeds(tight)["cumulativeCarLengths"][-1]
    looseAlone = frameSpeeds(loose)["cumulativeCarLengths"][-1]
    check("each car's own box would have given the wrong answer",
          abs(tightAlone - looseAlone) > 1.0,
          f"self-normalized: tight {tightAlone:.4f} vs loose {looseAlone:.4f} "
          f"for identical journeys")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
        return 1
    print("all unit checks passed")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_runUnitChecks())
