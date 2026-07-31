# diagnostic displays for the tracking half, matches patel_*_test.py convention
# run: python girish_tracking_test.py [key]   keys: speed, distanceNew, distance1, distance2, distance3
# 1 init masks: backsub vs three-frame differencing side by side
# 2 threshold sweep: which thresholds leave 2 car-sized regions and how convincing each pair is
# 3 covariance match-distance map over a whole frame, what the tracker minimizes
# 4 tracked window crop per frame, side by side -> fastest way to confirm identity never swapped

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import girish_evaluation as ge
import girish_overlays as go
import girish_sequences as gs
import girish_tracking as gt

OUTPUT_ROOT = "girish_output/diagnostics"


def showInitMasks(reduced, key, index, outputDir):
    # backsub next to three-frame differencing, at one frame
    frames = reduced["frames"]
    background = reduced["background"]
    current = frames[index]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(current)
    axes[0, 0].set_title(f"frame {index + 1}")
    axes[1, 0].imshow(background)
    axes[1, 0].set_title("reference background")

    backsubDifference = np.sqrt(((current - background) ** 2).sum(axis=2))
    axes[0, 1].imshow(backsubDifference, cmap="magma")
    axes[0, 1].set_title(f"backsub distance\nmedian {np.median(backsubDifference):.3f}")

    beforeMask = gt.motionMask(frames[index - 1], current, 0.15)
    afterMask = gt.motionMask(current, frames[index + 1], 0.15)
    axes[1, 1].imshow(beforeMask & afterMask, cmap="gray")
    axes[1, 1].set_title("three-frame difference\n(both neighbours)")

    for column, threshold in enumerate((0.25, 0.50), start=2):
        axes[0, column].imshow(backsubDifference > threshold, cmap="gray")
        axes[0, column].set_title(f"backsub > {threshold:.2f}")

        both = (gt.motionMask(frames[index - 1], current, threshold)
                & gt.motionMask(current, frames[index + 1], threshold))
        axes[1, column].imshow(both, cmap="gray")
        axes[1, column].set_title(f"differencing > {threshold:.2f}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{gs.SEQUENCES[key]['label']}: how the cars are first located", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outputDir, "01_init_masks.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def showThresholdSweep(reduced, key, index, outputDir):
    # which thresholds leave a convincing pair, and how convincing
    carNames = gs.SEQUENCES[key]["carNames"]
    frames = reduced["frames"]
    minArea = reduced["minArea"]
    background = reduced["background"]

    rows = []
    for threshold in gt.THRESHOLD_SWEEP:
        for method, regions in (
            ("backsub", gt.keepDominant([
                r for r in gt.backSubRegions(
                    frames[index], background, threshold, minArea,
                    moving=gt.motionMask(frames[index], frames[index + 1], 0.10))
                if not gt.isFinishLine(r, frames[index].shape[1])
                and r["motionOverlap"] >= gt.MOTION_OVERLAP_MIN])),
            ("differencing", gt.keepDominant(gt.threeFrameRegions(
                frames[index - 1], frames[index], frames[index + 1], threshold, minArea))),
        ):
            if len(regions) != 2:
                rows.append((method, float(threshold), len(regions), np.nan, np.nan, False))
                continue
            _, _, weakest, ratio, valid = gt.pairQuality(regions, carNames)
            rows.append((method, float(threshold), 2, weakest, ratio, valid))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for column, method in enumerate(("backsub", "differencing")):
        subset = [r for r in rows if r[0] == method]
        thresholds = [r[1] for r in subset]
        counts = [r[2] for r in subset]
        scores = [r[3] for r in subset]

        axes[column].bar(thresholds, counts, width=0.03, color="#8899aa", label="regions found")
        axes[column].axhline(2, color="black", linestyle="--", linewidth=1)
        twin = axes[column].twinx()
        twin.plot(thresholds, scores, "o-", color="#ff8800", label="colour score")
        twin.axhline(gt.MIN_COLOR_SCORE, color="#ff8800", linestyle=":", linewidth=1.5)
        twin.set_ylabel("weakest colour score")
        for method_, threshold, count, weakest, ratio, valid in subset:
            if valid:
                axes[column].plot(threshold, count, marker="*", markersize=16, color="#00aa44")
        axes[column].set_xlabel("threshold")
        axes[column].set_ylabel("car-sized regions")
        axes[column].set_title(f"{method}  (a star marks an accepted pair)")
    fig.suptitle(f"{gs.SEQUENCES[key]['label']}: threshold sweep at frame {index + 1}",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(outputDir, "02_threshold_sweep.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def showMatchSurface(reduced, setup, index, outputDir, key, step=3):
    # covariance distance from one car's model to every window in the frame, on a step-size grid
    tracks = setup["tracks"]
    level = setup["level"]
    extra = level - reduced["levels"]
    image = reduced["frames"][index]
    if extra > 0:
        image = gt.downsample(image, extra)
    factor = float(2 ** level)

    name = list(tracks)[0]
    track = tracks[name]
    height, width = np.array(track["sizes"][index]) / factor
    model = gt.windowDescriptor(image, *(np.array(track["centers"][index]) / factor),
                                height, width)

    rows, cols = image.shape[:2]
    surface = np.full((rows // step, cols // step), np.nan)
    for r, centerRow in enumerate(range(0, rows, step)):
        if r >= surface.shape[0]:
            break
        for c, centerCol in enumerate(range(0, cols, step)):
            if c >= surface.shape[1]:
                break
            C = gt.windowDescriptor(image, centerRow, centerCol, height, width)
            if C is not None:
                surface[r, c] = gt.covDistance(model, C)

    best = np.unravel_index(np.nanargmin(surface), surface.shape)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(image)
    axes[0].plot(np.array(track["centers"][index])[1] / factor,
                 np.array(track["centers"][index])[0] / factor,
                 "+", color=go.carColor(name), markersize=16, markeredgewidth=2)
    axes[0].set_title(f"frame {index + 1}, tracking the {name} car")
    surfaceImage = axes[1].imshow(surface, cmap="jet")
    axes[1].plot(best[1], best[0], "w+", markersize=14, markeredgewidth=2)
    fig.colorbar(surfaceImage, ax=axes[1])
    axes[1].set_title("covariance distance to the model\n(white cross = global minimum)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{gs.SEQUENCES[key]['label']}: what the tracker minimizes", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outputDir, "03_match_surface.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def showTrackedCrops(reduced, setup, outputDir, key):
    # every tracked window, in order, one row per car -> a swap shows as the wrong colour in a row
    tracks = setup["tracks"]
    frames = reduced["frames"]
    scale = 1.0 / reduced["factor"]
    count = len(frames)

    fig, axes = plt.subplots(len(tracks), count, figsize=(1.4 * count, 3.4 * len(tracks)))
    axes = np.atleast_2d(axes)
    for row, (name, track) in enumerate(tracks.items()):
        for index in range(count):
            centerRow, centerCol = np.array(track["centers"][index]) * scale
            height, width = np.array(track["sizes"][index]) * scale
            r0 = max(0, int(centerRow - height / 2))
            c0 = max(0, int(centerCol - width / 2))
            crop = frames[index][r0:int(centerRow + height / 2), c0:int(centerCol + width / 2)]
            ax = axes[row, index]
            if crop.size:
                ax.imshow(crop)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(go.carColor(name))
                spine.set_linewidth(2.5)
            if row == 0:
                ax.set_title(f"{index + 1}", fontsize=9)
            if index == 0:
                ax.set_ylabel(name, fontsize=11, color=go.carColor(name))
    fig.suptitle(f"{gs.SEQUENCES[key]['label']}: tracked window per frame "
                 f"(a swap would show as the wrong colour in a row)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outputDir, "04_tracked_crops.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def main(key):
    seq = gs.SEQUENCES[key]
    outputDir = os.path.join(OUTPUT_ROOT, key)
    os.makedirs(outputDir, exist_ok=True)
    print(f"=== diagnostics for {seq['label']} ===")

    reduced = gt.loadReduced(lambda i: gs.loadFrame(key, i), gs.frameCount(key),
                             gs.loadBackground(key))

    setup = None
    for startIndex in range(1, min(6, len(reduced["frames"]) - 1)):
        try:
            setup = gt.initTracks(reduced, seq["carNames"], startIndex=startIndex)
            break
        except RuntimeError as error:
            print(f"  init at frame {startIndex + 1} failed: {error}")
    if setup is None:
        print("  cannot initialize this sequence, nothing to diagnose")
        return
    gt.runTracker(reduced, setup)
    index = setup["startIndex"]

    print("  " + showInitMasks(reduced, key, index, outputDir))
    print("  " + showThresholdSweep(reduced, key, index, outputDir))
    print("  " + showMatchSurface(reduced, setup, index, outputDir, key))
    print("  " + showTrackedCrops(reduced, setup, outputDir, key))

    detections = ge.detectionsPerFrame(reduced, seq["carNames"])
    ge.printEvaluation(seq["label"], setup["tracks"], detections)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "speed")
