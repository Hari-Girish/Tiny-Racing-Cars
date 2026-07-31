# tracking evaluated on 3 things vs independently recomputed detections, not the tracker's own output:
# identity held -> does the tracked window sit on the correctly-coloured region
# agreement -> px distance between tracked centre and detected centroid
# consistency -> per-frame displacement / match distance, spike = lost track

import numpy as np

import girish_tracking as gt


# one entry per frame: car name -> region, or None when detection found no clean pair
# a None frame is itself a result -> where detection fails and the tracker carries identity alone
def detectionsPerFrame(reduced, carNames):
    smallBackground = reduced["background"]
    factor = reduced["factor"]
    minArea = reduced["minArea"]
    small = reduced["frames"]

    perFrame = []
    for index in range(len(small)):
        found = gt.locateCars(small, smallBackground, index, carNames, minArea)
        if found is None:
            perFrame.append(None)
            continue
        named, _ = gt.assignIdentities(found["cars"], carNames)
        scaled = {}
        for name, region in named.items():
            minRow, minCol, maxRow, maxCol = region["bbox"]
            scaled[name] = {
                "centroid": (region["centroid"][0] * factor, region["centroid"][1] * factor),
                "bbox": (minRow * factor, minCol * factor, maxRow * factor, maxCol * factor),
                "color": region["color"],
            }
        perFrame.append(scaled)
    return perFrame


def evaluateTracks(tracks, detections):
    # identity-held count and tracked-vs-detected agreement, over frames detection could see
    comparable = 0
    held = 0
    errors = []
    for index, frame in enumerate(detections):
        if frame is None:
            continue
        comparable += 1
        frameHeld = True
        for name, track in tracks.items():
            centerRow, centerCol = track["centers"][index]
            minRow, minCol, maxRow, maxCol = frame[name]["bbox"]
            inside = (minRow <= centerRow <= maxRow) and (minCol <= centerCol <= maxCol)
            frameHeld = frameHeld and inside
            detected = frame[name]["centroid"]
            errors.append(float(np.hypot(centerRow - detected[0], centerCol - detected[1])))
        if frameHeld:
            held += 1

    errors = np.array(errors) if errors else np.array([np.nan])
    return {
        "comparableFrames": comparable,
        "detectorBlindFrames": len(detections) - comparable,
        "identityHeld": held,
        "meanError": float(np.nanmean(errors)),
        "maxError": float(np.nanmax(errors)),
    }


def consistencyReport(tracks):
    # largest jump in displacement and in match distance, per car
    report = {}
    for name, track in tracks.items():
        centers = np.array(track["centers"])
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        distances = np.array(track["distances"])
        finite = distances[np.isfinite(distances)]
        report[name] = {
            "meanStep": float(np.mean(steps)) if steps.size else 0.0,
            "maxStep": float(np.max(steps)) if steps.size else 0.0,
            "meanMatch": float(np.mean(finite)) if finite.size else float("nan"),
            "maxMatch": float(np.max(finite)) if finite.size else float("nan"),
            "coastedFrames": int(np.sum(~np.isfinite(distances))),
        }
    return report


def printEvaluation(label, tracks, detections):
    summary = evaluateTracks(tracks, detections)
    consistency = consistencyReport(tracks)

    print(f"  evaluation for {label}")
    print(f"    frames where detection saw both cars : {summary['comparableFrames']}")
    print(f"    frames where detection could not     : {summary['detectorBlindFrames']}"
          f"   (tracker carried these alone)")
    print(f"    identity held                        : "
          f"{summary['identityHeld']} / {summary['comparableFrames']}")
    print(f"    tracked vs detected centre           : "
          f"mean {summary['meanError']:.1f} px, max {summary['maxError']:.1f} px")
    for name, values in consistency.items():
        print(f"    {name:7s} step mean {values['meanStep']:7.1f} px  max {values['maxStep']:7.1f} px"
              f"   match mean {values['meanMatch']:.3f}  max {values['maxMatch']:.3f}"
              f"   coasted {values['coastedFrames']}")
    return summary, consistency
