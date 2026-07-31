# distance races: track both cars -> winner = greatest total distance, in car lengths not pixels
# run: python girish_distance_tracking.py
# 4 sequences: distanceNew (fleece, 7 frames), distance1/2/3 (same carpet, 11/23/23 frames)
# races 1-3 share a background folder shot from a different pose -> backsub unusable there,
# falls back to three-frame differencing, which needs no background image at all

import numpy as np

import girish_evaluation as ge
import girish_overlays as go
import girish_sequences as gs
import girish_tracking as gt

KEYS = ("distanceNew", "distance1", "distance2", "distance3")
OUTPUT_ROOT = "girish_output"


def runRace(key, render=True):
    seq = gs.SEQUENCES[key]
    print("=" * 78)
    print(f"=== {seq['label']} ===")
    print(f"  {seq['note']}")

    reduced = gt.loadReduced(lambda i: gs.loadFrame(key, i), gs.frameCount(key),
                             gs.loadBackground(key))
    frames = reduced["frames"]
    print(f"  {len(frames)} frames of {reduced['fullShape'][1]} x {reduced['fullShape'][0]}, "
          f"held at {frames[0].shape[1]} x {frames[0].shape[0]}")

    # try successive start frames -> the first frame of a race can have a hand in it or a car half out of shot
    setup = None
    for startIndex in range(1, min(6, len(frames) - 1)):
        try:
            setup = gt.initTracks(reduced, seq["carNames"], startIndex=startIndex)
            gt.runTracker(reduced, setup)
            break
        except RuntimeError as error:
            print(f"  could not start at frame {startIndex + 1}: {error}")
            setup = None
    if setup is None:
        print("  SKIPPED: no frame in this sequence yielded a usable pair of cars")
        return None

    tracks = setup["tracks"]

    print(f"  initialized at frame {setup['startIndex'] + 1} by {setup['initMethod']} "
          f"at threshold {setup['threshold']:.2f}, identity margin {setup['identityMargin']:.3f}")
    print(f"  derived pyramid level {setup['level']} "
          f"(car is {max(t['sizes'][0][0] for t in tracks.values()):.0f} px tall), "
          f"derived search radius {setup['searchRadius']:.0f} px")

    # check the still-camera assumption rather than trust it -> several races were shot handheld
    shifts, cumulative = gt.cameraTrack(frames)
    shiftsFull = shifts * reduced["factor"]
    perFrame = np.linalg.norm(shiftsFull, axis=1)
    carHeight = max(t["sizes"][0][0] for t in tracks.values())
    print(f"  camera motion: median {np.median(perFrame[1:]):.0f} px/frame, "
          f"max {perFrame.max():.0f} px, total drift "
          f"{np.linalg.norm(cumulative[-1] * reduced['factor']):.0f} px "
          f"({np.linalg.norm(cumulative[-1] * reduced['factor']) / carHeight:.2f} car lengths)")
    steady = np.median(perFrame[1:]) < 0.05 * carHeight
    verdict = "steady" if steady else "MOVING, so distances are corrected for it"
    print(f"  treating the camera as {verdict}")

    ruler = gt.referenceHeight(tracks)
    print(f"  one car length = {ruler:.0f} px at the start "
          f"(shared by both cars, since they are the same toy)")
    speeds = {name: gt.frameSpeeds(track, cameraShifts=shiftsFull, reference=ruler)
              for name, track in tracks.items()}
    rawSpeeds = {name: gt.frameSpeeds(track, reference=ruler)
                 for name, track in tracks.items()}

    print("\n  per-frame speed and running total")
    for index in range(len(frames)):
        row = f"   frame {index + 1:2d} "
        for name in tracks:
            if index == 0:
                row += f"| {name:>7s}            start                 "
            else:
                row += (f"| {name:>7s} {speeds[name]['pixels'][index - 1]:7.1f} px/f "
                        f"{speeds[name]['carLengths'][index - 1]:5.2f} car/f "
                        f"total {speeds[name]['cumulativeCarLengths'][index]:6.2f} ")
        print(row)

    totals = {name: float(speeds[name]["cumulativeCarLengths"][-1]) for name in tracks}
    totalPixels = {name: float(speeds[name]["cumulativePixels"][-1]) for name in tracks}
    winner = max(totals, key=lambda n: totals[n])

    print()
    for name in tracks:
        raw = float(rawSpeeds[name]["cumulativeCarLengths"][-1])
        print(f"    {name:7s} travelled {totals[name]:6.2f} car lengths "
              f"({totalPixels[name]:8.0f} px along the tracked path)"
              f"   uncorrected {raw:6.2f}")
    print(f"  WINNER: the {winner} car travelled furthest")

    print()
    detections = ge.detectionsPerFrame(reduced, seq["carNames"])
    ge.printEvaluation(seq["label"], tracks, detections)

    if render:
        outputDir = f"{OUTPUT_ROOT}/{key}"
        print("\n  rendering annotated frames")
        paths = go.renderSequence(reduced, tracks, speeds, seq["kind"], outputDir,
                                  seq["label"], winner=winner, winnerFrom=len(frames) - 1)
        summary = go.renderSummary(reduced, tracks, speeds, seq["kind"],
                                   f"{outputDir}/summary.png", seq["label"])
        print(f"    {len(paths)} frames written to {outputDir}/")
        print(f"    summary figure written to {summary}")

    return {"key": key, "label": seq["label"], "winner": winner, "totals": totals,
            "setup": setup, "tracks": tracks, "speeds": speeds, "detections": detections}


def main():
    results = []
    for key in KEYS:
        try:
            result = runRace(key)
        except Exception as error: # one difficult race must not stop the rest
            print(f"  FAILED on {key}: {type(error).__name__}: {error}")
            result = None
        if result is not None:
            results.append(result)

    print("=" * 78)
    print("=== all distance races ===")
    print(f"  {'race':22s} {'init':14s} {'level':6s} {'winner':8s}  totals (car lengths)")
    for result in results:
        totals = "  ".join(f"{name} {value:.2f}" for name, value in result["totals"].items())
        print(f"  {result['label']:22s} {result['setup']['initMethod']:14s} "
              f"{result['setup']['level']:<6d} {result['winner']:8s}  {totals}")


if __name__ == "__main__":
    main()
