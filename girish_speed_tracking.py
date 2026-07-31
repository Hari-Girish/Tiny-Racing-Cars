# speed race: track both cars -> watch each leading edge -> first to cross the finish row wins
# run: python girish_speed_tracking.py
# from frame 15 the leading car merges into the finish-line region and backsub loses it entirely,
# but the tracker searches the raw image, not the detection mask, so it holds both identities through it

import numpy as np

import girish_evaluation as ge
import girish_overlays as go
import girish_sequences as gs
import girish_tracking as gt

KEY = "speed"
OUTPUT_DIR = "girish_output/speed"


def findFinishRow(setup):
    # the finish-line region comes back in quarter-scale coordinates, so scale
    # its centroid row up to match the tracks
    region = setup["finishLine"]
    if region is None:
        return None
    return region["centroid"][0] * setup["initScaleFactor"]


def declareWinner(tracks, finishRow):
    # first frame in which a car's leading edge crosses the finish line
    if finishRow is None:
        return None, None
    crossings = {}
    for name, track in tracks.items():
        for index in range(len(track["centers"])):
            if gt.leadingEdge(track, index) <= finishRow:
                crossings[name] = index
                break
    if not crossings:
        return None, None
    winner = min(crossings, key=lambda n: (crossings[n], gt.leadingEdge(tracks[n], crossings[n])))
    return winner, crossings[winner]


def main():
    seq = gs.SEQUENCES[KEY]
    print(f"=== {seq['label']} ===")
    print(f"  {seq['note']}")

    reduced = gt.loadReduced(lambda i: gs.loadFrame(KEY, i), gs.frameCount(KEY),
                             gs.loadBackground(KEY))
    frames = reduced["frames"]
    print(f"  {len(frames)} frames of {reduced['fullShape'][1]} x {reduced['fullShape'][0]}, "
          f"held at {frames[0].shape[1]} x {frames[0].shape[0]}")

    setup = gt.initTracks(reduced, seq["carNames"], startIndex=1)
    gt.runTracker(reduced, setup)
    tracks = setup["tracks"]

    print(f"  initialized by {setup['initMethod']} at threshold {setup['threshold']:.2f}, "
          f"identity margin {setup['identityMargin']:.3f}")
    print(f"  derived pyramid level {setup['level']} "
          f"(car is {max(t['sizes'][0][0] for t in tracks.values()):.0f} px tall), "
          f"derived search radius {setup['searchRadius']:.0f} px")

    ruler = gt.referenceHeight(tracks)
    speeds = {name: gt.frameSpeeds(track, reference=ruler)
              for name, track in tracks.items()}
    finishRow = findFinishRow(setup)
    winner, winnerFrame = declareWinner(tracks, finishRow)

    print(f"  finish line at row {finishRow:.0f}" if finishRow is not None
          else "  no finish line found")

    print("\n  per-frame speed")
    header = "   frame " + "".join(f"| {name:>26s} " for name in tracks)
    print(header)
    for index in range(len(frames)):
        row = f"   {index + 1:5d} "
        for name in tracks:
            if index == 0:
                row += "|            start           "
            else:
                row += (f"| {speeds[name]['pixels'][index - 1]:8.1f} px/f "
                        f"{speeds[name]['carLengths'][index - 1]:6.3f} car/f ")
        print(row)

    print()
    if winner is None:
        print("  no car crossed the finish line in this sequence")
    else:
        print(f"  WINNER: the {winner} car crosses the finish line at frame {winnerFrame + 1}")
        for name, track in tracks.items():
            print(f"    {name:7s} leading edge at frame {winnerFrame + 1}: "
                  f"row {gt.leadingEdge(track, winnerFrame):.0f}")

    print()
    detections = ge.detectionsPerFrame(reduced, seq["carNames"])
    ge.printEvaluation(seq["label"], tracks, detections)

    blind = [i + 1 for i, d in enumerate(detections) if d is None]
    if blind:
        print(f"    detection was blind on frames {blind}, tracker still reported both cars")

    print("\n  rendering annotated frames")
    paths = go.renderSequence(reduced, tracks, speeds, seq["kind"], OUTPUT_DIR, seq["label"],
                              finishLineRow=finishRow, winner=winner, winnerFrom=winnerFrame)
    summary = go.renderSummary(reduced, tracks, speeds, seq["kind"],
                               f"{OUTPUT_DIR}/summary.png", seq["label"],
                               finishLineRow=finishRow)
    print(f"    {len(paths)} frames written to {OUTPUT_DIR}/")
    print(f"    summary figure written to {summary}")


if __name__ == "__main__":
    main()
