"""Per-sequence configuration for the five race image sequences.

Every sequence lives under Data/, which is the master copy of the image data.
Data/Speed Race and Data/Distance Race New are byte-identical to the committed
SpeedRace/ and DistanceRaceNew/ folders, so reading everything from Data/ keeps
one source of truth.

Only the things that genuinely differ per sequence live here: where the files
are, which frames are usable, and how each frame is cropped.  Every tracking
parameter (pyramid level, search radius, window size, backsub threshold) is
derived from the data at run time in girish_tracking.py, so nothing in this file
is a tuned constant.
"""

import os

import numpy as np
from skimage import io

DATA_DIR = "Data" # All image sequences live under this folder


# Each entry describes one race.  "crop" is applied after the rotation and is
# stored as (rowStart, rowStop, colStart, colStop) with None meaning "no cut",
# which keeps the crops readable next to the reason they exist.
SEQUENCES = {
    "speed": {
        "label": "Speed Race",
        "folder": os.path.join(DATA_DIR, "Speed Race"),
        "pattern": "SPEED_{:02d}.jpeg",
        "frames": range(1, 20), # 19 action frames
        "bgPattern": "SPEED_BG{:02d}.jpeg",
        "bgFrames": range(1, 6), # 5 background frames, all in the same folder
        "bgFolder": os.path.join(DATA_DIR, "Speed Race"),
        "crop": (None, None, None, -100), # cut the right edge of the poster out
        "kind": "speed", # race type: first car across the finish line wins
        "carNames": ("red", "blue"),
        "note": "White poster, black tape finish line. From frame 15 the leading car "
                "merges into the finish-line region and detection loses it.",
    },
    "distanceNew": {
        "label": "Distance Race New",
        "folder": os.path.join(DATA_DIR, "Distance Race New"),
        "pattern": "DIST_NEW_{:02d}.jpeg",
        "frames": range(1, 8), # frame 8 has the blue car half out of the crop
        "bgPattern": "DIST_NEW_BG_{:02d}.jpeg",
        "bgFrames": range(1, 7),
        "bgFolder": os.path.join(DATA_DIR, "Distance Race New"),
        "crop": (300, None, None, None), # cut the wall at the top out
        "kind": "distance", # race type: car that travels furthest wins
        "carNames": ("red", "blue"),
        "note": "Beige fleece background. Cars recede strongly, region area falls "
                "roughly 4x across the sequence.",
    },
    "distance1": {
        "label": "Distance Race 1",
        "folder": os.path.join(DATA_DIR, "Distance Race 1"),
        "pattern": "DIST1_{:02d}.jpeg",
        "frames": range(1, 12), # 11 frames
        "bgPattern": "DIST_BG_{:02d}.jpeg",
        "bgFrames": range(1, 13),
        "bgFolder": os.path.join(DATA_DIR, "Background Distance Race"),
        "crop": (None, None, None, None),
        "kind": "distance",
        "carNames": ("red", "yellow"),
        "note": "Textured office carpet with a strong left-bright to right-dark "
                "illumination gradient. Shortest sequence of the carpet races.",
    },
    "distance2": {
        "label": "Distance Race 2",
        "folder": os.path.join(DATA_DIR, "Distance Race 2"),
        "pattern": "DIST2_{:02d}.jpeg",
        "frames": range(2, 25), # frame 1 has a hand in it releasing the cars
        "bgPattern": "DIST_BG_{:02d}.jpeg",
        "bgFrames": range(1, 13),
        "bgFolder": os.path.join(DATA_DIR, "Background Distance Race"),
        "crop": (None, None, None, None),
        "kind": "distance",
        "carNames": ("red", "yellow"),
        "note": "Same carpet as races 1 and 3. DIST2_01 shows a hand releasing the "
                "cars, which background subtraction reports as a third region, so "
                "the sequence starts at frame 2.",
    },
    "distance3": {
        "label": "Distance Race 3",
        "folder": os.path.join(DATA_DIR, "Distance Race 3"),
        "pattern": "DIST3_{:02d}.jpeg",
        "frames": range(2, 25), # DIST3_01 is a duplicate of a background frame
        "bgPattern": "DIST_BG_{:02d}.jpeg",
        "bgFrames": range(1, 13),
        "bgFolder": os.path.join(DATA_DIR, "Background Distance Race"),
        "crop": (None, None, None, None),
        "kind": "distance",
        "carNames": ("red", "yellow"),
        "note": "Same carpet as races 1 and 2. DIST3_01 is byte-identical to "
                "DIST_BG_01, so it is a background frame and the race starts at "
                "frame 2. Longest sequence at 23 usable frames.",
    },
}


def loadImage(path, crop):
    # read -> float 0-1 -> rotate upright -> crop, matching the convention the
    # detection scripts already use so both halves see the same pixel coordinates.
    # float32 is deliberate: a 12 MP frame is 283 MB in float64 and only 141 MB
    # here, which roughly halves the cost of every pyramid level.  Anything that
    # needs full precision (the covariance matrices) casts back up locally.
    image = io.imread(path).astype(np.float32) / np.float32(255.0)
    image = np.rot90(image, -1)
    rowStart, rowStop, colStart, colStop = crop
    return np.ascontiguousarray(image[rowStart:rowStop, colStart:colStop, :])


def framePath(key, index):
    # index is 0-based over the usable frames, not the number in the file name
    seq = SEQUENCES[key]
    number = list(seq["frames"])[index]
    return os.path.join(seq["folder"], seq["pattern"].format(number))


def frameCount(key):
    return len(SEQUENCES[key]["frames"])


def loadFrame(key, index):
    # one action frame at full resolution, loaded on demand
    seq = SEQUENCES[key]
    return loadImage(framePath(key, index), seq["crop"])


def loadSequence(key):
    """Every action frame of one race at full resolution, in order.

    A 12 MP frame is 141 MB as float32, so a 23-frame sequence is over 3 GB and
    will not fit in memory on a 8 GB machine alongside everything else.  Prefer
    loadFrame or girish_tracking.loadReduced unless full resolution is genuinely
    needed for every frame at once.
    """
    return [loadFrame(key, index) for index in range(frameCount(key))]


def loadBackground(key):
    # the first background frame is the reference image for subtraction; the rest
    # exist so a per-pixel model could be built later if one reference is not enough
    seq = SEQUENCES[key]
    first = next(iter(seq["bgFrames"]))
    return loadImage(os.path.join(seq["bgFolder"], seq["bgPattern"].format(first)), seq["crop"])


def loadAllBackgrounds(key):
    # all background frames for a sequence, used when checking reference stability
    seq = SEQUENCES[key]
    return [loadImage(os.path.join(seq["bgFolder"], seq["bgPattern"].format(i)), seq["crop"])
            for i in seq["bgFrames"]]


if __name__ == "__main__":
    # sanity listing: confirm every configured file actually exists on disk
    for key, seq in SEQUENCES.items():
        missing = [seq["pattern"].format(i) for i in seq["frames"]
                   if not os.path.exists(os.path.join(seq["folder"], seq["pattern"].format(i)))]
        missingBg = [seq["bgPattern"].format(i) for i in seq["bgFrames"]
                     if not os.path.exists(os.path.join(seq["bgFolder"], seq["bgPattern"].format(i)))]
        print(f"{key:12s} {seq['label']:20s} {len(seq['frames']):3d} frames, "
              f"{len(seq['bgFrames']):2d} bg, missing {len(missing)} / {len(missingBg)}")
        for name in missing + missingBg:
            print("   MISSING", name)
