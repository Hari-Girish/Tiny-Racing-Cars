"""Per-frame race graphics drawn on top of the tracked results.

Everything here reads from the tracks produced by girish_tracking.py.  The
proposal promised, for each frame: a trailing line showing where each car has
been, a line in front of each car, a line indicating the distance between the
leading and trailing car, and a live speed readout.  The distance races also
show the running total distance travelled, and the last frame carries the
winner.

The cars travel up the frame after the rotation, so "in front of the car" means
towards smaller row indices, and a marker drawn across the lane in front of a
car is a horizontal line on screen.
"""

import os

import matplotlib
matplotlib.use("Agg") # render to files, no interactive window needed

import matplotlib.pyplot as plt
import numpy as np

import girish_tracking as gt

# drawing colours, chosen to stay visible against carpet, fleece and white poster
CAR_COLORS = {
    "red": "#ff2d2d",
    "blue": "#2d7dff",
    "yellow": "#ffd21f",
}

GAP_COLOR = "#00e5b0"
TEXT_BOX = dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.65, edgecolor="none")


def carColor(name):
    return CAR_COLORS.get(name, "#ffffff")


def drawFrame(ax, image, tracks, index, speeds, kind, finishLineRow=None, winner=None,
              scale=1.0):
    """Draw one annotated frame onto an existing axis.

    Tracks are stored in full-resolution coordinates.  Figures are drawn on a
    reduced copy of the frame, because a 12 MP frame is 141 MB and rendering
    twenty of them at full size will not fit in memory, so `scale` converts.
    """
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    rows, cols = image.shape[:2]
    if finishLineRow is not None:
        finishLineRow = finishLineRow * scale

    if finishLineRow is not None:
        ax.axhline(finishLineRow, color="#ffffff", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(cols * 0.02, finishLineRow - rows * 0.012, "finish",
                color="white", fontsize=9, bbox=TEXT_BOX)

    for name, track in tracks.items():
        color = carColor(name)
        centers = np.array(track["centers"][:index + 1]) * scale
        centerRow, centerCol = np.array(track["centers"][index]) * scale
        height, width = np.array(track["sizes"][index]) * scale

        # the trail: everywhere this car has been so far
        if len(centers) > 1:
            ax.plot(centers[:, 1], centers[:, 0], color=color, linewidth=2.0, alpha=0.85)
        ax.plot([centerCol], [centerRow], marker="o", markersize=5, color=color)

        # the tracked window
        ax.add_patch(plt.Rectangle((centerCol - width / 2, centerRow - height / 2),
                                   width, height, fill=False, edgecolor=color, linewidth=2))

        # the line in front of the car, spanning its lane
        front = centerRow - height / 2
        ax.plot([centerCol - width * 0.75, centerCol + width * 0.75], [front, front],
                color=color, linewidth=2.5)

        # live speed readout, placed ahead of the car so it never covers it
        if index > 0:
            pixelSpeed = speeds[name]["pixels"][index - 1]
            lengthSpeed = speeds[name]["carLengths"][index - 1]
            label = f"{name}  {pixelSpeed:.0f} px/f  ({lengthSpeed:.2f} car/f)"
        else:
            label = f"{name}  start"
        if kind == "distance":
            label += f"\ntotal {speeds[name]['cumulativeCarLengths'][index]:.2f} car lengths"
        labelRow = np.clip(front - rows * 0.035, rows * 0.03, rows * 0.97)
        ax.text(np.clip(centerCol, cols * 0.02, cols * 0.72), labelRow, label,
                color=color, fontsize=9, va="bottom", bbox=TEXT_BOX)

    # the gap between the leading and the trailing car
    leader, trailer, gap = gt.gapBetween(tracks, index)
    leaderCol = tracks[leader]["centers"][index][1] * scale
    trailerCol = tracks[trailer]["centers"][index][1] * scale
    leaderFront = gt.leadingEdge(tracks[leader], index) * scale
    trailerFront = gt.leadingEdge(tracks[trailer], index) * scale
    midCol = (leaderCol + trailerCol) / 2.0
    ax.plot([leaderCol, midCol, midCol, trailerCol],
            [leaderFront, leaderFront, trailerFront, trailerFront],
            color=GAP_COLOR, linewidth=2.0, linestyle=":")
    ax.text(midCol, (leaderFront + trailerFront) / 2.0,
            f"{leader} leads by {gap:.0f} px", color=GAP_COLOR, fontsize=9,
            ha="center", va="center", bbox=TEXT_BOX)

    if winner is not None:
        # the cars finish at the top of the frame, so the banner goes at the
        # bottom where it cannot cover a car or its speed label
        ax.text(cols / 2, rows * 0.96, f"WINNER: {winner.upper()} CAR",
                color=carColor(winner), fontsize=16, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8,
                          edgecolor=carColor(winner), linewidth=2))


def renderSequence(reduced, tracks, speeds, kind, outputDir, label,
                   finishLineRow=None, winner=None, winnerFrom=None):
    """Write one annotated PNG per frame; returns the list of paths written.

    Draws on the reduced frames the tracker already holds, so nothing has to be
    read back from disk and no full-resolution frame is ever materialised.
    """
    os.makedirs(outputDir, exist_ok=True)
    frames = reduced["frames"]
    scale = 1.0 / reduced["factor"]

    paths = []
    for index in range(len(frames)):
        showWinner = winner if (winnerFrom is not None and index >= winnerFrom) else None
        fig, ax = plt.subplots(figsize=(9, 12))
        drawFrame(ax, frames[index], tracks, index, speeds, kind,
                  finishLineRow=finishLineRow, winner=showWinner, scale=scale)
        ax.set_title(f"{label} - frame {index + 1} of {len(frames)}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(outputDir, f"frame_{index + 1:02d}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def renderSummary(reduced, tracks, speeds, kind, outputPath, label, finishLineRow=None):
    """One figure holding the full trajectory plus the speed and match curves."""
    frames = reduced["frames"]
    scale = 1.0 / reduced["factor"]
    count = len(frames)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6),
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})

    axes[0].imshow(frames[-1])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    if finishLineRow is not None:
        axes[0].axhline(finishLineRow * scale, color="white", linestyle="--", linewidth=1.5)
    for name, track in tracks.items():
        centers = np.array(track["centers"]) * scale
        axes[0].plot(centers[:, 1], centers[:, 0], color=carColor(name), linewidth=2.5,
                     marker="o", markersize=3, label=name)
    axes[0].legend(loc="lower right")
    axes[0].set_title(f"{label}: full tracked path")

    for name, track in tracks.items():
        axes[1].plot(range(2, count + 1), speeds[name]["carLengths"],
                     color=carColor(name), marker="o", markersize=3, label=name)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("car lengths per frame")
    axes[1].set_title("speed, perspective corrected")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for name, track in tracks.items():
        finite = [d if np.isfinite(d) else np.nan for d in track["distances"]]
        axes[2].plot(range(1, count + 1), finite,
                     color=carColor(name), marker="o", markersize=3, label=name)
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("covariance match distance")
    axes[2].set_title("match quality (a spike means a lost track)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(outputPath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return outputPath
