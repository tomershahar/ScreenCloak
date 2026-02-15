"""Frame differencing — skip OCR on unchanged screen regions."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("screencloak.frame_diff")


class FrameDiffer:
    """
    Detects whether a screen frame has changed enough to warrant re-running OCR.

    Why this exists:
        OCR is expensive (~200–500ms per frame). Streamers typically have
        large static areas on screen (desktop wallpaper, static UI chrome,
        idle browser tabs). Running OCR every frame wastes CPU on content
        that hasn't changed.

        FrameDiffer compares the current frame to the previous one. If fewer
        than `min_change_pct` percent of pixels changed, it returns False and
        the main loop skips OCR entirely.

    Algorithm:
        1. Convert both frames to grayscale (colour differences don't add info)
        2. Compute per-pixel absolute difference
        3. Gaussian blur to suppress single-pixel noise (camera grain, screen dither)
        4. Threshold: pixels with diff > `pixel_threshold` count as "changed"
        5. Find contours in the change mask
        6. If total changed area >= `min_change_pct` of total frame area → changed

    Typical savings:
        On a mostly-static stream (coding, browsing), 70–80% of frames are
        skipped. OCR only fires when text actually appears or changes.

    Parameters:
        pixel_threshold (int): Grayscale diff 0-255 above which a pixel counts
            as changed. Default 25 — low enough to catch faint text appearing,
            high enough to ignore subtle display refresh noise.
        min_change_pct (float): Minimum fraction of total pixels that must
            change to trigger OCR. Default 0.02 (2%). A single new word
            on a 1080p screen = ~0.1%, so 2% requires a meaningful new text
            block to appear.
        blur_kernel (int): Gaussian blur kernel size (must be odd). Default 5.
            Larger = more tolerant of noise, may miss very thin text.
    """

    def __init__(
        self,
        pixel_threshold: int = 25,
        min_change_pct: float = 0.02,
        blur_kernel: int = 5,
    ) -> None:
        self.pixel_threshold = pixel_threshold
        self.min_change_pct = min_change_pct
        self.blur_kernel = blur_kernel

        self._prev_gray: np.ndarray | None = None

    def has_changed(self, frame: np.ndarray) -> bool:
        """
        Check if a new frame is meaningfully different from the previous one.

        On the very first call (no previous frame), always returns True so
        OCR runs on the first frame regardless.

        Updates the internal previous-frame reference after each call, so
        successive identical frames are detected correctly.

        Args:
            frame: Current RGB frame (H, W, 3) uint8

        Returns:
            True if enough pixels changed to warrant re-running OCR
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None:
            # First frame — always trigger OCR
            self._prev_gray = gray
            return True

        # Resize prev to match if resolution changed (e.g., window resize)
        if self._prev_gray.shape != gray.shape:
            logger.debug("Frame resolution changed — resetting diff baseline")
            self._prev_gray = gray
            return True

        changed = self._compute_change_fraction(self._prev_gray, gray)
        self._prev_gray = gray

        if changed >= self.min_change_pct:
            logger.debug(f"Frame changed: {changed:.1%} of pixels differ")
            return True

        return False

    def reset(self) -> None:
        """
        Clear the previous frame reference.

        Call this when switching scenes in OBS or resuming after a pause,
        to force OCR on the next frame rather than comparing against a
        stale baseline.
        """
        self._prev_gray = None
        logger.debug("FrameDiffer reset — next frame will trigger OCR")

    def get_changed_regions(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """
        Find bounding boxes of regions that changed between two frames.

        Used when the caller wants to know WHERE on screen changes occurred,
        not just whether they occurred. The main loop currently passes the
        full frame to OCR, but this method is available for future
        region-targeted OCR optimisation.

        Algorithm:
        1. Compute diff mask (same as has_changed)
        2. Find contours in the mask
        3. Filter tiny contours (< min_contour_area pixels)
        4. Return each contour's bounding rect as (x, y, w, h)

        Args:
            prev_frame: Previous RGB frame (H, W, 3)
            curr_frame: Current RGB frame (H, W, 3)

        Returns:
            List of changed region bounding boxes (x, y, w, h).
            Empty list if no significant changes.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        mask = self._build_diff_mask(prev_gray, curr_gray)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_pixels = mask.shape[0] * mask.shape[1]
        min_contour_area = max(100, total_pixels * 0.0001)  # At least 0.01% of frame

        regions: list[tuple[int, int, int, int]] = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))

        return regions

    def merge_bboxes(
        self, bboxes: list[tuple[int, int, int, int]]
    ) -> tuple[int, int, int, int]:
        """
        Merge a list of (x, y, w, h) bounding boxes into one enclosing box.

        Used to combine multiple changed regions into a single crop for OCR,
        reducing the number of OCR calls when several nearby regions change
        simultaneously (e.g., a new line of text appearing).

        Args:
            bboxes: List of (x, y, w, h) bounding boxes

        Returns:
            Single (x, y, w, h) bounding box enclosing all inputs.
            Returns (0, 0, 0, 0) for empty input.
        """
        if not bboxes:
            return (0, 0, 0, 0)

        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[0] + b[2] for b in bboxes)
        y_max = max(b[1] + b[3] for b in bboxes)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_diff_mask(
        self, gray_a: np.ndarray, gray_b: np.ndarray
    ) -> np.ndarray:
        """
        Build a binary mask of pixels that changed between two grayscale frames.

        Args:
            gray_a: Previous grayscale frame
            gray_b: Current grayscale frame

        Returns:
            Binary mask (uint8): 255 where pixels changed, 0 elsewhere
        """
        diff = cv2.absdiff(gray_a, gray_b)
        blurred = cv2.GaussianBlur(diff, (self.blur_kernel, self.blur_kernel), 0)
        _, mask = cv2.threshold(blurred, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _compute_change_fraction(
        self, gray_a: np.ndarray, gray_b: np.ndarray
    ) -> float:
        """
        Compute the fraction of pixels that changed between two frames.

        Args:
            gray_a: Previous grayscale frame
            gray_b: Current grayscale frame

        Returns:
            Float in [0.0, 1.0] — fraction of pixels above change threshold
        """
        mask = self._build_diff_mask(gray_a, gray_b)
        changed_pixels = int(np.count_nonzero(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        return changed_pixels / total_pixels if total_pixels > 0 else 0.0


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------


def test_frame_diff() -> None:
    """Test FrameDiffer with synthetic frames."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    print("Testing Frame Differencing (Task 16)\n" + "=" * 60)

    differ = FrameDiffer(pixel_threshold=25, min_change_pct=0.02)

    # Test 1: First frame always triggers OCR
    print("\nTest 1: First frame always triggers OCR")
    frame1 = np.full((480, 640, 3), 100, dtype=np.uint8)
    if differ.has_changed(frame1):
        print("  ✓ First frame triggered OCR (as expected)")
    else:
        print("  ✗ First frame should always trigger")

    # Test 2: Identical frame is skipped
    print("\nTest 2: Identical frame — OCR skipped")
    frame2 = frame1.copy()
    if not differ.has_changed(frame2):
        print("  ✓ Identical frame correctly skipped")
    else:
        print("  ✗ Identical frame should be skipped")

    # Test 3: Small change (1 pixel) is skipped
    print("\nTest 3: Tiny change (1 pixel) — OCR skipped")
    frame3 = frame2.copy()
    frame3[100, 100] = [255, 255, 255]  # Single white pixel
    if not differ.has_changed(frame3):
        print("  ✓ Sub-threshold change correctly skipped")
    else:
        print("  ✗ 1-pixel change should be below 2% threshold")

    # Test 4: Large change (white block) triggers OCR
    print("\nTest 4: Large change (white block) — OCR triggered")
    frame4 = frame3.copy()
    frame4[50:200, 50:300] = [255, 255, 255]  # Large white rectangle
    if differ.has_changed(frame4):
        print("  ✓ Significant change correctly triggered OCR")
    else:
        print("  ✗ Large change should trigger OCR")

    # Test 5: Resolution change resets baseline
    print("\nTest 5: Resolution change — resets baseline and triggers OCR")
    frame5 = np.full((720, 1280, 3), 100, dtype=np.uint8)  # Different resolution
    if differ.has_changed(frame5):
        print("  ✓ Resolution change triggered OCR and reset baseline")
    else:
        print("  ✗ Resolution change should trigger OCR")

    # Test 6: reset() clears baseline — next identical frame triggers
    print("\nTest 6: reset() — forces OCR on next frame")
    frame6 = frame5.copy()
    differ.has_changed(frame6)  # Consume frame6 as baseline
    differ.reset()
    if differ.has_changed(frame6):  # Same frame, but baseline cleared
        print("  ✓ reset() caused next frame to trigger OCR")
    else:
        print("  ✗ reset() should clear baseline")

    # Test 7: get_changed_regions returns bounding boxes
    print("\nTest 7: get_changed_regions — detects changed area bboxes")
    base = np.full((480, 640, 3), 50, dtype=np.uint8)
    modified = base.copy()
    # Add a large bright rectangle in a known location
    modified[100:200, 200:400] = 255
    regions = differ.get_changed_regions(base, modified)
    if regions:
        x, y, w, h = regions[0]
        print(f"  ✓ Found {len(regions)} changed region(s): first bbox=({x},{y},{w},{h})")
    else:
        print("  ✗ Expected at least one changed region")

    # Test 8: merge_bboxes combines overlapping regions
    print("\nTest 8: merge_bboxes — combines multiple regions")
    boxes = [(10, 10, 50, 30), (70, 10, 40, 30), (10, 50, 100, 20)]
    merged = differ.merge_bboxes(boxes)
    expected_x2 = max(10 + 50, 70 + 40, 10 + 100)  # = 120
    expected_y2 = max(10 + 30, 10 + 30, 50 + 20)   # = 70
    x, y, w, h = merged
    if x == 10 and y == 10 and (x + w) == expected_x2 and (y + h) == expected_y2:
        print(f"  ✓ Merged to ({x},{y},{w},{h}) — enclosing all boxes")
    else:
        print(f"  ✗ Expected (10,10,{expected_x2-10},{expected_y2-10}), got {merged}")

    # Test 9: merge_bboxes handles empty input
    print("\nTest 9: merge_bboxes — empty input")
    if differ.merge_bboxes([]) == (0, 0, 0, 0):
        print("  ✓ Empty list returns (0,0,0,0)")
    else:
        print("  ✗ Expected (0,0,0,0)")

    print("\n" + "=" * 60)
    print("Frame Differencing Tests Complete!")


if __name__ == "__main__":
    test_frame_diff()
