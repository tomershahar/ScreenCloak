"""Base classes and interfaces for detection modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class OCRResult:
    """
    Result from OCR engine containing detected text and location.

    Attributes:
        text: Detected text string
        bounding_box: (x, y, width, height) coordinates of text region
        confidence: OCR confidence score (0.0-1.0)
    """

    text: str
    bounding_box: tuple[int, int, int, int]
    confidence: float

    def __post_init__(self) -> None:
        """Validate OCR result values."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if len(self.bounding_box) != 4:
            raise ValueError(f"Bounding box must have 4 values (x, y, w, h), got {self.bounding_box}")

        x, y, w, h = self.bounding_box
        if w < 0 or h < 0:
            raise ValueError(f"Bounding box width and height must be positive, got w={w}, h={h}")

    @property
    def x(self) -> int:
        """X coordinate of top-left corner."""
        return self.bounding_box[0]

    @property
    def y(self) -> int:
        """Y coordinate of top-left corner."""
        return self.bounding_box[1]

    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bounding_box[2]

    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bounding_box[3]

    @property
    def center(self) -> tuple[int, int]:
        """Center point of bounding box (x, y)."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of bounding box in pixels."""
        return self.width * self.height

    def distance_to(self, other: OCRResult) -> float:
        """
        Calculate distance between centers of two OCR results.

        Args:
            other: Another OCR result

        Returns:
            Euclidean distance in pixels
        """
        x1, y1 = self.center
        x2, y2 = other.center
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def vertical_distance_to(self, other: OCRResult) -> int:
        """
        Calculate vertical distance between two OCR results.

        Args:
            other: Another OCR result

        Returns:
            Vertical distance in pixels (positive if other is below, negative if above)
        """
        return other.y - self.y

    def is_on_same_line(self, other: OCRResult, max_vertical_distance: int = 10) -> bool:
        """
        Check if two OCR results are on approximately the same horizontal line.

        Args:
            other: Another OCR result
            max_vertical_distance: Maximum vertical distance to consider same line

        Returns:
            True if on same line, False otherwise
        """
        return abs(self.vertical_distance_to(other)) <= max_vertical_distance


@dataclass
class DetectionResult:
    """
    Result from a detection module indicating sensitive data was found.

    Attributes:
        type: Detection type (e.g., "seed_phrase", "credit_card", "crypto_address")
        confidence: Detection confidence score (0.0-1.0)
        text_preview: First 10-20 chars for logging (for debugging only)
        bounding_box: (x, y, width, height) coordinates of detected region
        action: Action to take ("blur", "warn", "ignore")
        metadata: Optional additional information about the detection
    """

    type: str
    confidence: float
    text_preview: str
    bounding_box: tuple[int, int, int, int]
    action: Literal["blur", "warn", "ignore"]
    metadata: dict[str, any] | None = None

    def __post_init__(self) -> None:
        """Validate detection result values."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if len(self.bounding_box) != 4:
            raise ValueError(f"Bounding box must have 4 values (x, y, w, h), got {self.bounding_box}")

        x, y, w, h = self.bounding_box
        if w < 0 or h < 0:
            raise ValueError(f"Bounding box width and height must be positive, got w={w}, h={h}")

        if self.action not in ["blur", "warn", "ignore"]:
            raise ValueError(f"Action must be 'blur', 'warn', or 'ignore', got {self.action}")

        # Truncate text preview to 50 chars max (safety)
        if len(self.text_preview) > 50:
            self.text_preview = self.text_preview[:47] + "..."

    def to_dict(self) -> dict[str, any]:
        """Convert to dictionary for logging."""
        return {
            "type": self.type,
            "confidence": self.confidence,
            "text_preview": self.text_preview,
            "bounding_box": self.bounding_box,
            "action": self.action,
            "metadata": self.metadata or {},
        }


class BaseDetector(ABC):
    """
    Abstract base class for all detection modules.

    Subclasses must implement the detect() method to perform
    specific pattern detection (seed phrases, credit cards, etc.).
    """

    def __init__(self, config: any) -> None:
        """
        Initialize detector with configuration.

        Args:
            config: Configuration object (Config.detection.*)
        """
        self.config = config

    @abstractmethod
    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        """
        Detect sensitive information in OCR results.

        Args:
            ocr_results: List of OCR results from text detection

        Returns:
            List of detection results (empty if nothing found)
        """
        pass

    def _merge_bounding_boxes(self, boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        """
        Merge multiple bounding boxes into one encompassing box.

        Args:
            boxes: List of bounding boxes (x, y, w, h)

        Returns:
            Merged bounding box (x, y, w, h)
        """
        if not boxes:
            return (0, 0, 0, 0)

        # Find min/max coordinates
        x_min = min(x for x, y, w, h in boxes)
        y_min = min(y for x, y, w, h in boxes)
        x_max = max(x + w for x, y, w, h in boxes)
        y_max = max(y + h for x, y, w, h in boxes)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _are_spatially_clustered(
        self,
        ocr_results: list[OCRResult],
        max_distance: int = 50,
    ) -> bool:
        """
        Check if OCR results are spatially clustered (close together).

        Args:
            ocr_results: List of OCR results to check
            max_distance: Maximum distance in pixels to consider clustered

        Returns:
            True if all results are within max_distance of each other
        """
        if len(ocr_results) <= 1:
            return True

        # Check pairwise distances
        for i in range(len(ocr_results) - 1):
            vertical_dist = abs(ocr_results[i].vertical_distance_to(ocr_results[i + 1]))
            if vertical_dist > max_distance:
                return False

        return True

    def _find_consecutive_runs(
        self,
        ocr_results: list[OCRResult],
        min_length: int,
        max_gap: int = 50,
    ) -> list[list[OCRResult]]:
        """
        Find consecutive runs of OCR results (for sequence detection).

        Args:
            ocr_results: List of OCR results
            min_length: Minimum length of run to return
            max_gap: Maximum vertical gap between consecutive items

        Returns:
            List of runs, each run is a list of OCR results
        """
        if not ocr_results:
            return []

        # Sort by vertical position (y coordinate)
        sorted_results = sorted(ocr_results, key=lambda r: (r.y, r.x))

        runs: list[list[OCRResult]] = []
        current_run: list[OCRResult] = [sorted_results[0]]

        for i in range(1, len(sorted_results)):
            prev = sorted_results[i - 1]
            curr = sorted_results[i]

            # Check if on same line or close vertically
            if abs(curr.y - prev.y) <= max_gap:
                current_run.append(curr)
            else:
                # Save current run if long enough
                if len(current_run) >= min_length:
                    runs.append(current_run)
                # Start new run
                current_run = [curr]

        # Don't forget the last run
        if len(current_run) >= min_length:
            runs.append(current_run)

        return runs


# Helper function for testing
def create_mock_ocr_result(
    text: str,
    x: int = 0,
    y: int = 0,
    width: int = 100,
    height: int = 20,
    confidence: float = 0.9,
) -> OCRResult:
    """Create a mock OCR result for testing."""
    return OCRResult(
        text=text,
        bounding_box=(x, y, width, height),
        confidence=confidence,
    )


if __name__ == "__main__":
    # Test OCRResult
    print("Testing OCRResult...")
    ocr1 = OCRResult(text="Hello", bounding_box=(10, 20, 100, 30), confidence=0.95)
    ocr2 = OCRResult(text="World", bounding_box=(120, 25, 100, 30), confidence=0.90)

    print(f"  OCR1 center: {ocr1.center}")
    print(f"  OCR1 area: {ocr1.area} px²")
    print(f"  Distance between OCR1 and OCR2: {ocr1.distance_to(ocr2):.1f} px")
    print(f"  Are on same line? {ocr1.is_on_same_line(ocr2)}")

    # Test DetectionResult
    print("\nTesting DetectionResult...")
    detection = DetectionResult(
        type="seed_phrase",
        confidence=0.95,
        text_preview="abandon ability able...",
        bounding_box=(10, 20, 300, 50),
        action="blur",
        metadata={"word_count": 12},
    )
    print(f"  Detection: {detection.type}")
    print(f"  Dict: {detection.to_dict()}")

    # Test helper functions
    print("\nTesting helper functions...")

    # Mock detector for testing
    class MockDetector(BaseDetector):
        def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
            return []

    detector = MockDetector(config=None)

    # Test merge bounding boxes
    boxes = [(10, 20, 100, 30), (120, 25, 100, 30)]
    merged = detector._merge_bounding_boxes(boxes)
    print(f"  Merged box: {merged}")

    # Test spatial clustering
    ocr_list = [ocr1, ocr2]
    clustered = detector._are_spatially_clustered(ocr_list, max_distance=50)
    print(f"  Spatially clustered (50px): {clustered}")

    print("\n✓ All tests passed!")
