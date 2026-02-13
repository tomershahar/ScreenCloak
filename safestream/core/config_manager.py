"""Configuration management for SafeStream."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class OCRConfig:
    """OCR engine configuration."""

    engine: Literal["auto", "paddleocr", "easyocr", "tesseract"] = "auto"
    gpu_enabled: bool = True
    frame_sample_rate: int = 5  # Process every Nth frame


@dataclass
class SeedPhraseConfig:
    """Seed phrase detection configuration."""

    enabled: bool = True
    min_word_count: int = 12  # Require 12 or 24 word sequences


@dataclass
class CreditCardConfig:
    """Credit card detection configuration."""

    enabled: bool = True


@dataclass
class CryptoAddressConfig:
    """Crypto address detection configuration."""

    enabled: bool = True
    networks: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])


@dataclass
class APIKeysConfig:
    """API keys detection configuration."""

    enabled: bool = False  # Paid tier feature


@dataclass
class PersonalStringsConfig:
    """Personal strings detection configuration."""

    enabled: bool = True
    max_free: int = 3  # Free tier limit
    strings: list[str] = field(default_factory=list)
    fuzzy_threshold: int = 85  # 0-100, lower = more lenient matching


@dataclass
class DetectionConfig:
    """Detection settings."""

    seed_phrases: SeedPhraseConfig = field(default_factory=SeedPhraseConfig)
    credit_cards: CreditCardConfig = field(default_factory=CreditCardConfig)
    crypto_addresses: CryptoAddressConfig = field(default_factory=CryptoAddressConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    personal_strings: PersonalStringsConfig = field(default_factory=PersonalStringsConfig)


@dataclass
class CaptureConfig:
    """Screen capture configuration."""

    monitor: int = 1  # 0 = all monitors, 1 = primary, 2 = secondary, etc.
    roi_enabled: bool = True  # Only OCR changed regions


@dataclass
class OBSConfig:
    """OBS integration configuration."""

    enabled: bool = True
    host: str = "localhost"
    port: int = 4455
    password: str = ""
    privacy_scene: str = "Privacy Mode"  # Scene to switch to on detection
    auto_return: bool = True  # Return to previous scene after N seconds
    return_delay: int = 3  # Seconds


@dataclass
class PrivacyConfig:
    """Privacy and logging configuration."""

    log_detections: bool = True
    log_sanitized: bool = True  # Log "[REDACTED]" instead of actual data
    telemetry_opt_in: bool = False


@dataclass
class PerformanceConfig:
    """Performance settings."""

    max_latency_ms: int = 500  # Warn if OCR takes longer
    memory_limit_mb: int = 2048  # Restart OCR engine if exceeded


@dataclass
class Config:
    """Main configuration class."""

    ocr: OCRConfig = field(default_factory=OCRConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    obs: OBSConfig = field(default_factory=OBSConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


class ConfigManager:
    """Manages configuration loading and validation."""

    @staticmethod
    def load(config_path: str | Path = "config.yaml") -> Config:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file (default: config.yaml)

        Returns:
            Config object with loaded settings

        Raises:
            FileNotFoundError: If config file doesn't exist and no default can be created
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)

        # Create default config if file doesn't exist
        if not config_path.exists():
            print(f"Config file not found at {config_path}, creating default config...")
            ConfigManager._create_default_config(config_path)

        # Load YAML
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

        # Parse into Config object
        if config_dict is None:
            config_dict = {}

        try:
            config = ConfigManager._parse_config(config_dict)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")

        # Validate
        ConfigManager._validate_config(config)

        return config

    @staticmethod
    def _parse_config(config_dict: dict) -> Config:
        """Parse dictionary into Config object."""
        # OCR config
        ocr_dict = config_dict.get("ocr", {})
        ocr_config = OCRConfig(
            engine=ocr_dict.get("engine", "auto"),
            gpu_enabled=ocr_dict.get("gpu_enabled", True),
            frame_sample_rate=ocr_dict.get("frame_sample_rate", 5),
        )

        # Detection config
        detection_dict = config_dict.get("detection", {})

        seed_phrases_dict = detection_dict.get("seed_phrases", {})
        seed_phrases_config = SeedPhraseConfig(
            enabled=seed_phrases_dict.get("enabled", True),
            min_word_count=seed_phrases_dict.get("min_word_count", 12),
        )

        credit_cards_dict = detection_dict.get("credit_cards", {})
        credit_cards_config = CreditCardConfig(enabled=credit_cards_dict.get("enabled", True))

        crypto_addresses_dict = detection_dict.get("crypto_addresses", {})
        crypto_addresses_config = CryptoAddressConfig(
            enabled=crypto_addresses_dict.get("enabled", True),
            networks=crypto_addresses_dict.get("networks", ["BTC", "ETH", "SOL"]),
        )

        api_keys_dict = detection_dict.get("api_keys", {})
        api_keys_config = APIKeysConfig(enabled=api_keys_dict.get("enabled", False))

        personal_strings_dict = detection_dict.get("personal_strings", {})
        personal_strings_config = PersonalStringsConfig(
            enabled=personal_strings_dict.get("enabled", True),
            max_free=personal_strings_dict.get("max_free", 3),
            strings=personal_strings_dict.get("strings") or [],
            fuzzy_threshold=personal_strings_dict.get("fuzzy_threshold", 85),
        )

        detection_config = DetectionConfig(
            seed_phrases=seed_phrases_config,
            credit_cards=credit_cards_config,
            crypto_addresses=crypto_addresses_config,
            api_keys=api_keys_config,
            personal_strings=personal_strings_config,
        )

        # Capture config
        capture_dict = config_dict.get("capture", {})
        capture_config = CaptureConfig(
            monitor=capture_dict.get("monitor", 1),
            roi_enabled=capture_dict.get("roi_enabled", True),
        )

        # OBS config
        obs_dict = config_dict.get("obs", {})
        obs_config = OBSConfig(
            enabled=obs_dict.get("enabled", True),
            host=obs_dict.get("host", "localhost"),
            port=obs_dict.get("port", 4455),
            password=obs_dict.get("password", ""),
            privacy_scene=obs_dict.get("privacy_scene", "Privacy Mode"),
            auto_return=obs_dict.get("auto_return", True),
            return_delay=obs_dict.get("return_delay", 3),
        )

        # Privacy config
        privacy_dict = config_dict.get("privacy", {})
        privacy_config = PrivacyConfig(
            log_detections=privacy_dict.get("log_detections", True),
            log_sanitized=privacy_dict.get("log_sanitized", True),
            telemetry_opt_in=privacy_dict.get("telemetry_opt_in", False),
        )

        # Performance config
        performance_dict = config_dict.get("performance", {})
        performance_config = PerformanceConfig(
            max_latency_ms=performance_dict.get("max_latency_ms", 500),
            memory_limit_mb=performance_dict.get("memory_limit_mb", 2048),
        )

        return Config(
            ocr=ocr_config,
            detection=detection_config,
            capture=capture_config,
            obs=obs_config,
            privacy=privacy_config,
            performance=performance_config,
        )

    @staticmethod
    def _validate_config(config: Config) -> None:
        """Validate configuration values."""
        # Validate OCR engine
        valid_engines = ["auto", "paddleocr", "easyocr", "tesseract"]
        if config.ocr.engine not in valid_engines:
            raise ValueError(f"Invalid OCR engine: {config.ocr.engine}")

        # Validate frame sample rate
        if config.ocr.frame_sample_rate < 1:
            raise ValueError("frame_sample_rate must be >= 1")

        # Validate seed phrase min word count
        if config.detection.seed_phrases.min_word_count not in [12, 24]:
            raise ValueError("seed_phrases.min_word_count must be 12 or 24")

        # Validate personal strings count (free tier limit)
        strings = config.detection.personal_strings.strings or []
        if len(strings) > config.detection.personal_strings.max_free:
            print(
                f"Warning: You have {len(strings)} "
                f"personal strings, but free tier limit is {config.detection.personal_strings.max_free}"
            )

        # Validate fuzzy threshold
        if not 0 <= config.detection.personal_strings.fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be between 0 and 100")

        # Validate monitor
        if config.capture.monitor < 0:
            raise ValueError("monitor must be >= 0")

        # Validate OBS port
        if not 1 <= config.obs.port <= 65535:
            raise ValueError("OBS port must be between 1 and 65535")

        # Validate return delay
        if config.obs.return_delay < 0:
            raise ValueError("return_delay must be >= 0")

    @staticmethod
    def _create_default_config(config_path: Path) -> None:
        """Create default config.yaml file."""
        default_config = """# SafeStream Configuration

# OCR Settings
ocr:
  engine: "auto"  # auto, paddleocr, easyocr, tesseract
  gpu_enabled: true
  frame_sample_rate: 5  # Process every Nth frame

# Detection Settings
detection:
  seed_phrases:
    enabled: true
    min_word_count: 12  # Require 12 or 24 word sequences

  credit_cards:
    enabled: true

  crypto_addresses:
    enabled: true
    networks: ["BTC", "ETH", "SOL"]

  api_keys:
    enabled: false  # Paid tier feature

  personal_strings:
    enabled: true
    max_free: 3  # Free tier limit
    strings:
      # Add your personal info to detect (examples):
      # - "John Doe"
      # - "johndoe@email.com"
      # - "555-123-4567"
    fuzzy_threshold: 85  # 0-100, lower = more lenient matching

# Screen Capture
capture:
  monitor: 1  # 0 = all monitors, 1 = primary, 2 = secondary, etc.
  roi_enabled: true  # Only OCR changed regions

# OBS Integration
obs:
  enabled: true
  host: "localhost"
  port: 4455
  password: ""
  privacy_scene: "Privacy Mode"  # Scene to switch to on detection
  auto_return: true  # Return to previous scene after N seconds
  return_delay: 3

# Privacy & Logging
privacy:
  log_detections: true
  log_sanitized: true  # Log "[REDACTED]" instead of actual data
  telemetry_opt_in: false

# Performance
performance:
  max_latency_ms: 500  # Warn if OCR takes longer
  memory_limit_mb: 2048  # Restart OCR engine if exceeded
"""

        with open(config_path, "w") as f:
            f.write(default_config)

        print(f"Created default config at {config_path}")
