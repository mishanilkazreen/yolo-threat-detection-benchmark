"""
Output Manager for ML Experiment Outputs.

This module provides centralized path construction and directory management
for ML experiment outputs, ensuring consistent organization across training,
evaluation, and explainability results.
"""

from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class PathConfiguration:
    """Configuration for output path construction."""

    base_dir: str = "."
    runs_dir: str = "runs"
    outputs_dir: str = "outputs"
    explanations_dir: str = "explanations"
    default_round_name: str = "train"

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration field is empty.
        """
        if not self.base_dir:
            raise ValueError("base_dir cannot be empty")
        if not self.runs_dir:
            raise ValueError("runs_dir cannot be empty")
        if not self.outputs_dir:
            raise ValueError("outputs_dir cannot be empty")
        if not self.explanations_dir:
            raise ValueError("explanations_dir cannot be empty")


@dataclass
class CheckpointSearchResult:
    """Result of checkpoint path resolution."""

    checkpoint_path: Path
    search_pattern: str  # Pattern that matched (e.g., "standardized", "legacy_nested")
    searched_paths: list[Path]  # All paths checked during search

    def is_legacy(self) -> bool:
        """Check if checkpoint was found using legacy pattern.

        Returns:
            True if checkpoint was found using a legacy pattern, False otherwise.
        """
        return "legacy" in self.search_pattern


@dataclass
class DirectoryCreationResult:
    """Result of directory creation operation."""

    path: Path
    created: bool  # True if directory was created, False if already existed
    error: str | None = None

    def is_success(self) -> bool:
        """Check if operation succeeded.

        Returns:
            True if operation succeeded (no error), False otherwise.
        """
        return self.error is None


class Output_Manager:
    """Centralized manager for experiment output paths.

    This class serves as the single source of truth for path construction,
    implementing a consistent hierarchy based on model name and experiment round:
    - runs/{model_name}/{round_name}/ for training outputs
    - outputs/{model_name}/{round_name}/ for evaluation results
    - explanations/{model_name}/{round_name}/ for explainability outputs

    Attributes:
        base_dir: Base directory for all outputs (default: current directory)
        logger: Logger instance for this class
    """

    def __init__(self, base_dir: str = "."):
        """
        Initialize Output_Manager.

        Args:
            base_dir: Base directory for all outputs (default: current directory)
        """
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Output_Manager initialized with base_dir: {self.base_dir}")

    def sanitize_name(self, name: str) -> str:
        """
        Sanitize model or round name for filesystem compatibility.

        Implements the following transformations:
        1. Convert to lowercase for consistency
        2. Replace spaces with underscores
        3. Remove all characters except alphanumeric, hyphens, and underscores
        4. Collapse consecutive underscores/hyphens to single character
        5. Strip leading/trailing underscores/hyphens

        Args:
            name: Raw name string

        Returns:
            Sanitized name containing only alphanumeric, hyphens, and underscores

        Examples:
            >>> om = Output_Manager()
            >>> om.sanitize_name("YOLOv11n XAI Example")
            'yolov11n_xai_example'
            >>> om.sanitize_name("model-name__v2.1")
            'model-name_v2_1'
            >>> om.sanitize_name("  test_model  ")
            'test_model'
        """
        import re

        # Step 1: Convert to lowercase
        sanitized = name.lower()

        # Step 2: Replace spaces with underscores
        sanitized = sanitized.replace(" ", "_")

        # Step 3: Remove invalid characters (keep only alphanumeric, hyphens, underscores)
        sanitized = re.sub(r"[^a-z0-9_-]", "_", sanitized)

        # Step 4: Collapse consecutive underscores/hyphens to single character
        # Replace multiple underscores with single underscore
        sanitized = re.sub(r"_+", "_", sanitized)
        # Replace multiple hyphens with single hyphen
        sanitized = re.sub(r"-+", "-", sanitized)
        # Replace mixed sequences of underscores and hyphens with single underscore
        sanitized = re.sub(r"[-_]+", lambda m: "_" if "_" in m.group() else "-", sanitized)

        # Step 5: Strip leading/trailing underscores/hyphens
        sanitized = sanitized.strip("_-")

        return sanitized

    def _validate_and_sanitize(self, name: str, label: str) -> str:
        """Validate and sanitize a name for filesystem use.

        Raises:
            ValueError: If name is empty or sanitizes to empty.
        """
        if not name or not name.strip():
            raise ValueError(f"{label} cannot be empty")
        sanitized = self.sanitize_name(name)
        if not sanitized:
            raise ValueError(f"{label} contains no valid characters")
        return sanitized

    def _mkdir(self, path: Path) -> None:
        """Create directory with standardized error handling.

        Raises:
            PermissionError: If the directory cannot be created due to permissions.
            OSError: If directory creation fails for any other reason.
        """
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o755)
            self.logger.debug(f"Created directory: {path}")
        except PermissionError as e:
            error_msg = f"Permission denied creating directory {path}"
            self.logger.error(error_msg)
            raise PermissionError(error_msg) from e
        except OSError as e:
            error_msg = f"Failed to create directory {path}: {e}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from e

    def get_training_output_path(
        self, model_name: str, round_name: str | None = None, create: bool = True
    ) -> Path:
        """Get standardized training output path: runs/{model_name}/{round_name}/

        Args:
            model_name: Model identifier
            round_name: Training round identifier (default: "train")
            create: Whether to create the directory if it doesn't exist

        Returns:
            Path object for runs/{model_name}/{round_name}/

        Raises:
            ValueError: If model_name is empty or invalid
            OSError: If directory creation fails
        """
        sm = self._validate_and_sanitize(model_name, "model_name")
        sr = self._validate_and_sanitize(round_name or "train", "round_name")
        output_path = self.base_dir / "runs" / sm / sr
        if create:
            self._mkdir(output_path)
        return output_path

    def get_evaluation_output_path(
        self, model_name: str, round_name: str | None = None, create: bool = True
    ) -> Path:
        """Get standardized evaluation output path: outputs/{model_name}/{round_name}/

        Args:
            model_name: Model identifier
            round_name: Training round identifier (default: "train")
            create: Whether to create the directory if it doesn't exist

        Returns:
            Path object for outputs/{model_name}/{round_name}/

        Raises:
            ValueError: If model_name is empty or invalid
            OSError: If directory creation fails
        """
        sm = self._validate_and_sanitize(model_name, "model_name")
        sr = self._validate_and_sanitize(round_name or "train", "round_name")
        output_path = self.base_dir / "outputs" / sm / sr
        if create:
            self._mkdir(output_path)
        return output_path

    def get_explainability_output_path(
        self,
        model_name: str,
        round_name: str | None = None,
        method: str | None = None,
        create: bool = True,
    ) -> Path:
        """Get standardized explainability output path.

        Returns explanations/{model_name}/{round_name}/ or
        explanations/{model_name}/{round_name}/{method}/ when method is given.

        Args:
            model_name: Model identifier
            round_name: Training round identifier (default: "train")
            method: XAI method name (e.g., "gradcam", "lrp", "shap")
            create: Whether to create the directory if it doesn't exist

        Raises:
            ValueError: If model_name is empty or invalid
            OSError: If directory creation fails
        """
        sm = self._validate_and_sanitize(model_name, "model_name")
        sr = self._validate_and_sanitize(round_name or "train", "round_name")
        output_path = self.base_dir / "explanations" / sm / sr
        if method is not None:
            output_path = output_path / self._validate_and_sanitize(method, "method")
        if create:
            self._mkdir(output_path)
        return output_path

    def get_yolo_project_parameter(self, model_name: str) -> str:
        """
        Get YOLO project parameter as absolute path to prevent YOLO's default behavior.

        Returns absolute path to "runs" directory to prevent YOLO from adding
        its default "detect" subdirectory or other nesting issues.

        Args:
            model_name: Model identifier (not used, kept for API compatibility)

        Returns:
            Absolute path to runs directory as string

        Examples:
            >>> om = Output_Manager()
            >>> om.get_yolo_project_parameter("yolov11n")
            '/absolute/path/to/runs'
        """
        return str((self.base_dir / "runs").resolve())

    def ensure_unique_round_name(self, model_name: str, round_name: str) -> str:
        """Ensure round name is unique within model directory.

        Returns the sanitized round_name unchanged if the directory does not
        yet exist, otherwise appends a timestamp suffix.

        Raises:
            ValueError: If model_name or round_name is empty or invalid
        """
        from datetime import datetime

        sm = self._validate_and_sanitize(model_name, "model_name")
        sr = self._validate_and_sanitize(round_name, "round_name")

        round_path = self.base_dir / "runs" / sm / sr

        if not round_path.exists():
            self.logger.debug(f"Round directory does not exist: {round_path}. Using: {sr}")
            return sr

        unique = f"{sr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Round directory exists: {round_path}. New name: {unique}")
        return unique

    def resolve_checkpoint_path(
        self, model_name: str, round_name: str | None = None, checkpoint_type: str = "best"
    ) -> Path:
        """Resolve checkpoint path with fallback to legacy patterns.

        Search order:
        1. runs/{model_name}/{round_name}/weights/{type}.pt  (standardized)
        2. runs/detect/runs/detect/{model_name}/{round_name}/weights/{type}.pt  (legacy nested)
        3. runs/detect/{model_name}/weights/{type}.pt  (legacy flat)
        4. runs/detect/{model_name}_baseline/weights/{type}.pt  (legacy baseline)

        Raises:
            ValueError: If model_name is empty/invalid or checkpoint_type is invalid
            FileNotFoundError: If checkpoint not found in any searched location
        """
        if checkpoint_type not in ("best", "last"):
            raise ValueError("checkpoint_type must be 'best' or 'last'")

        sm = self._validate_and_sanitize(model_name, "model_name")
        sr = self._validate_and_sanitize(round_name or "train", "round_name")
        ckpt = f"{checkpoint_type}.pt"

        search_patterns = [
            ("standardized", self.base_dir / "runs" / sm / sr / "weights" / ckpt),
            (
                "legacy_nested",
                self.base_dir / "runs" / "detect" / "runs" / "detect" / sm / sr / "weights" / ckpt,
            ),
            ("legacy_flat", self.base_dir / "runs" / "detect" / sm / "weights" / ckpt),
            (
                "legacy_baseline",
                self.base_dir / "runs" / "detect" / f"{sm}_baseline" / "weights" / ckpt,
            ),
        ]

        searched_paths = []
        for pattern_name, path in search_patterns:
            searched_paths.append(path)
            self.logger.debug(f"Searching for checkpoint at {pattern_name} path: {path}")
            if path.exists():
                if "legacy" in pattern_name:
                    self.logger.warning(
                        f"Checkpoint found at legacy path ({pattern_name}): {path}. "
                        f"Consider migrating to standardized structure."
                    )
                else:
                    self.logger.info(f"Checkpoint found at {pattern_name} path: {path}")
                return path.resolve()

        searched_paths_str = "\n  ".join(str(p) for p in searched_paths)
        error_msg = (
            f"Checkpoint not found for model '{model_name}', round '{round_name}', "
            f"type '{checkpoint_type}'. Searched paths:\n  {searched_paths_str}"
        )
        self.logger.error(error_msg)
        raise FileNotFoundError(error_msg)
