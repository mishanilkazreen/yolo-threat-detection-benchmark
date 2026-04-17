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

    def get_training_output_path(
        self, model_name: str, round_name: str | None = None, create: bool = True
    ) -> Path:
        """
        Get standardized training output path.

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
        # Validate model_name
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        # Apply sanitization
        sanitized_model_name = self.sanitize_name(model_name)
        if not sanitized_model_name:
            raise ValueError("model_name contains no valid characters")

        # Use default round name if not provided
        if round_name is None:
            round_name = "train"

        sanitized_round_name = self.sanitize_name(round_name)
        if not sanitized_round_name:
            raise ValueError("round_name contains no valid characters")

        # Construct path: runs/{model_name}/{round_name}/
        output_path = self.base_dir / "runs" / sanitized_model_name / sanitized_round_name

        # Create directory if requested
        if create:
            try:
                output_path.mkdir(parents=True, exist_ok=True, mode=0o755)
                self.logger.debug(f"Created directory: {output_path}")
            except PermissionError as e:
                error_msg = f"Permission denied creating directory {output_path}"
                self.logger.error(error_msg)
                raise PermissionError(error_msg) from e
            except OSError as e:
                error_msg = f"Failed to create directory {output_path}: {e}"
                self.logger.error(error_msg)
                raise OSError(error_msg) from e

        return output_path

    def get_evaluation_output_path(
        self, model_name: str, round_name: str | None = None, create: bool = True
    ) -> Path:
        """
        Get standardized evaluation output path.

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
        # Validate model_name
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        # Apply sanitization
        sanitized_model_name = self.sanitize_name(model_name)
        if not sanitized_model_name:
            raise ValueError("model_name contains no valid characters")

        # Use default round name if not provided
        if round_name is None:
            round_name = "train"

        sanitized_round_name = self.sanitize_name(round_name)
        if not sanitized_round_name:
            raise ValueError("round_name contains no valid characters")

        # Construct path: outputs/{model_name}/{round_name}/
        output_path = self.base_dir / "outputs" / sanitized_model_name / sanitized_round_name

        # Create directory if requested
        if create:
            try:
                output_path.mkdir(parents=True, exist_ok=True, mode=0o755)
                self.logger.debug(f"Created directory: {output_path}")
            except PermissionError as e:
                error_msg = f"Permission denied creating directory {output_path}"
                self.logger.error(error_msg)
                raise PermissionError(error_msg) from e
            except OSError as e:
                error_msg = f"Failed to create directory {output_path}: {e}"
                self.logger.error(error_msg)
                raise OSError(error_msg) from e

        return output_path

    def get_explainability_output_path(
        self,
        model_name: str,
        round_name: str | None = None,
        method: str | None = None,
        create: bool = True,
    ) -> Path:
        """
        Get standardized explainability output path.

        Args:
            model_name: Model identifier
            round_name: Training round identifier (default: "train")
            method: XAI method name (e.g., "gradcam", "lrp", "shap")
            create: Whether to create the directory if it doesn't exist

        Returns:
            Path object for explanations/{model_name}/{round_name}/{method}/
            If method is None, returns explanations/{model_name}/{round_name}/

        Raises:
            ValueError: If model_name is empty or invalid
            OSError: If directory creation fails
        """
        # Validate model_name
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        # Apply sanitization
        sanitized_model_name = self.sanitize_name(model_name)
        if not sanitized_model_name:
            raise ValueError("model_name contains no valid characters")

        # Use default round name if not provided
        if round_name is None:
            round_name = "train"

        sanitized_round_name = self.sanitize_name(round_name)
        if not sanitized_round_name:
            raise ValueError("round_name contains no valid characters")

        # Construct base path: explanations/{model_name}/{round_name}/
        output_path = self.base_dir / "explanations" / sanitized_model_name / sanitized_round_name

        # Add method subdirectory if provided
        if method is not None:
            sanitized_method = self.sanitize_name(method)
            if not sanitized_method:
                raise ValueError("method contains no valid characters")
            output_path = output_path / sanitized_method

        # Create directory if requested
        if create:
            try:
                output_path.mkdir(parents=True, exist_ok=True, mode=0o755)
                self.logger.debug(f"Created directory: {output_path}")
            except PermissionError as e:
                error_msg = f"Permission denied creating directory {output_path}"
                self.logger.error(error_msg)
                raise PermissionError(error_msg) from e
            except OSError as e:
                error_msg = f"Failed to create directory {output_path}: {e}"
                self.logger.error(error_msg)
                raise OSError(error_msg) from e

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
        """
        Ensure round name is unique within model directory.

        Checks if runs/{model_name}/{round_name}/ exists. If it does,
        appends a timestamp suffix to ensure uniqueness.

        Args:
            model_name: Model identifier
            round_name: Proposed round name

        Returns:
            Unique round name (appends timestamp if collision detected)

        Raises:
            ValueError: If model_name or round_name is empty or invalid

        Examples:
            >>> om = Output_Manager()
            >>> om.ensure_unique_round_name("yolov11n", "round_1")
            'round_1'  # If directory doesn't exist
            >>> om.ensure_unique_round_name("yolov11n", "round_1")
            'round_1_20240115_143022'  # If directory exists
        """
        from datetime import datetime

        # Validate model_name
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        # Validate round_name
        if not round_name or not round_name.strip():
            raise ValueError("round_name cannot be empty")

        # Apply sanitization
        sanitized_model_name = self.sanitize_name(model_name)
        if not sanitized_model_name:
            raise ValueError("model_name contains no valid characters")

        sanitized_round_name = self.sanitize_name(round_name)
        if not sanitized_round_name:
            raise ValueError("round_name contains no valid characters")

        # Check if runs/{model_name}/{round_name}/ exists
        round_path = self.base_dir / "runs" / sanitized_model_name / sanitized_round_name

        if not round_path.exists():
            # Directory doesn't exist, return original round name
            self.logger.debug(
                f"Round directory does not exist: {round_path}. "
                f"Using original round name: {sanitized_round_name}"
            )
            return sanitized_round_name

        # Directory exists, append timestamp suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_round_name = f"{sanitized_round_name}_{timestamp}"

        self.logger.info(
            f"Round directory already exists: {round_path}. "
            f"Appending timestamp suffix. New round name: {unique_round_name}"
        )

        return unique_round_name

    def resolve_checkpoint_path(
        self, model_name: str, round_name: str | None = None, checkpoint_type: str = "best"
    ) -> Path:
        """
        Resolve checkpoint path with fallback to legacy patterns.

        Searches for checkpoints in the following order:
        1. Standardized path: runs/{model_name}/{round_name}/weights/{checkpoint_type}.pt
        2. Legacy nested path: runs/detect/runs/detect/{model_name}/{round_name}/weights/{checkpoint_type}.pt
        3. Legacy flat path: runs/detect/{model_name}/weights/{checkpoint_type}.pt
        4. Legacy baseline path: runs/detect/{model_name}_baseline/weights/{checkpoint_type}.pt

        Args:
            model_name: Model identifier
            round_name: Training round identifier (default: "train")
            checkpoint_type: Type of checkpoint ("best" or "last")

        Returns:
            Absolute path to checkpoint file

        Raises:
            ValueError: If model_name is empty or invalid, or checkpoint_type is invalid
            FileNotFoundError: If checkpoint not found in any searched location

        Examples:
            >>> om = Output_Manager()
            >>> om.resolve_checkpoint_path("yolov11n", "round_1", "best")
            PosixPath('/absolute/path/to/runs/yolov11n/round_1/weights/best.pt')
        """
        # Validate model_name
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        # Apply sanitization
        sanitized_model_name = self.sanitize_name(model_name)
        if not sanitized_model_name:
            raise ValueError("model_name contains no valid characters")

        # Use default round name if not provided
        if round_name is None:
            round_name = "train"

        sanitized_round_name = self.sanitize_name(round_name)
        if not sanitized_round_name:
            raise ValueError("round_name contains no valid characters")

        # Validate checkpoint_type
        if checkpoint_type not in ["best", "last"]:
            raise ValueError("checkpoint_type must be 'best' or 'last'")

        # Define search patterns
        checkpoint_filename = f"{checkpoint_type}.pt"

        # Pattern 1: Standardized path
        standardized_path = (
            self.base_dir
            / "runs"
            / sanitized_model_name
            / sanitized_round_name
            / "weights"
            / checkpoint_filename
        )

        # Pattern 2: Legacy nested path
        legacy_nested_path = (
            self.base_dir
            / "runs"
            / "detect"
            / "runs"
            / "detect"
            / sanitized_model_name
            / sanitized_round_name
            / "weights"
            / checkpoint_filename
        )

        # Pattern 3: Legacy flat path
        legacy_flat_path = (
            self.base_dir
            / "runs"
            / "detect"
            / sanitized_model_name
            / "weights"
            / checkpoint_filename
        )

        # Pattern 4: Legacy baseline path
        legacy_baseline_path = (
            self.base_dir
            / "runs"
            / "detect"
            / f"{sanitized_model_name}_baseline"
            / "weights"
            / checkpoint_filename
        )

        # Search in order
        search_patterns = [
            ("standardized", standardized_path),
            ("legacy_nested", legacy_nested_path),
            ("legacy_flat", legacy_flat_path),
            ("legacy_baseline", legacy_baseline_path),
        ]

        searched_paths = []

        for pattern_name, path in search_patterns:
            searched_paths.append(path)
            self.logger.debug(f"Searching for checkpoint at {pattern_name} path: {path}")

            if path.exists():
                # Log warning for legacy paths
                if "legacy" in pattern_name:
                    self.logger.warning(
                        f"Checkpoint found at legacy path ({pattern_name}): {path}. "
                        f"Consider migrating to standardized structure."
                    )
                else:
                    self.logger.info(f"Checkpoint found at {pattern_name} path: {path}")

                # Return absolute path
                return path.resolve()

        # Checkpoint not found in any location
        searched_paths_str = "\n  ".join(str(p) for p in searched_paths)
        error_msg = (
            f"Checkpoint not found for model '{model_name}', round '{round_name}', "
            f"type '{checkpoint_type}'. Searched paths:\n  {searched_paths_str}"
        )
        self.logger.error(error_msg)
        raise FileNotFoundError(error_msg)
