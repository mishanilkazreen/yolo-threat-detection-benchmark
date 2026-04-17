"""
Migration utility for reorganizing legacy experiment outputs.

This module provides functions to migrate existing experiment outputs from
legacy directory structures to the standardized format defined by Output_Manager.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


@dataclass
class LegacyOutput:
    """Represents a legacy output directory structure."""

    model_name: str
    round_name: str
    legacy_path: Path
    output_type: str  # "training", "evaluation", or "explainability"


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    source_path: Path
    destination_path: Path
    files_migrated: int
    success: bool
    error: str | None = None


def scan_legacy_outputs(base_dir: str = ".") -> list[LegacyOutput]:
    """
    Scan for legacy directory structures that need migration.

    Searches for the following legacy patterns:
    - runs/detect/{model_name}/
    - runs/detect/runs/detect/{model_name}/
    - runs/detect/{model_name}_baseline/

    Args:
        base_dir: Base directory to scan (default: current directory)

    Returns:
        List of LegacyOutput objects representing found legacy structures

    Examples:
        >>> legacy_outputs = scan_legacy_outputs()
        >>> for output in legacy_outputs:
        ...     print(f"Found: {output.model_name} at {output.legacy_path}")
    """
    base_path = Path(base_dir)
    legacy_outputs = []

    # Pattern 1: runs/detect/{model_name}/
    detect_dir = base_path / "runs" / "detect"
    if detect_dir.exists() and detect_dir.is_dir():
        logger.info(f"Scanning legacy detect directory: {detect_dir}")

        for item in detect_dir.iterdir():
            if not item.is_dir():
                continue

            # Skip nested "runs" directory (will be handled by Pattern 2)
            if item.name == "runs":
                continue

            # Check if this is a model directory (has weights subdirectory)
            weights_dir = item / "weights"
            if weights_dir.exists() and weights_dir.is_dir():
                # Determine model name and round name
                model_name = item.name
                round_name = "train"  # Default for legacy flat structure

                # Check if this is a baseline model
                if model_name.endswith("_baseline"):
                    model_name = model_name[:-9]  # Remove "_baseline" suffix
                    round_name = f"{model_name}_baseline"

                legacy_outputs.append(
                    LegacyOutput(
                        model_name=model_name,
                        round_name=round_name,
                        legacy_path=item,
                        output_type="training",
                    )
                )
                logger.debug(f"Found legacy training output: {item}")

    # Pattern 2: runs/detect/runs/detect/{model_name}/
    nested_detect_dir = base_path / "runs" / "detect" / "runs" / "detect"
    if nested_detect_dir.exists() and nested_detect_dir.is_dir():
        logger.info(f"Scanning nested legacy detect directory: {nested_detect_dir}")

        for item in nested_detect_dir.iterdir():
            if not item.is_dir():
                continue

            # Check if this is a model directory (has weights subdirectory)
            weights_dir = item / "weights"
            if weights_dir.exists() and weights_dir.is_dir():
                model_name = item.name
                round_name = "train"  # Default for legacy nested structure

                legacy_outputs.append(
                    LegacyOutput(
                        model_name=model_name,
                        round_name=round_name,
                        legacy_path=item,
                        output_type="training",
                    )
                )
                logger.debug(f"Found nested legacy training output: {item}")

    logger.info(f"Scan complete. Found {len(legacy_outputs)} legacy output directories.")
    return legacy_outputs


def migrate_model_outputs(
    legacy_output: LegacyOutput, base_dir: str = ".", dry_run: bool = False
) -> MigrationResult:
    """
    Migrate files from legacy structure to standardized paths.

    Moves all files from the legacy directory to the standardized location
    defined by Output_Manager. Preserves directory structure within the
    model/round directory.

    Args:
        legacy_output: LegacyOutput object describing the legacy structure
        base_dir: Base directory for outputs (default: current directory)
        dry_run: If True, only log what would be done without making changes

    Returns:
        MigrationResult object with migration status and statistics

    Examples:
        >>> legacy = LegacyOutput(
        ...     model_name="yolov11n",
        ...     round_name="train",
        ...     legacy_path=Path("runs/detect/yolov11n"),
        ...     output_type="training"
        ... )
        >>> result = migrate_model_outputs(legacy, dry_run=True)
        >>> print(f"Would migrate {result.files_migrated} files")
    """
    from .output_manager import Output_Manager

    base_path = Path(base_dir)
    output_manager = Output_Manager(base_dir=str(base_path))

    # Determine destination path based on output type
    if legacy_output.output_type == "training":
        destination_path = output_manager.get_training_output_path(
            model_name=legacy_output.model_name,
            round_name=legacy_output.round_name,
            create=not dry_run,
        )
    elif legacy_output.output_type == "evaluation":
        destination_path = output_manager.get_evaluation_output_path(
            model_name=legacy_output.model_name,
            round_name=legacy_output.round_name,
            create=not dry_run,
        )
    elif legacy_output.output_type == "explainability":
        destination_path = output_manager.get_explainability_output_path(
            model_name=legacy_output.model_name,
            round_name=legacy_output.round_name,
            create=not dry_run,
        )
    else:
        error_msg = f"Unknown output type: {legacy_output.output_type}"
        logger.error(error_msg)
        return MigrationResult(
            source_path=legacy_output.legacy_path,
            destination_path=Path(),
            files_migrated=0,
            success=False,
            error=error_msg,
        )

    # Check if destination already exists and has content
    if destination_path.exists() and any(destination_path.iterdir()):
        logger.warning(
            f"Destination path already exists and is not empty: {destination_path}. "
            f"Skipping migration to avoid overwriting existing data."
        )
        return MigrationResult(
            source_path=legacy_output.legacy_path,
            destination_path=destination_path,
            files_migrated=0,
            success=False,
            error="Destination path already exists and is not empty",
        )

    # Count files to migrate
    files_to_migrate = []
    for item in legacy_output.legacy_path.rglob("*"):
        if item.is_file():
            files_to_migrate.append(item)

    if dry_run:
        logger.info(
            f"[DRY RUN] Would migrate {len(files_to_migrate)} files from "
            f"{legacy_output.legacy_path} to {destination_path}"
        )
        for file_path in files_to_migrate:
            relative_path = file_path.relative_to(legacy_output.legacy_path)
            dest_file = destination_path / relative_path
            logger.debug(f"[DRY RUN] Would copy: {file_path} -> {dest_file}")

        return MigrationResult(
            source_path=legacy_output.legacy_path,
            destination_path=destination_path,
            files_migrated=len(files_to_migrate),
            success=True,
            error=None,
        )

    # Perform actual migration
    try:
        logger.info(
            f"Migrating {len(files_to_migrate)} files from "
            f"{legacy_output.legacy_path} to {destination_path}"
        )

        files_migrated = 0
        for file_path in files_to_migrate:
            # Calculate relative path within the legacy directory
            relative_path = file_path.relative_to(legacy_output.legacy_path)
            dest_file = destination_path / relative_path

            # Create parent directories if needed
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file (preserve metadata)
            shutil.copy2(file_path, dest_file)
            files_migrated += 1
            logger.debug(f"Copied: {file_path} -> {dest_file}")

        logger.info(
            f"Migration complete. Migrated {files_migrated} files from "
            f"{legacy_output.legacy_path} to {destination_path}"
        )

        return MigrationResult(
            source_path=legacy_output.legacy_path,
            destination_path=destination_path,
            files_migrated=files_migrated,
            success=True,
            error=None,
        )

    except Exception as e:
        error_msg = f"Migration failed: {e}"
        logger.error(error_msg)
        return MigrationResult(
            source_path=legacy_output.legacy_path,
            destination_path=destination_path,
            files_migrated=0,
            success=False,
            error=error_msg,
        )


def verify_migration(migration_result: MigrationResult) -> bool:
    """
    Verify that all files were preserved after migration.

    Checks that:
    1. All files from source exist in destination
    2. File sizes match between source and destination
    3. No files were lost during migration

    Args:
        migration_result: MigrationResult object from migrate_model_outputs()

    Returns:
        True if migration is verified successfully, False otherwise

    Examples:
        >>> result = migrate_model_outputs(legacy_output)
        >>> if verify_migration(result):
        ...     print("Migration verified successfully")
    """
    if not migration_result.success:
        logger.error("Cannot verify failed migration")
        return False

    source_path = migration_result.source_path
    destination_path = migration_result.destination_path

    # Check that source and destination exist
    if not source_path.exists():
        logger.error(f"Source path does not exist: {source_path}")
        return False

    if not destination_path.exists():
        logger.error(f"Destination path does not exist: {destination_path}")
        return False

    # Collect all files from source
    source_files = {}
    for item in source_path.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(source_path)
            source_files[relative_path] = item.stat().st_size

    # Verify all files exist in destination with matching sizes
    verification_passed = True
    for relative_path, source_size in source_files.items():
        dest_file = destination_path / relative_path

        if not dest_file.exists():
            logger.error(f"File missing in destination: {relative_path}")
            verification_passed = False
            continue

        dest_size = dest_file.stat().st_size
        if dest_size != source_size:
            logger.error(
                f"File size mismatch for {relative_path}: "
                f"source={source_size} bytes, destination={dest_size} bytes"
            )
            verification_passed = False

    if verification_passed:
        logger.info(
            f"Migration verification passed. All {len(source_files)} files preserved correctly."
        )
    else:
        logger.error("Migration verification failed. Some files are missing or corrupted.")

    return verification_passed
