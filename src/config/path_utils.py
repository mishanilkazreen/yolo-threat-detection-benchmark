"""Path resolution utilities for configuration files."""

from pathlib import Path


class PathResolver:
    """Utilities for resolving relative and absolute paths."""

    @staticmethod
    def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
        """
        Resolve a path to an absolute path.

        If the path is already absolute, return it as-is.
        If the path is relative, resolve it relative to base_dir (or cwd if base_dir is None).

        Args:
            path: Path to resolve (can be relative or absolute)
            base_dir: Base directory for resolving relative paths (defaults to current working directory)

        Returns:
            Resolved absolute Path object
        """
        path = Path(path)

        # If already absolute, return as-is
        if path.is_absolute():
            return path

        # Resolve relative to base_dir or cwd
        base_dir = Path.cwd() if base_dir is None else Path(base_dir)

        return (base_dir / path).resolve()

    @staticmethod
    def resolve_config_path(config_path: str | Path, relative_path: str | Path) -> Path:
        """
        Resolve a path relative to a configuration file's directory.

        This is useful for resolving paths specified in configuration files
        relative to the configuration file's location.

        Args:
            config_path: Path to the configuration file
            relative_path: Path specified in the configuration (may be relative or absolute)

        Returns:
            Resolved absolute Path object
        """
        config_path = Path(config_path)
        config_dir = config_path.parent if config_path.is_file() else config_path

        return PathResolver.resolve_path(relative_path, base_dir=config_dir)

    @staticmethod
    def ensure_dir_exists(path: str | Path) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path

        Returns:
            Path object for the directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def normalize_path(path: str | Path) -> str:
        """
        Normalize a path to use forward slashes and resolve . and .. components.

        Args:
            path: Path to normalize

        Returns:
            Normalized path string
        """
        return str(Path(path).as_posix())
