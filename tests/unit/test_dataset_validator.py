"""Unit tests for Dataset_Validator."""

from pathlib import Path
import tempfile

from src.data.validator import Dataset_Validator


class TestDatasetValidator:
    """Tests for Dataset_Validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = Dataset_Validator()

    def create_test_dataset(self, tmpdir, structure):
        """
        Helper to create test dataset structure.

        Args:
            tmpdir: Temporary directory path
            structure: Dict describing dataset structure
                {
                    'train': {'images': ['img1.jpg', 'img2.jpg'], 'labels': ['img1.txt']},
                    'val': {'images': ['img3.jpg'], 'labels': ['img3.txt']}
                }
        """
        base_path = Path(tmpdir)

        for split, content in structure.items():
            # Create image directory
            img_dir = base_path / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)

            # Create label directory
            label_dir = base_path / "labels" / split
            label_dir.mkdir(parents=True, exist_ok=True)

            # Create image files
            for img_name in content.get("images", []):
                (img_dir / img_name).touch()

            # Create label files
            for label_name in content.get("labels", []):
                label_path = label_dir / label_name
                # Write content if specified
                if "label_content" in content and label_name in content["label_content"]:
                    label_path.write_text(content["label_content"][label_name])
                else:
                    label_path.write_text("0 0.5 0.5 0.3 0.4\n")

        return base_path

    def test_validate_perfect_dataset(self):
        """Test validation of perfect dataset with no issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "train": {"images": ["img1.jpg", "img2.jpg"], "labels": ["img1.txt", "img2.txt"]},
                "val": {"images": ["img3.jpg"], "labels": ["img3.txt"]},
            }
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train", "val": "images/val"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert result.total_images == 3
            assert result.total_labels == 3
            assert len(result.missing_labels) == 0
            assert len(result.empty_labels) == 0
            assert len(result.duplicate_files) == 0
            assert result.is_valid

    def test_validate_missing_labels(self):
        """
        Test validation detects missing label files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "train": {
                    "images": ["img1.jpg", "img2.jpg", "img3.jpg"],
                    "labels": ["img1.txt"],  # Missing img2.txt and img3.txt
                }
            }
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert result.total_images == 3
            assert result.total_labels == 1
            assert len(result.missing_labels) == 2
            assert not result.is_valid

            # Verify specific files are reported
            missing_names = [Path(p).stem for p in result.missing_labels]
            assert "img2" in missing_names
            assert "img3" in missing_names

    def test_validate_empty_labels(self):
        """
        Test validation detects empty annotation files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "train": {
                    "images": ["img1.jpg", "img2.jpg"],
                    "labels": ["img1.txt", "img2.txt"],
                    "label_content": {
                        "img1.txt": "0 0.5 0.5 0.3 0.4\n",
                        "img2.txt": "",  # Empty file
                    },
                }
            }
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert result.total_images == 2
            assert result.total_labels == 2
            assert len(result.empty_labels) == 1
            assert not result.is_valid

            # Verify specific file is reported
            assert "img2.txt" in result.empty_labels[0]

    def test_validate_duplicate_filenames(self):
        """
        Test validation detects duplicate filenames across splits.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "train": {"images": ["img1.jpg", "img2.jpg"], "labels": ["img1.txt", "img2.txt"]},
                "val": {
                    "images": ["img1.jpg", "img3.jpg"],  # img1 is duplicate
                    "labels": ["img1.txt", "img3.txt"],
                },
            }
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train", "val": "images/val"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert len(result.duplicate_files) == 1
            assert not result.is_valid
            assert "img1" in result.duplicate_files[0]
            assert "train" in result.duplicate_files[0]
            assert "val" in result.duplicate_files[0]

    def test_validate_multiple_issues(self):
        """Test validation with multiple types of issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "train": {
                    "images": ["img1.jpg", "img2.jpg"],
                    "labels": ["img2.txt"],  # Missing img1.txt
                    "label_content": {
                        "img2.txt": ""  # Empty file
                    },
                },
                "val": {
                    "images": ["img1.jpg"],  # Duplicate with train
                    "labels": ["img1.txt"],
                },
            }
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train", "val": "images/val"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert len(result.missing_labels) == 1
            assert len(result.empty_labels) == 1
            assert len(result.duplicate_files) == 1
            assert not result.is_valid

    def test_validate_nonexistent_directory(self):
        """Test validation with non-existent image directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            data_config = {
                "train": "images/train"  # Directory doesn't exist
            }

            result = self.validator.validate_dataset(data_config, base_path)

            assert result.total_images == 0
            assert len(result.warnings) > 0
            assert "not found" in result.warnings[0]

    def test_validate_multiple_splits(self):
        """Test validation with train, val, and test splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "train": {"images": ["img1.jpg", "img2.jpg"], "labels": ["img1.txt", "img2.txt"]},
                "val": {"images": ["img3.jpg"], "labels": ["img3.txt"]},
                "test": {"images": ["img4.jpg", "img5.jpg"], "labels": ["img4.txt", "img5.txt"]},
            }
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train", "val": "images/val", "test": "images/test"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert result.total_images == 5
            assert result.total_labels == 5
            assert result.is_valid

    def test_validate_various_image_extensions(self):
        """Test validation with various image file extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "train": {
                    "images": ["img1.jpg", "img2.jpeg", "img3.png", "img4.bmp"],
                    "labels": ["img1.txt", "img2.txt", "img3.txt", "img4.txt"],
                }
            }
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert result.total_images == 4
            assert result.total_labels == 4
            assert result.is_valid

    def test_error_reporting_includes_file_paths(self):
        """
        Test that error messages include specific file paths.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {"train": {"images": ["missing_label.jpg"], "labels": []}}
            base_path = self.create_test_dataset(tmpdir, structure)

            data_config = {"train": "images/train"}

            result = self.validator.validate_dataset(data_config, base_path)

            assert len(result.missing_labels) == 1
            # Verify the file path is in the error message
            assert "missing_label" in result.missing_labels[0]
