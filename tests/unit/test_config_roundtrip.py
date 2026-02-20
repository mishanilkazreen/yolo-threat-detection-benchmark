"""Property tests for configuration parsing/serialization round-trip."""

from pathlib import Path
import tempfile

from hypothesis import given
from hypothesis import strategies as st

from src.config.parser import (
    Configuration,
    ConfigurationParser,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from src.config.serializer import ConfigurationSerializer


# Strategy for generating valid training configs
@st.composite
def training_config_strategy(draw):
    """Generate valid TrainingConfig objects."""
    runs = draw(st.integers(min_value=1, max_value=10))
    seeds = None
    if draw(st.booleans()):
        seeds = draw(
            st.lists(st.integers(min_value=0, max_value=10000), min_size=runs, max_size=runs)
        )

    return TrainingConfig(
        epochs=draw(st.integers(min_value=1, max_value=500)),
        patience=draw(st.integers(min_value=0, max_value=100)),
        image_size=draw(st.integers(min_value=32, max_value=1280)),
        device=draw(st.sampled_from(["cuda", "cpu", "mps", "auto"])),
        runs=runs,
        seeds=seeds,
    )


# Strategy for generating valid model configs
@st.composite
def model_config_strategy(draw):
    """Generate valid ModelConfig objects."""
    return ModelConfig(
        name=draw(st.sampled_from(["yolov8n", "yolov11n", "yolov12n", "yolo26n"])),
        weights=draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=".-_/"
                ),
            )
        ),
    )


# Strategy for generating valid data configs
@st.composite
def data_config_strategy(draw):
    """Generate valid DataConfig objects."""
    return DataConfig(
        yaml_path=draw(
            st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=".-_/"
                ),
            )
        )
    )


# Strategy for generating valid configurations
@st.composite
def configuration_strategy(draw):
    """Generate valid Configuration objects."""
    return Configuration(
        training=draw(training_config_strategy()),
        model=draw(model_config_strategy()),
        data=draw(data_config_strategy()),
    )


class TestConfigurationRoundTrip:
    """Property-based tests for configuration round-trip conversion.

    Validates: Property 51 - Configuration Round-Trip
    Requirements: 20.4
    """

    @given(config=configuration_strategy())
    def test_configuration_roundtrip(self, config):
        """
        Property 51: Configuration Round-Trip

        For any valid Configuration object, serializing then parsing should
        produce an equivalent object.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Serialize
            ConfigurationSerializer.serialize(config, config_path)

            # Parse
            parsed_config = ConfigurationParser.parse(config_path)

            # Verify equivalence
            assert parsed_config.training.epochs == config.training.epochs
            assert parsed_config.training.patience == config.training.patience
            assert parsed_config.training.image_size == config.training.image_size
            assert parsed_config.training.device == config.training.device
            assert parsed_config.training.runs == config.training.runs
            assert parsed_config.training.seeds == config.training.seeds

            assert parsed_config.model.name == config.model.name
            assert parsed_config.model.weights == config.model.weights

            assert parsed_config.data.yaml_path == config.data.yaml_path

    def test_minimal_config_roundtrip(self):
        """Test round-trip with minimal configuration (no optional fields)."""
        config = Configuration(
            training=TrainingConfig(epochs=100, patience=10, image_size=640, device="cuda"),
            model=ModelConfig(name="yolov8n", weights="yolov8n.pt"),
            data=DataConfig(yaml_path="config/data/dataset.yaml"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Serialize
            ConfigurationSerializer.serialize(config, config_path)

            # Parse
            parsed_config = ConfigurationParser.parse(config_path)

            # Verify equivalence
            assert parsed_config.training.epochs == config.training.epochs
            assert parsed_config.training.runs == 1  # Default
            assert parsed_config.training.seeds is None  # Default

    def test_full_config_roundtrip(self):
        """Test round-trip with all optional fields."""
        config = Configuration(
            training=TrainingConfig(
                epochs=100, patience=10, image_size=640, device="cuda", runs=3, seeds=[42, 123, 456]
            ),
            model=ModelConfig(name="yolov8n", weights="yolov8n.pt"),
            data=DataConfig(yaml_path="config/data/dataset.yaml"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Serialize
            ConfigurationSerializer.serialize(config, config_path)

            # Parse
            parsed_config = ConfigurationParser.parse(config_path)

            # Verify equivalence
            assert parsed_config.training.runs == 3
            assert parsed_config.training.seeds == [42, 123, 456]
