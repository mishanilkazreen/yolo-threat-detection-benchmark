"""Property-based and integration tests for XAI configuration bugs.

Preservation and property-based tests verifying the fixed behavior
holds across a wide range of inputs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_LRP_RULES = ["epsilon", "gamma", "alpha-beta"]
_ARCH_KEYS = ("yolov11", "yolov12", "yolo26", "yolov8")

# Strategy for valid lrp_rule values
lrp_rule_st = st.sampled_from(_LRP_RULES)

# Strategy for non-empty lists of non-empty strings (gradcam_target_classes)
class_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
)
gradcam_classes_st = st.lists(class_name_st, min_size=1, max_size=10)

# Strategy for target_layer strings (valid layer path format)
layer_str_st = st.from_regex(r"model\.\d+", fullmatch=True)

# Strategy for XAIConfig kwargs (only the fields relevant to the fixed bugs)
xai_config_kwargs_st = st.fixed_dictionaries(
    {
        "lrp_rule": lrp_rule_st,
        "gradcam_target_classes": gradcam_classes_st,
    }
)


def _make_xai_config(**kwargs):
    """Create an XAIConfig with the given kwargs."""
    from src.explainability.xai.config import XAIConfig

    return XAIConfig(**kwargs)


# ---------------------------------------------------------------------------
# PBT: round-trip serialization preserves all XAIConfig fields
# ---------------------------------------------------------------------------


class TestRoundTripSerializationProperty5:
    """PBT: serialize via _xai_to_dict then parse via _parse_xai must produce
    an XAIConfig equal to the original for all generated inputs.

    Focuses on lrp_rule and gradcam_target_classes (the newly fixed fields).
    """

    @given(kwargs=xai_config_kwargs_st)
    @settings(max_examples=50, deadline=None)
    def test_round_trip_preserves_lrp_rule_and_gradcam_target_classes(self, kwargs):
        """For any XAIConfig with non-default lrp_rule or gradcam_target_classes,
        serializing then parsing must produce an equal XAIConfig.
        """
        from src.config.parser import ConfigurationParser
        from src.config.serializer import ConfigurationSerializer

        original = _make_xai_config(**kwargs)
        serialized = ConfigurationSerializer._xai_to_dict(original)
        parsed = ConfigurationParser._parse_xai(serialized, source="<pbt-test>")

        assert parsed.lrp_rule == original.lrp_rule, (
            f"Round-trip lost lrp_rule: original={original.lrp_rule!r}, "
            f"parsed={parsed.lrp_rule!r}, serialized={serialized!r}"
        )
        assert parsed.gradcam_target_classes == original.gradcam_target_classes, (
            f"Round-trip lost gradcam_target_classes: "
            f"original={original.gradcam_target_classes!r}, "
            f"parsed={parsed.gradcam_target_classes!r}"
        )

    @given(lrp_rule=lrp_rule_st)
    @settings(max_examples=20, deadline=None)
    def test_round_trip_preserves_lrp_rule_alone(self, lrp_rule):
        """lrp_rule alone must survive a round-trip for all valid rule values."""
        from src.config.parser import ConfigurationParser
        from src.config.serializer import ConfigurationSerializer

        original = _make_xai_config(lrp_rule=lrp_rule)
        serialized = ConfigurationSerializer._xai_to_dict(original)
        parsed = ConfigurationParser._parse_xai(serialized, source="<pbt-test>")

        assert parsed.lrp_rule == lrp_rule, (
            f"lrp_rule={lrp_rule!r} not preserved after round-trip. Got: {parsed.lrp_rule!r}"
        )

    @given(classes=gradcam_classes_st)
    @settings(max_examples=50, deadline=None)
    def test_round_trip_preserves_gradcam_target_classes_alone(self, classes):
        """gradcam_target_classes alone must survive a round-trip for any list of strings."""
        from src.config.parser import ConfigurationParser
        from src.config.serializer import ConfigurationSerializer

        original = _make_xai_config(gradcam_target_classes=classes)
        serialized = ConfigurationSerializer._xai_to_dict(original)
        parsed = ConfigurationParser._parse_xai(serialized, source="<pbt-test>")

        assert parsed.gradcam_target_classes == classes, (
            f"gradcam_target_classes={classes!r} not preserved after round-trip. "
            f"Got: {parsed.gradcam_target_classes!r}"
        )


# ---------------------------------------------------------------------------
# PBT: _parse_xai always stores YAML gradcam_target_classes value
# ---------------------------------------------------------------------------


class TestParseXAIGradcamTargetClassesProperty6:
    """PBT: _parse_xai({"gradcam_target_classes": classes}) always returns config
    with those exact classes.
    """

    @given(classes=gradcam_classes_st)
    @settings(max_examples=50, deadline=None)
    def test_parse_xai_always_stores_gradcam_target_classes(self, classes):
        """For any list of strings, _parse_xai must store exactly that list."""
        from src.config.parser import ConfigurationParser

        data = {"gradcam_target_classes": classes}
        config = ConfigurationParser._parse_xai(data, source="<pbt-test>")

        assert config.gradcam_target_classes == classes, (
            f"_parse_xai did not store gradcam_target_classes={classes!r}. "
            f"Got: {config.gradcam_target_classes!r}"
        )

    @given(classes=gradcam_classes_st)
    @settings(max_examples=50, deadline=None)
    def test_parse_xai_gradcam_classes_independent_of_other_fields(self, classes):
        """gradcam_target_classes must be stored even when other fields are present."""
        from src.config.parser import ConfigurationParser

        # Include other fields to ensure they don't interfere
        data = {
            "gradcam_target_classes": classes,
            "enabled": True,
            "sample_limit": 16,
        }
        config = ConfigurationParser._parse_xai(data, source="<pbt-test>")

        assert config.gradcam_target_classes == classes, (
            f"gradcam_target_classes={classes!r} not stored when other fields present. "
            f"Got: {config.gradcam_target_classes!r}"
        )


# ---------------------------------------------------------------------------
# PBT: target_layer string stored consistently across all arch keys
# ---------------------------------------------------------------------------


class TestParseXAITargetLayerStringProperty7:
    """PBT: _parse_xai({"target_layer": layer_str}) stores layer_str for ALL arch keys."""

    @given(layer_str=layer_str_st)
    @settings(max_examples=50)
    def test_target_layer_string_stored_for_all_arch_keys(self, layer_str):
        """For any target_layer string, all arch keys must map to that string."""
        from src.config.parser import ConfigurationParser

        data = {"target_layer": layer_str}
        config = ConfigurationParser._parse_xai(data, source="<pbt-test>")

        for arch in _ARCH_KEYS:
            assert arch in config.target_layers, (
                f"Architecture key '{arch}' missing from target_layers after parsing "
                f"target_layer={layer_str!r}. Got: {config.target_layers}"
            )
            assert config.target_layers[arch] == layer_str, (
                f"target_layers['{arch}'] should be {layer_str!r}, "
                f"got {config.target_layers[arch]!r}"
            )

    @given(layer_str=layer_str_st)
    @settings(max_examples=50)
    def test_get_target_layer_returns_stored_value_for_all_archs(self, layer_str):
        """get_target_layer must return the stored value for all architecture variants."""
        from src.config.parser import ConfigurationParser

        data = {"target_layer": layer_str}
        config = ConfigurationParser._parse_xai(data, source="<pbt-test>")

        for arch_variant in ("yolov11n", "yolov12s", "yolo26n", "yolov8n"):
            result = config.get_target_layer(arch_variant)
            assert result == layer_str, (
                f"get_target_layer('{arch_variant}') returned {result!r}, expected {layer_str!r}"
            )

    @given(layer_str=layer_str_st)
    @settings(max_examples=30)
    def test_all_arch_keys_have_same_value(self, layer_str):
        """All arch keys must have the same value when target_layer is a string."""
        from src.config.parser import ConfigurationParser

        data = {"target_layer": layer_str}
        config = ConfigurationParser._parse_xai(data, source="<pbt-test>")

        values = set(config.target_layers.values())
        assert values == {layer_str}, (
            f"Not all arch keys have the same value. "
            f"Expected all to be {layer_str!r}, got: {config.target_layers}"
        )


# ---------------------------------------------------------------------------
# PBT: non-YOLO26 Grad-CAM output unchanged by _target_layer fix
# ---------------------------------------------------------------------------


class TestGradcamPreservationProperty8:
    """PBT: generate_gradcam_attribution with empty target_layer always uses
    model.model.model[-2] (the penultimate layer fallback).

    This is a preservation property: the fix must not change behavior for callers
    that pass an empty target_layer.
    """

    def _make_mock_model(self):
        """Build a minimal mock model."""
        import torch

        mock_model = MagicMock()
        real_module = torch.nn.Linear(4, 4)
        mock_model.model = real_module
        mock_model.names = {0: "knife", 1: "pistol"}
        mock_model.imgsz = 64
        return mock_model

    def _save_test_image(self, tmp_path):
        """Save a tiny RGB image and return its path."""
        from PIL import Image

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)
        return image_path

    @given(
        model_name=st.sampled_from(["yolov11n", "yolov12s", "yolov8n"]),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_empty_target_layer_always_uses_penultimate_layer(self, model_name, tmp_path):
        """For any model config, empty target_layer must use model.model.model[-2].

        get_target_layer_module must NOT be called when target_layer is empty.
        """
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_model()

        penultimate = torch.nn.Linear(4, 4)
        mock_model.model.model = MagicMock()
        mock_model.model.model.__getitem__ = MagicMock(return_value=penultimate)

        captured_layers = []

        def fake_eigencam(model, target_layers, **kwargs):
            captured_layers.extend(target_layers)
            cam_instance = MagicMock()
            cam_instance.return_value = np.zeros((1, 64, 64), dtype=np.float32)
            return cam_instance

        with (
            patch("src.explainability.xai.gradcam.get_target_layer_module") as mock_get_layer,
            patch(
                "src.explainability.xai.gradcam.EigenCAM",
                side_effect=fake_eigencam,
            ),
            patch(
                "src.explainability.xai.gradcam._has_target_class_detections",
                return_value=True,
            ),
            patch(
                "src.explainability.xai.gradcam.show_cam_on_image",
                return_value=np.zeros((64, 64, 3), dtype=np.uint8),
            ),
        ):
            generate_gradcam_attribution(
                model=mock_model,
                image_path=image_path,
                detections={},
                gt_boxes=[],
                target_layer="",  # empty → must use penultimate layer
            )

        mock_get_layer.assert_not_called()
        assert len(captured_layers) == 1
        assert captured_layers[0] is penultimate, (
            f"With empty target_layer and model_name={model_name!r}, "
            "EigenCAM must use model.model.model[-2], not a resolved layer."
        )

        penultimate = torch.nn.Linear(4, 4)
        mock_model.model.model = MagicMock()
        mock_model.model.model.__getitem__ = MagicMock(return_value=penultimate)

        captured_layers = []

        def fake_eigencam(model, target_layers, **kwargs):
            captured_layers.extend(target_layers)
            cam_instance = MagicMock()
            cam_instance.return_value = np.zeros((1, 64, 64), dtype=np.float32)
            return cam_instance

        with (
            patch("src.explainability.xai.gradcam.get_target_layer_module") as mock_get_layer,
            patch(
                "src.explainability.xai.gradcam.EigenCAM",
                side_effect=fake_eigencam,
            ),
            patch(
                "src.explainability.xai.gradcam._has_target_class_detections",
                return_value=True,
            ),
            patch(
                "src.explainability.xai.gradcam.show_cam_on_image",
                return_value=np.zeros((64, 64, 3), dtype=np.uint8),
            ),
        ):
            generate_gradcam_attribution(
                model=mock_model,
                image_path=image_path,
                detections={},
                gt_boxes=[],
                target_layer="",  # empty → must use penultimate layer
            )

        mock_get_layer.assert_not_called()
        assert len(captured_layers) == 1
        assert captured_layers[0] is penultimate, (
            f"With empty target_layer and model_name={model_name!r}, "
            "EigenCAM must use model.model.model[-2], not a resolved layer."
        )

    def test_empty_string_never_calls_get_target_layer_module(self, tmp_path):
        """Empty string target_layer must never call get_target_layer_module.

        This is the core preservation invariant: the fix only activates for
        non-empty target_layer strings.
        """
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_model()
        penultimate = torch.nn.Linear(4, 4)
        mock_model.model.model = MagicMock()
        mock_model.model.model.__getitem__ = MagicMock(return_value=penultimate)

        with (
            patch("src.explainability.xai.gradcam.get_target_layer_module") as mock_get_layer,
            patch(
                "src.explainability.xai.gradcam.EigenCAM",
                return_value=MagicMock(return_value=np.zeros((1, 64, 64), dtype=np.float32)),
            ),
            patch(
                "src.explainability.xai.gradcam._has_target_class_detections",
                return_value=True,
            ),
            patch(
                "src.explainability.xai.gradcam.show_cam_on_image",
                return_value=np.zeros((64, 64, 3), dtype=np.uint8),
            ),
        ):
            generate_gradcam_attribution(
                model=mock_model,
                image_path=image_path,
                detections={},
                gt_boxes=[],
                target_layer="",
            )

        mock_get_layer.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: load all model YAML configs end-to-end without errors
# ---------------------------------------------------------------------------


class TestLoadAllModelYAMLConfigs:
    """Integration: load each YAML file in config/models/ through ConfigurationParser
    and assert no parse or validation errors.
    """

    _MODEL_YAML_DIR = "config/models"

    def _get_yaml_files(self):
        """Return all .yaml files in config/models/."""

        yaml_dir = Path(self._MODEL_YAML_DIR)
        return list(yaml_dir.glob("*.yaml"))

    def test_all_model_yamls_exist(self):
        """At least one model YAML file must exist in config/models/."""
        yaml_files = self._get_yaml_files()
        assert yaml_files, f"No YAML files found in {self._MODEL_YAML_DIR}"

    def test_all_model_yamls_parse_without_error(self):
        """Every model YAML file must parse without ConfigurationParseError."""
        from src.config.parser import ConfigurationParser

        yaml_files = self._get_yaml_files()
        assert yaml_files, f"No YAML files found in {self._MODEL_YAML_DIR}"

        for yaml_path in yaml_files:
            config = ConfigurationParser.parse(yaml_path)
            assert config is not None, f"parse({yaml_path}) returned None"
            assert config.training is not None
            assert config.model is not None
            assert config.data is not None
            assert config.xai is not None

    def test_all_model_yamls_have_valid_xai_config(self):
        """Every model YAML file must produce a valid XAIConfig after parsing."""
        from src.config.parser import ConfigurationParser

        yaml_files = self._get_yaml_files()

        for yaml_path in yaml_files:
            config = ConfigurationParser.parse(yaml_path)
            xai = config.xai

            # XAI config must have valid lrp_rule
            assert xai.lrp_rule in ("epsilon", "gamma", "alpha-beta"), (
                f"{yaml_path}: invalid lrp_rule={xai.lrp_rule!r}"
            )

            # target_layers must be a dict with at least one key
            assert isinstance(xai.target_layers, dict), (
                f"{yaml_path}: target_layers must be a dict, got {type(xai.target_layers)}"
            )
            assert xai.target_layers, f"{yaml_path}: target_layers must not be empty"

            # gradcam_target_classes must be a non-empty list of strings
            assert isinstance(xai.gradcam_target_classes, list), (
                f"{yaml_path}: gradcam_target_classes must be a list"
            )
            assert xai.gradcam_target_classes, (
                f"{yaml_path}: gradcam_target_classes must not be empty"
            )

    def test_model_yamls_with_target_layer_string_parse_correctly(self):
        """Model YAMLs using target_layer (string) must parse to a valid target_layers dict.

        All current model YAMLs use `target_layer: "model.9"` (singular string).
        After the fix, this must be expanded to a dict covering all arch keys.
        """
        from src.config.parser import ConfigurationParser

        yaml_files = self._get_yaml_files()

        for yaml_path in yaml_files:
            config = ConfigurationParser.parse(yaml_path)
            xai = config.xai

            # All arch keys must be present in target_layers
            for arch in _ARCH_KEYS:
                assert arch in xai.target_layers, (
                    f"{yaml_path}: arch key '{arch}' missing from target_layers. "
                    f"Got: {xai.target_layers}"
                )


# ---------------------------------------------------------------------------
# Integration: lrp_rule from YAML flows through to XAIConfig
# ---------------------------------------------------------------------------


class TestLRPRuleFlowsThrough:
    """Integration: lrp_rule from YAML flows through ConfigurationParser to XAIConfig."""

    def _make_full_yaml_dict(self, lrp_rule: str) -> dict:
        """Build a minimal valid configuration dict with the given lrp_rule."""
        return {
            "training": {
                "epochs": 10,
                "patience": 5,
                "image_size": 640,
                "device": "cpu",
            },
            "model": {
                "name": "yolov11n",
                "weights": "yolov11n.pt",
            },
            "data": {
                "yaml_path": "data.yaml",
            },
            "xai": {
                "enabled": True,
                "lrp_rule": lrp_rule,
            },
        }

    def test_lrp_rule_gamma_flows_through(self):
        """lrp_rule='gamma' in YAML must produce XAIConfig.lrp_rule == 'gamma'."""
        from src.config.parser import ConfigurationParser

        data = self._make_full_yaml_dict("gamma")
        config = ConfigurationParser._parse_dict(data, source="<test>")

        assert config.xai.lrp_rule == "gamma", (
            f"lrp_rule='gamma' from YAML not preserved. Got: {config.xai.lrp_rule!r}"
        )

    def test_lrp_rule_alpha_beta_flows_through(self):
        """lrp_rule='alpha-beta' in YAML must produce XAIConfig.lrp_rule == 'alpha-beta'."""
        from src.config.parser import ConfigurationParser

        data = self._make_full_yaml_dict("alpha-beta")
        config = ConfigurationParser._parse_dict(data, source="<test>")

        assert config.xai.lrp_rule == "alpha-beta", (
            f"lrp_rule='alpha-beta' from YAML not preserved. Got: {config.xai.lrp_rule!r}"
        )

    def test_lrp_rule_epsilon_flows_through(self):
        """lrp_rule='epsilon' (default) in YAML must produce XAIConfig.lrp_rule == 'epsilon'."""
        from src.config.parser import ConfigurationParser

        data = self._make_full_yaml_dict("epsilon")
        config = ConfigurationParser._parse_dict(data, source="<test>")

        assert config.xai.lrp_rule == "epsilon", (
            f"lrp_rule='epsilon' from YAML not preserved. Got: {config.xai.lrp_rule!r}"
        )

    def test_all_valid_lrp_rules_flow_through(self):
        """All valid lrp_rule values must flow through the parser correctly."""
        from src.config.parser import ConfigurationParser

        for rule in _LRP_RULES:
            data = self._make_full_yaml_dict(rule)
            config = ConfigurationParser._parse_dict(data, source="<test>")

            assert config.xai.lrp_rule == rule, (
                f"lrp_rule={rule!r} from YAML not preserved. Got: {config.xai.lrp_rule!r}"
            )

    def test_lrp_rule_from_yaml_file_flows_through(self, tmp_path):
        """lrp_rule written to a YAML file must be read back correctly via parse()."""
        import yaml

        from src.config.parser import ConfigurationParser

        yaml_content = {
            "training": {"epochs": 5, "patience": 3, "image_size": 640, "device": "cpu"},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {"enabled": True, "lrp_rule": "gamma"},
        }

        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_content, f)

        config = ConfigurationParser.parse(yaml_path)

        assert config.xai.lrp_rule == "gamma", (
            f"lrp_rule='gamma' from YAML file not preserved. Got: {config.xai.lrp_rule!r}"
        )
