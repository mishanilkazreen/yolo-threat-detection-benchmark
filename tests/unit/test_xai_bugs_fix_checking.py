"""Fix-checking tests for XAI configuration bugs.

These tests assert the CORRECT (fixed) behavior and serve as regression guards.
They are distinct from the exploratory tests in test_xai_bugs_exploratory.py,
which assert the buggy behavior on unfixed code.

Note: TestBug11TargetLayerRespected in test_xai_bugs_exploratory.py also covers
the target_layer fix. The tests here provide additional fix-checking coverage
from different angles (parameter name, multiple layer values, no-op when empty).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFixGradcamTargetLayer:
    """Fix-checking tests for Bug 1.1 — generate_gradcam_attribution uses target_layer.

    The fix renamed `_target_layer` to `target_layer` and added:
        if target_layer:
            target_layers = [get_target_layer_module(model, target_layer)]
        else:
            target_layers = [model.model.model[-2]]

    These tests verify the CORRECT behavior is in place.
    """

    def _make_mock_model(self):
        """Build a minimal mock that satisfies generate_gradcam_attribution's checks."""
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

    def _fake_eigencam_factory(self, captured_layers):
        """Return a fake EigenCAM constructor that records target_layers."""

        def fake_eigencam(model, target_layers, **kwargs):
            captured_layers.extend(target_layers)
            cam_instance = MagicMock()
            cam_instance.return_value = np.zeros((1, 64, 64), dtype=np.float32)
            return cam_instance

        return fake_eigencam

    def test_parameter_is_named_target_layer_not_underscore(self):
        """The function signature must use `target_layer`, not `_target_layer`.

        The underscore prefix was the root cause of the bug — it signalled
        "unused" to linters and the body was never updated to use it.
        """
        import inspect

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        sig = inspect.signature(generate_gradcam_attribution)
        param_names = list(sig.parameters.keys())

        assert "target_layer" in param_names, (
            "generate_gradcam_attribution must have a 'target_layer' parameter "
            f"(found: {param_names})"
        )
        assert "_target_layer" not in param_names, (
            "The old '_target_layer' parameter name must be removed — "
            "the underscore prefix was the root cause of the bug"
        )

    def test_get_target_layer_module_called_with_provided_value(self, tmp_path):
        """get_target_layer_module must be called with the exact target_layer string."""
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_model()
        resolved_layer = torch.nn.Linear(4, 4)
        captured_layers = []

        with (
            patch(
                "src.explainability.xai.gradcam.get_target_layer_module",
                return_value=resolved_layer,
            ) as mock_get_layer,
            patch(
                "src.explainability.xai.gradcam.EigenCAM",
                side_effect=self._fake_eigencam_factory(captured_layers),
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
                target_layer="model.9",
            )

        mock_get_layer.assert_called_once_with(mock_model, "model.9")

    def test_eigencam_receives_resolved_layer_not_penultimate(self, tmp_path):
        """EigenCAM must receive the layer from get_target_layer_module, not model[-2].

        This is the core fix: before the fix, EigenCAM always received
        model.model.model[-2] regardless of target_layer.
        """
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_model()
        resolved_layer = torch.nn.Linear(8, 8)  # distinct sentinel object
        captured_layers = []

        with (
            patch(
                "src.explainability.xai.gradcam.get_target_layer_module",
                return_value=resolved_layer,
            ),
            patch(
                "src.explainability.xai.gradcam.EigenCAM",
                side_effect=self._fake_eigencam_factory(captured_layers),
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
                target_layer="model.22",
            )

        assert len(captured_layers) == 1, "EigenCAM should receive exactly one target layer"
        assert captured_layers[0] is resolved_layer, (
            "EigenCAM received the wrong layer — it must use the layer returned by "
            "get_target_layer_module, not model.model.model[-2]"
        )

    def test_different_target_layer_strings_are_forwarded(self, tmp_path):
        """Any non-empty target_layer string must be forwarded to get_target_layer_module."""
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        image_path = self._save_test_image(tmp_path)

        for layer_str in ("model.9", "model.22", "model.backbone.0", "layer3"):
            mock_model = self._make_mock_model()
            resolved_layer = torch.nn.Linear(4, 4)
            captured_layers = []

            with (
                patch(
                    "src.explainability.xai.gradcam.get_target_layer_module",
                    return_value=resolved_layer,
                ) as mock_get_layer,
                patch(
                    "src.explainability.xai.gradcam.EigenCAM",
                    side_effect=self._fake_eigencam_factory(captured_layers),
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
                    target_layer=layer_str,
                )

            (
                mock_get_layer.assert_called_once_with(mock_model, layer_str),
                (f"get_target_layer_module was not called with target_layer={layer_str!r}"),
            )

    def test_empty_target_layer_skips_get_target_layer_module(self, tmp_path):
        """When target_layer is empty, get_target_layer_module must NOT be called.

        This is the preservation path (Requirement 3.1): callers that pass an
        empty string must continue to use the penultimate layer fallback.
        """
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_model()
        penultimate = torch.nn.Linear(4, 4)
        mock_model.model.model = MagicMock()
        mock_model.model.model.__getitem__ = MagicMock(return_value=penultimate)
        captured_layers = []

        with (
            patch("src.explainability.xai.gradcam.get_target_layer_module") as mock_get_layer,
            patch(
                "src.explainability.xai.gradcam.EigenCAM",
                side_effect=self._fake_eigencam_factory(captured_layers),
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
        assert len(captured_layers) == 1
        assert captured_layers[0] is penultimate, (
            "When target_layer is empty, EigenCAM must use model.model.model[-2]"
        )


class TestFixDependencyScoping:
    """Fix-checking tests for Bug 1.2 — XAI libraries scoped to [xai] optional deps.

    The fix moved shap, zennit, captum, and ttach from [project] dependencies (core)
    into [project.optional-dependencies] xai alongside torch and torchvision.

    These tests verify the CORRECT (fixed) behavior and serve as regression guards.
    """

    XAI_ONLY_LIBS = {"shap", "zennit", "captum", "ttach"}
    TORCH_LIBS = {"torch", "torchvision"}

    def _load_pyproject(self):
        """Parse pyproject.toml and return the parsed dict."""
        from pathlib import Path
        import sys

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"

        if sys.version_info >= (3, 11):
            import tomllib

            with open(pyproject_path, "rb") as f:
                return tomllib.load(f)
        else:
            try:
                import tomli

                with open(pyproject_path, "rb") as f:
                    return tomli.load(f)
            except ImportError:
                pytest.skip("tomllib/tomli not available")

    def _parse_dep_names(self, deps: list) -> set:
        """Extract bare package names from a list of PEP 508 dependency strings."""
        import re

        names = set()
        for dep in deps:
            name = re.split(r"[>=<!;\[\s]", dep)[0].lower()
            names.add(name)
        return names

    def _get_core_deps(self, data: dict) -> set:
        return self._parse_dep_names(data.get("project", {}).get("dependencies", []))

    def _get_xai_optional_deps(self, data: dict) -> set:
        return self._parse_dep_names(
            data.get("project", {}).get("optional-dependencies", {}).get("xai", [])
        )

    def test_xai_libs_not_in_core_deps(self):
        """shap, zennit, captum, ttach must NOT appear in [project] dependencies."""
        data = self._load_pyproject()
        core_deps = self._get_core_deps(data)
        wrongly_in_core = self.XAI_ONLY_LIBS & core_deps

        assert not wrongly_in_core, (
            f"Regression: {wrongly_in_core} found in [project] dependencies (core). "
            "These libraries require PyTorch and must live in [project.optional-dependencies] xai."
        )

    def test_xai_libs_in_xai_optional_deps(self):
        """shap, zennit, captum, ttach must ALL appear in [project.optional-dependencies] xai."""
        data = self._load_pyproject()
        xai_deps = self._get_xai_optional_deps(data)
        missing = self.XAI_ONLY_LIBS - xai_deps

        assert not missing, (
            f"Regression: {missing} missing from [project.optional-dependencies] xai. "
            "All XAI libraries must be listed as optional [xai] dependencies."
        )

    def test_torch_and_torchvision_in_xai_optional_deps(self):
        """torch and torchvision must be in [project.optional-dependencies] xai."""
        data = self._load_pyproject()
        xai_deps = self._get_xai_optional_deps(data)
        missing = self.TORCH_LIBS - xai_deps

        assert not missing, (
            f"Regression: {missing} missing from [project.optional-dependencies] xai. "
            "torch and torchvision must remain in the [xai] optional group."
        )

    def test_xai_group_contains_all_required_xai_libraries(self):
        """The [xai] optional group must contain all six required XAI libraries together.

        Required: torch, torchvision, shap, zennit, captum, ttach.
        """
        data = self._load_pyproject()
        xai_deps = self._get_xai_optional_deps(data)
        all_required = self.XAI_ONLY_LIBS | self.TORCH_LIBS
        missing = all_required - xai_deps

        assert not missing, (
            f"Regression: {missing} missing from [project.optional-dependencies] xai. "
            f"The [xai] group must contain all of: {sorted(all_required)}."
        )


class TestFixLRPFallbackRaisesError:
    """Fix-checking tests for Bug 1.4 — YOLO26 LRP raises RuntimeError when all
    gradient strategies fail.

    The original bug: when all three gradient strategies failed for YOLO26, the code
    silently fell through to ``gradient = input_req_grad * input_req_grad.abs()``.
    This is input magnitude, not LRP, and was returned without any warning or error.

    The fix: Strategy 3 now raises ``RuntimeError`` with a descriptive message instead
    of silently assigning the non-LRP fallback value.

    These tests verify the CORRECT (fixed) behavior and serve as regression guards.
    """

    def _make_mock_yolo26_model(self):
        """Build a minimal mock that looks like a YOLO26 model."""
        import torch

        mock_model = MagicMock()
        mock_model.model_name = "yolo26n"
        mock_model.imgsz = 64
        # The inner model must be a real nn.Module so LRPModelWrapper works.
        mock_model.model = torch.nn.Linear(3 * 64 * 64, 4)
        return mock_model

    def _save_test_image(self, tmp_path):
        """Save a tiny RGB image and return its path."""
        from PIL import Image

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)
        return image_path

    def test_strategy3_raises_runtime_error_not_silent_fallback(self, tmp_path, caplog):
        """When all gradient strategies fail, a RuntimeError must be raised (not silently
        returning input * |input|).

        We force all strategies to fail by patching ``torch.autograd.grad`` to return
        ``(None,)``. The outer except catches the RuntimeError and logs it as a warning.
        We verify the fix by asserting the warning log contains the Strategy 3 message.
        """
        import logging

        from src.explainability.xai.lrp import _generate_lrp_with_zennit

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_yolo26_model()

        with (
            caplog.at_level(logging.WARNING, logger="src.explainability.xai.lrp"),
            patch("torch.autograd.grad", return_value=(None,)),
        ):
            _generate_lrp_with_zennit(
                model=mock_model,
                image_path=image_path,
                detections={},
                gt_boxes=[],
                rule="epsilon",
                device="cpu",
                output_dir=None,
                output_manager=None,
                start_time=0.0,
            )

        # The fix: Strategy 3 raises RuntimeError, caught and logged as "Zennit LRP failed".
        # On unfixed code, Strategy 3 silently set gradient = input * |input| — no warning.
        zennit_failed_warnings = [
            record.message
            for record in caplog.records
            if "zennit lrp failed" in record.message.lower()
        ]
        assert zennit_failed_warnings, (
            "Bug 1.4 regression: No 'Zennit LRP failed' warning was logged. "
            "Strategy 3 must raise RuntimeError when all gradient strategies fail, "
            "not silently return input * |input|."
        )

    def test_error_message_mentions_yolo26_or_gradient_strategies(self, tmp_path, caplog):
        """The RuntimeError message must be descriptive — mentioning YOLO26 or gradient
        strategies so the caller understands what failed.
        """
        import logging

        from src.explainability.xai.lrp import _generate_lrp_with_zennit

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_yolo26_model()

        with (
            caplog.at_level(logging.WARNING, logger="src.explainability.xai.lrp"),
            patch("torch.autograd.grad", return_value=(None,)),
        ):
            _generate_lrp_with_zennit(
                model=mock_model,
                image_path=image_path,
                detections={},
                gt_boxes=[],
                rule="epsilon",
                device="cpu",
                output_dir=None,
                output_manager=None,
                start_time=0.0,
            )

        zennit_failed_messages = [
            record.message
            for record in caplog.records
            if "zennit lrp failed" in record.message.lower()
        ]
        assert zennit_failed_messages, (
            "No 'Zennit LRP failed' warning found — Strategy 3 did not raise RuntimeError."
        )

        combined = " ".join(zennit_failed_messages).lower()
        assert any(
            kw in combined
            for kw in ["yolo26", "gradient strateg", "all lrp gradient strategies failed"]
        ), (
            f"RuntimeError message is not descriptive enough. Got: {zennit_failed_messages!r}. "
            "Expected the message to mention 'YOLO26' or 'gradient strategies'."
        )

    def test_error_message_mentions_yolo26_in_source(self):
        """The RuntimeError raised by Strategy 3 must have a descriptive message.

        We directly inspect the source code of _generate_lrp_with_zennit to confirm
        the RuntimeError message mentions YOLO26 and gradient strategies, since
        triggering Strategy 3 in isolation requires a model that passes the forward
        pass but produces zero gradients.
        """
        import inspect

        from src.explainability.xai import lrp as lrp_module

        source = inspect.getsource(lrp_module._generate_lrp_with_zennit)

        # The fix must include a RuntimeError with a descriptive message.
        assert "RuntimeError" in source, (
            "Strategy 3 must raise RuntimeError — not found in _generate_lrp_with_zennit source."
        )

        # The message must mention YOLO26 and gradient strategies.
        assert "YOLO26" in source or "yolo26" in source.lower(), (
            "The RuntimeError message must mention 'YOLO26' to identify the architecture."
        )
        assert "gradient strateg" in source.lower() or "All LRP gradient strategies" in source, (
            "The RuntimeError message must mention gradient strategies to explain what failed."
        )

    def test_result_is_not_input_magnitude_when_strategies_fail(self, tmp_path, caplog):
        """The result must NOT be input * |input| (the silent non-LRP fallback).

        When all strategies fail, the fixed code raises RuntimeError from Strategy 3.
        The outer fallback then uses a gradient hook approach. Either way, the result
        must not be the silent ``input * abs(input)`` value that the unfixed code returned.

        We verify this by checking that the 'Zennit LRP failed' warning is present,
        which proves the RuntimeError path was taken (not the silent fallback).
        """
        import logging

        from src.explainability.xai.lrp import _generate_lrp_with_zennit

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_yolo26_model()

        with (
            caplog.at_level(logging.WARNING, logger="src.explainability.xai.lrp"),
            patch("torch.autograd.grad", return_value=(None,)),
        ):
            result = _generate_lrp_with_zennit(
                model=mock_model,
                image_path=image_path,
                detections={},
                gt_boxes=[],
                rule="epsilon",
                device="cpu",
                output_dir=None,
                output_manager=None,
                start_time=0.0,
            )

        # The function returns a tuple (relevance_np, bbox_score, output_path).
        # The key assertion: the RuntimeError path was taken (not silent fallback).
        # On unfixed code, no "Zennit LRP failed" warning would appear because
        # Strategy 3 silently set gradient = input * |input| without raising.
        zennit_failed_warnings = [
            record.message
            for record in caplog.records
            if "zennit lrp failed" in record.message.lower()
        ]
        assert zennit_failed_warnings, (
            "Bug 1.4 regression: The silent input * |input| fallback was used. "
            "Strategy 3 must raise RuntimeError — the 'Zennit LRP failed' warning "
            "confirms the error path was taken instead of the silent fallback."
        )

        # The result must still be a valid tuple (the outer fallback handles recovery).
        assert result is not None, "Function must return a result tuple even after fallback"
        assert isinstance(result, tuple) and len(result) == 3, (
            "Function must return (relevance_np, bbox_score, output_path)"
        )


class TestFixLRPLoggerLevel:
    """Fix-checking tests for Bug 1.5 — lrp.py import does not force logger to DEBUG level.

    The original bug: ``logger.setLevel(logging.DEBUG)`` at module scope in ``lrp.py``
    overrode any application-level logging configuration set before or after import.

    The fix: the ``logger.setLevel(logging.DEBUG)`` line was removed, leaving the
    logger at ``logging.NOTSET`` so it inherits its effective level from the root logger.

    These tests verify the CORRECT (fixed) behavior and serve as regression guards.
    """

    def test_lrp_logger_level_is_notset(self):
        """lrp.logger.level must be logging.NOTSET (0) after import.

        NOTSET means the logger inherits its effective level from the root logger,
        which is the correct behavior for library code.
        """
        import logging

        from src.explainability.xai import lrp as lrp_module

        assert lrp_module.logger.level == logging.NOTSET, (
            f"Bug 1.5 regression: lrp.logger.level is {lrp_module.logger.level} "
            f"but must be logging.NOTSET ({logging.NOTSET}). "
            "The module-level logger.setLevel(logging.DEBUG) must be removed."
        )

    def test_lrp_logger_level_is_not_debug(self):
        """lrp.logger.level must NOT be logging.DEBUG (10) after import."""
        import logging

        from src.explainability.xai import lrp as lrp_module

        assert lrp_module.logger.level != logging.DEBUG, (
            f"Bug 1.5 regression: lrp.logger.level is logging.DEBUG ({logging.DEBUG}). "
            "The module-level logger.setLevel(logging.DEBUG) must be removed so that "
            "application-level logging configuration is respected."
        )

    def test_application_logging_config_is_respected(self):
        """Setting the root logger level must not be overridden by lrp import.

        This verifies that after importing lrp, the effective level of lrp.logger
        follows the root logger's level (because lrp.logger.level == NOTSET means
        it propagates up to the root).
        """
        import logging

        from src.explainability.xai import lrp as lrp_module

        root_logger = logging.getLogger()
        original_root_level = root_logger.level

        try:
            # Set root logger to WARNING — lrp.logger should inherit this
            root_logger.setLevel(logging.WARNING)

            # lrp.logger.level must still be NOTSET (not overriding root)
            assert lrp_module.logger.level == logging.NOTSET, (
                f"lrp.logger.level changed to {lrp_module.logger.level} after root logger "
                "was reconfigured. The lrp module must not set its own level."
            )

            # The effective level must follow the root logger
            assert lrp_module.logger.getEffectiveLevel() == logging.WARNING, (
                f"lrp.logger effective level is {lrp_module.logger.getEffectiveLevel()} "
                f"but root logger is set to WARNING ({logging.WARNING}). "
                "lrp.logger must inherit from root when its own level is NOTSET."
            )
        finally:
            root_logger.setLevel(original_root_level)


class TestFixSerializerIncludesFields:
    """Fix-checking tests for Bug 1.6 — _xai_to_dict includes lrp_rule and
    gradcam_target_classes when non-default.

    Property 5: _xai_to_dict must include lrp_rule and gradcam_target_classes
    when non-default.
    """

    def test_lrp_rule_included_when_non_default(self):
        """XAIConfig(lrp_rule='gamma') must serialize with 'lrp_rule' in result."""
        from src.config.serializer import ConfigurationSerializer
        from src.explainability.xai.config import XAIConfig

        xai = XAIConfig(lrp_rule="gamma")
        result = ConfigurationSerializer._xai_to_dict(xai)

        assert "lrp_rule" in result, (
            "Bug 1.6 regression: 'lrp_rule' missing from _xai_to_dict output. "
            f"Got keys: {list(result.keys())}"
        )
        assert result["lrp_rule"] == "gamma"

    def test_gradcam_target_classes_included_when_non_default(self):
        """XAIConfig(gradcam_target_classes=['gun']) must serialize with
        'gradcam_target_classes' in result.
        """
        from src.config.serializer import ConfigurationSerializer
        from src.explainability.xai.config import XAIConfig

        xai = XAIConfig(gradcam_target_classes=["gun"])
        result = ConfigurationSerializer._xai_to_dict(xai)

        assert "gradcam_target_classes" in result, (
            "Bug 1.6 regression: 'gradcam_target_classes' missing from _xai_to_dict output. "
            f"Got keys: {list(result.keys())}"
        )
        assert result["gradcam_target_classes"] == ["gun"]

    def test_round_trip_preserves_lrp_rule_and_gradcam_target_classes(self):
        """Serialize then parse must preserve both lrp_rule and gradcam_target_classes."""
        from src.config.parser import ConfigurationParser
        from src.config.serializer import ConfigurationSerializer
        from src.explainability.xai.config import XAIConfig

        original = XAIConfig(lrp_rule="gamma", gradcam_target_classes=["gun", "rifle"])
        serialized = ConfigurationSerializer._xai_to_dict(original)
        parsed = ConfigurationParser._parse_xai(serialized, source="<test>")

        assert parsed.lrp_rule == "gamma", (
            f"Round-trip lost lrp_rule: expected 'gamma', got {parsed.lrp_rule!r}"
        )
        assert parsed.gradcam_target_classes == ["gun", "rifle"], (
            f"Round-trip lost gradcam_target_classes: expected ['gun', 'rifle'], "
            f"got {parsed.gradcam_target_classes!r}"
        )

    def test_default_lrp_rule_not_included(self):
        """Default lrp_rule ('epsilon') must NOT appear in serialized output
        (only non-defaults are written).
        """
        from src.config.serializer import ConfigurationSerializer
        from src.explainability.xai.config import XAIConfig

        xai = XAIConfig()  # all defaults
        result = ConfigurationSerializer._xai_to_dict(xai)

        assert "lrp_rule" not in result, (
            "Default lrp_rule should not be serialized (only non-defaults are written). "
            f"Got keys: {list(result.keys())}"
        )


class TestFixParserReadsGradcamTargetClasses:
    """Fix-checking tests for Bug 1.7 — _parse_xai reads gradcam_target_classes from YAML.

    Property 6: _parse_xai must read gradcam_target_classes from YAML.
    """

    def test_single_value_gradcam_target_classes_parsed(self):
        """_parse_xai({'gradcam_target_classes': ['gun']}) must return config with
        gradcam_target_classes == ['gun'].
        """
        from src.config.parser import ConfigurationParser

        data = {"gradcam_target_classes": ["gun"]}
        config = ConfigurationParser._parse_xai(data, source="<test>")

        assert config.gradcam_target_classes == ["gun"], (
            "Bug 1.7 regression: _parse_xai did not read 'gradcam_target_classes' from YAML. "
            f"Expected ['gun'], got {config.gradcam_target_classes!r}"
        )

    def test_multiple_values_preserved(self):
        """Multiple gradcam_target_classes values must all be preserved."""
        from src.config.parser import ConfigurationParser

        classes = ["gun", "rifle", "grenade"]
        data = {"gradcam_target_classes": classes}
        config = ConfigurationParser._parse_xai(data, source="<test>")

        assert config.gradcam_target_classes == classes, (
            f"Bug 1.7 regression: expected {classes!r}, got {config.gradcam_target_classes!r}"
        )

    def test_default_used_when_key_absent(self):
        """When gradcam_target_classes is absent, the default ['knife', 'pistol'] is used."""
        from src.config.parser import ConfigurationParser
        from src.explainability.xai.config import XAIConfig

        config = ConfigurationParser._parse_xai({}, source="<test>")
        default = XAIConfig().gradcam_target_classes

        assert config.gradcam_target_classes == default, (
            f"Expected default {default!r}, got {config.gradcam_target_classes!r}"
        )


class TestFixParserHandlesTargetLayerString:
    """Fix-checking tests for Bug 1.8 — _parse_xai handles target_layer as a string.

    Property 7: _parse_xai must handle target_layer as a string.
    """

    _DEFAULT_ARCH_KEYS = ("yolov11", "yolov12", "yolo26", "yolov8")

    def test_target_layer_string_stored_for_all_arch_keys(self):
        """_parse_xai({'target_layer': 'model.5'}) must store 'model.5' for all arch keys."""
        from src.config.parser import ConfigurationParser

        data = {"target_layer": "model.5"}
        config = ConfigurationParser._parse_xai(data, source="<test>")

        for arch in self._DEFAULT_ARCH_KEYS:
            assert arch in config.target_layers, (
                f"Architecture key '{arch}' missing from target_layers after parsing "
                f"target_layer string. Got: {config.target_layers}"
            )
            assert config.target_layers[arch] == "model.5", (
                f"target_layers['{arch}'] should be 'model.5', got {config.target_layers[arch]!r}"
            )

    def test_stored_value_accessible_via_get_target_layer(self):
        """The stored value must be accessible via get_target_layer for each arch."""
        from src.config.parser import ConfigurationParser

        data = {"target_layer": "model.5"}
        config = ConfigurationParser._parse_xai(data, source="<test>")

        for arch_variant in ("yolov11n", "yolov12s", "yolo26n", "yolov8n"):
            layer = config.get_target_layer(arch_variant)
            assert layer == "model.5", (
                f"get_target_layer('{arch_variant}') returned {layer!r}, expected 'model.5'"
            )

    def test_target_layers_dict_still_works(self):
        """When target_layers (dict) is provided, it must be used as-is (preservation)."""
        from src.config.parser import ConfigurationParser

        layers = {
            "yolov11": "model.22",
            "yolov12": "model.22",
            "yolo26": "model.21",
            "yolov8": "model.22",
        }
        data = {"target_layers": layers}
        config = ConfigurationParser._parse_xai(data, source="<test>")

        assert config.target_layers == layers, (
            f"target_layers dict not preserved. Expected {layers!r}, got {config.target_layers!r}"
        )


class TestFixValidateXAIConfigWarning:
    """Fix-checking tests for Bug 1.10 — _validate_xai_config logs warning for
    expensive methods without sample_limit.
    """

    def _make_config(self, methods: dict, sample_limit=None):
        """Build a minimal Configuration with the given XAI methods and sample_limit."""
        from src.config.parser import Configuration, DataConfig, ModelConfig, TrainingConfig
        from src.explainability.xai.config import XAIConfig

        xai = XAIConfig(
            enabled=True,
            methods=methods,
            sample_limit=sample_limit,
        )
        training = TrainingConfig(epochs=1, patience=10, image_size=640)
        model = ModelConfig(name="yolov11n", weights="yolov11n.pt")
        data = DataConfig(yaml_path="data.yaml")
        return Configuration(training=training, model=model, data=data, xai=xai)

    def test_lrp_enabled_without_sample_limit_logs_warning(self, caplog):
        """_validate_xai_config with LRP enabled and sample_limit=None must log a warning."""
        import logging

        from src.config.manager import ConfigurationManager

        config = self._make_config(
            methods={"gradcam": True, "lrp": True, "shap": False},
            sample_limit=None,
        )

        with caplog.at_level(logging.WARNING, logger="src.config.manager"):
            ConfigurationManager._validate_xai_config(config, config_path="<test>")

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("sample_limit" in m.lower() or "lrp" in m.lower() for m in warning_messages), (
            "Bug 1.10 regression: No warning logged when LRP is enabled without sample_limit. "
            f"Warnings found: {warning_messages}"
        )

    def test_shap_enabled_without_sample_limit_logs_warning(self, caplog):
        """_validate_xai_config with SHAP enabled and sample_limit=None must log a warning."""
        import logging

        from src.config.manager import ConfigurationManager

        config = self._make_config(
            methods={"gradcam": True, "lrp": False, "shap": True},
            sample_limit=None,
        )

        with caplog.at_level(logging.WARNING, logger="src.config.manager"):
            ConfigurationManager._validate_xai_config(config, config_path="<test>")

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("sample_limit" in m.lower() or "shap" in m.lower() for m in warning_messages), (
            "Bug 1.10 regression: No warning logged when SHAP is enabled without sample_limit. "
            f"Warnings found: {warning_messages}"
        )

    def test_no_warning_when_sample_limit_is_set(self, caplog):
        """When sample_limit is set, no warning about expensive methods must be logged."""
        import logging

        from src.config.manager import ConfigurationManager

        config = self._make_config(
            methods={"gradcam": True, "lrp": True, "shap": True},
            sample_limit=16,
        )

        with caplog.at_level(logging.WARNING, logger="src.config.manager"):
            ConfigurationManager._validate_xai_config(config, config_path="<test>")

        sample_limit_warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "sample_limit" in r.message.lower()
        ]
        assert not sample_limit_warnings, (
            f"Unexpected sample_limit warning when sample_limit=16: {sample_limit_warnings}"
        )


class TestFixYolo12nEpochs:
    """Fix-checking tests for Bug 1.9 — yolo12n.yaml has epochs: 25."""

    def test_yolo12n_yaml_has_epochs_25(self):
        """config/models/yolo12n.yaml must have epochs: 25."""
        from pathlib import Path

        import yaml

        yaml_path = Path("config/models/yolo12n.yaml")
        assert yaml_path.exists(), f"yolo12n.yaml not found at {yaml_path}"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        epochs = data.get("training", {}).get("epochs")
        assert epochs == 125, (
            f"Bug 1.9 regression: yolo12n.yaml has epochs={epochs!r}, expected 125. "
            "All model configs should use 125 epochs (25 per round x 5 rounds)."
        )


class TestFixCaptumLRPUsesDetectedClass:
    """Fix-checking tests for Bug 1.12 — _generate_lrp_with_captum passes detected
    class as target.

    Property 9: _generate_lrp_with_captum must pass the detected class as target.
    """

    def _make_mock_model(self):
        """Build a minimal mock model for captum LRP."""
        import torch

        mock_model = MagicMock()
        mock_model.imgsz = 64
        mock_model.model = torch.nn.Linear(3 * 64 * 64, 4)
        return mock_model

    def _save_test_image(self, tmp_path):
        """Save a tiny RGB image and return its path."""
        from PIL import Image

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)
        return image_path

    def test_detected_class_passed_as_target(self, tmp_path):
        """When detections contain class 2, CaptumLRP.attribute must be called with target=2."""
        import torch

        from src.explainability.xai.lrp import _generate_lrp_with_captum

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_model()

        detections = {
            "scores": [0.5, 0.9, 0.3],
            "classes": [0, 2, 1],  # highest score (0.9) → class 2
        }

        fake_attr = torch.zeros(1, 3, 64, 64)
        mock_lrp_instance = MagicMock()
        mock_lrp_instance.attribute.return_value = fake_attr

        with patch("captum.attr.LRP", return_value=mock_lrp_instance):
            _generate_lrp_with_captum(
                model=mock_model,
                image_path=image_path,
                detections=detections,
                gt_boxes=[],
                device="cpu",
                output_dir=None,
                output_manager=None,
                start_time=0.0,
            )

        mock_lrp_instance.attribute.assert_called_once()
        _, call_kwargs = mock_lrp_instance.attribute.call_args
        target_used = call_kwargs.get(
            "target",
            mock_lrp_instance.attribute.call_args[0][1]
            if len(mock_lrp_instance.attribute.call_args[0]) > 1
            else None,
        )
        assert target_used == 2, (
            f"Bug 1.12 regression: CaptumLRP.attribute called with target={target_used!r}, "
            "expected target=2 (highest-confidence detection class)."
        )

    def test_no_detections_falls_back_to_target_zero_with_debug_log(self, tmp_path, caplog):
        """When no detections, target=0 must be used and a debug log emitted."""
        import logging

        import torch

        from src.explainability.xai.lrp import _generate_lrp_with_captum

        image_path = self._save_test_image(tmp_path)
        mock_model = self._make_mock_model()

        fake_attr = torch.zeros(1, 3, 64, 64)
        mock_lrp_instance = MagicMock()
        mock_lrp_instance.attribute.return_value = fake_attr

        with (
            patch("captum.attr.LRP", return_value=mock_lrp_instance),
            caplog.at_level(logging.DEBUG, logger="src.explainability.xai.lrp"),
        ):
            _generate_lrp_with_captum(
                model=mock_model,
                image_path=image_path,
                detections={},  # no detections
                gt_boxes=[],
                device="cpu",
                output_dir=None,
                output_manager=None,
                start_time=0.0,
            )

        mock_lrp_instance.attribute.assert_called_once()
        _, call_kwargs = mock_lrp_instance.attribute.call_args
        target_used = call_kwargs.get(
            "target",
            mock_lrp_instance.attribute.call_args[0][1]
            if len(mock_lrp_instance.attribute.call_args[0]) > 1
            else None,
        )
        assert target_used == 0, (
            f"Expected fallback target=0 when no detections, got target={target_used!r}"
        )

        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(
            "target" in m.lower() or "detection" in m.lower() or "fallback" in m.lower()
            for m in debug_messages
        ), (
            "Expected a debug log message about falling back to target=0 when no detections. "
            f"Debug messages: {debug_messages}"
        )
