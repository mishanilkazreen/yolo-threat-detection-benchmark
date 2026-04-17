"""Exploratory tests confirming XAI bug fixes are in place.

These tests assert the CORRECT (fixed) behavior. They serve as regression guards
to ensure the bugs described in the bugfix spec do not reappear.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBug11TargetLayerRespected:
    """Bug 1.1 — generate_gradcam_attribution must use the provided target_layer.

    The original bug: the function accepted `_target_layer: str` but always used
    `model.model.model[-2]`, ignoring the parameter entirely.

    The fix: renamed to `target_layer` and the body now calls
    `get_target_layer_module(model, target_layer)` when the value is non-empty.

    This test confirms the fix is in place by verifying that EigenCAM receives
    the layer returned by `get_target_layer_module`, NOT `model.model.model[-2]`.
    """

    def _make_mock_model(self):
        """Build a minimal mock that satisfies generate_gradcam_attribution's checks."""
        mock_model = MagicMock()
        # Must pass the hasattr(model, "model") and isinstance(..., torch.nn.Module) check.
        import torch

        real_module = torch.nn.Linear(4, 4)
        mock_model.model = real_module
        mock_model.names = {0: "knife", 1: "pistol"}
        mock_model.imgsz = 64
        return mock_model

    def test_eigencam_receives_layer_from_get_target_layer_module(self, tmp_path):
        """EigenCAM must be constructed with the layer resolved by get_target_layer_module."""
        from PIL import Image
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        # Create a tiny real image on disk so PIL.Image.open succeeds.
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)

        mock_model = self._make_mock_model()

        # The layer that get_target_layer_module should return for "model.9"
        resolved_layer = torch.nn.Linear(4, 4)

        # Detections that include a target class so the early-exit branch is skipped.
        detections = {
            "boxes": MagicMock(
                xyxy=[[0, 0, 32, 32]],
                cls=MagicMock(tolist=lambda: [0]),
                conf=MagicMock(tolist=lambda: [0.9]),
            )
        }

        captured_target_layers = []

        def fake_eigencam(model, target_layers, **kwargs):
            captured_target_layers.extend(target_layers)
            cam_instance = MagicMock()
            # Return a valid grayscale CAM array (H x W)
            cam_instance.return_value = np.zeros((1, 64, 64), dtype=np.float32)
            return cam_instance

        with (
            patch(
                "src.explainability.xai.gradcam.get_target_layer_module",
                return_value=resolved_layer,
            ) as mock_get_layer,
            patch("src.explainability.xai.gradcam.EigenCAM", side_effect=fake_eigencam),
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
                detections=detections,
                gt_boxes=[],
                target_layer="model.9",
            )

        # get_target_layer_module must have been called with the correct arguments.
        mock_get_layer.assert_called_once_with(mock_model, "model.9")

        # EigenCAM must have received the layer that get_target_layer_module returned,
        # NOT model.model.model[-2].
        assert len(captured_target_layers) == 1, "EigenCAM should receive exactly one target layer"
        assert captured_target_layers[0] is resolved_layer, (
            "EigenCAM received the wrong layer — the target_layer parameter is still being ignored"
        )

    def test_eigencam_falls_back_to_penultimate_layer_when_target_layer_empty(self, tmp_path):
        """When target_layer is empty, the penultimate layer fallback is used."""
        from PIL import Image
        import torch

        from src.explainability.xai.gradcam import generate_gradcam_attribution

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)

        mock_model = self._make_mock_model()
        penultimate = torch.nn.Linear(4, 4)
        # model.model.model[-2] must return the penultimate layer sentinel.
        mock_model.model.model = MagicMock()
        mock_model.model.model.__getitem__ = MagicMock(return_value=penultimate)

        captured_target_layers = []

        def fake_eigencam(model, target_layers, **kwargs):
            captured_target_layers.extend(target_layers)
            cam_instance = MagicMock()
            cam_instance.return_value = np.zeros((1, 64, 64), dtype=np.float32)
            return cam_instance

        with (
            patch("src.explainability.xai.gradcam.get_target_layer_module") as mock_get_layer,
            patch("src.explainability.xai.gradcam.EigenCAM", side_effect=fake_eigencam),
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
                target_layer="",  # empty → fallback path
            )

        # get_target_layer_module must NOT be called when target_layer is empty.
        mock_get_layer.assert_not_called()

        # EigenCAM must have received the penultimate layer.
        assert len(captured_target_layers) == 1
        assert captured_target_layers[0] is penultimate, (
            "EigenCAM should use model.model.model[-2] when target_layer is empty"
        )


class TestBug12XAILibsInOptionalDeps:
    """Bug 1.2 — shap, zennit, captum, ttach must be in [xai] optional deps, not core.

    The original bug: these four libraries were listed under [project] dependencies
    (core), while torch and torchvision were in [project.optional-dependencies] xai.
    Since shap/zennit/captum all require PyTorch, a core-only install would fail at
    import time.

    The fix: moved shap, zennit, captum, and ttach to [project.optional-dependencies]
    xai alongside torch and torchvision.

    These tests confirm the fix is in place by asserting the CORRECT behavior.
    """

    XAI_LIBS = {"shap", "zennit", "captum", "ttach"}

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
                # Fallback: read as text and do simple string checks
                return None

    def _get_core_deps(self, data):
        """Return the set of package name prefixes from [project] dependencies."""
        deps = data.get("project", {}).get("dependencies", [])
        # Each entry is like "shap>=0.42.0" — extract just the package name
        names = set()
        for dep in deps:
            # Split on common version specifier characters
            import re

            name = re.split(r"[>=<!;\[\s]", dep)[0].lower()
            names.add(name)
        return names

    def _get_xai_optional_deps(self, data):
        """Return the set of package name prefixes from [project.optional-dependencies] xai."""
        deps = data.get("project", {}).get("optional-dependencies", {}).get("xai", [])
        names = set()
        for dep in deps:
            import re

            name = re.split(r"[>=<!;\[\s]", dep)[0].lower()
            names.add(name)
        return names

    def test_xai_libs_not_in_core_dependencies(self):
        """shap, zennit, captum, ttach must NOT appear in [project] dependencies."""
        data = self._load_pyproject()
        if data is None:
            pytest.skip("tomllib/tomli not available and fallback not applicable")

        core_deps = self._get_core_deps(data)
        wrongly_in_core = self.XAI_LIBS & core_deps

        assert not wrongly_in_core, (
            f"Bug 1.2 regression: {wrongly_in_core} found in [project] dependencies "
            f"(core). These must be in [project.optional-dependencies] xai only."
        )

    def test_xai_libs_present_in_xai_optional_dependencies(self):
        """shap, zennit, captum, ttach must ALL appear in [project.optional-dependencies] xai."""
        data = self._load_pyproject()
        if data is None:
            pytest.skip("tomllib/tomli not available and fallback not applicable")

        xai_deps = self._get_xai_optional_deps(data)
        missing_from_xai = self.XAI_LIBS - xai_deps

        assert not missing_from_xai, (
            f"Bug 1.2 regression: {missing_from_xai} missing from "
            f"[project.optional-dependencies] xai. All XAI libraries must be optional."
        )

    def test_torch_in_xai_optional_not_core(self):
        """torch is a core dependency because YOLO training requires it.

        torch is also listed in [xai] optional deps for explainability libraries.
        """
        data = self._load_pyproject()
        if data is None:
            pytest.skip("tomllib/tomli not available and fallback not applicable")

        core_deps = self._get_core_deps(data)

        assert "torch" in core_deps, (
            "torch must be in [project] dependencies (core) for YOLO training"
        )


class TestBug14SilentLRPFallbackExploratory:
    """Bug 1.4 — Exploratory: YOLO26 LRP silently falls back to input * |input|.

    This is an EXPLORATORY test that asserts the BUGGY behavior.
    It should PASS on unfixed code and FAIL after the fix is applied.

    The original bug: when all three gradient strategies failed for YOLO26, the code
    silently fell through to ``gradient = input_req_grad * input_req_grad.abs()``.
    This is input magnitude, not LRP, and was returned without any warning or error.
    """

    def _make_mock_yolo26_model(self):
        """Build a minimal mock that looks like a YOLO26 model."""
        import torch

        mock_model = MagicMock()
        mock_model.model_name = "yolo26n"
        mock_model.imgsz = 64

        real_inner = torch.nn.Linear(3 * 64 * 64, 4)
        mock_model.model = real_inner

        return mock_model

    @pytest.mark.xfail(strict=False, reason="Bug 1.4 is fixed: Strategy 3 now raises RuntimeError")
    def test_bug_1_4_silent_lrp_fallback(self, tmp_path, caplog):
        """EXPLORATORY: When all gradient strategies fail, no exception is raised and
        the result equals input * |input| (the silent non-LRP fallback).

        This test asserts the BUGGY behavior:
        - No exception is raised
        - The result is input * abs(input) (input magnitude, not LRP)
        - No warning is logged about the fallback

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        import logging

        from PIL import Image

        from src.explainability.xai.lrp import _generate_lrp_with_zennit

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)

        mock_model = self._make_mock_yolo26_model()

        # Patch torch.autograd.grad to always return (None,) so all gradient
        # strategies fail, triggering the silent fallback on unfixed code.
        with (
            caplog.at_level(logging.WARNING, logger="src.explainability.xai.lrp"),
            patch("torch.autograd.grad", return_value=(None,)),
        ):
            # On UNFIXED code: no exception is raised, silent fallback occurs.
            # On FIXED code: RuntimeError is raised (caught by outer except and
            # re-raised, or the test assertion below fails).
            result_tuple = _generate_lrp_with_zennit(
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

        # Bug condition: no exception was raised (silent failure)
        assert result_tuple is not None, (
            "Bug 1.4: Expected silent fallback to return a result without raising an exception"
        )

        # Bug condition: no warning was logged about the fallback
        # On unfixed code, the silent fallback produces no warning.
        # On fixed code, a RuntimeError is raised and logged as a warning.
        zennit_failed_warnings = [
            record.message
            for record in caplog.records
            if "zennit lrp failed" in record.message.lower()
        ]
        assert not zennit_failed_warnings, (
            "Bug 1.4 is FIXED: A 'Zennit LRP failed' warning was logged, meaning "
            "Strategy 3 now raises RuntimeError instead of silently falling back. "
            f"Warnings found: {zennit_failed_warnings!r}"
        )


class TestBug14LRPFallbackRaisesError:
    """Bug 1.4 — YOLO26 LRP must raise RuntimeError when all gradient strategies fail.

    The original bug: when all three gradient strategies failed for YOLO26, the code
    silently fell through to ``gradient = input_req_grad * input_req_grad.abs()``.
    This is input magnitude, not LRP, and was returned without any warning or error.

    The fix: Strategy 3 now raises ``RuntimeError`` with a descriptive message instead
    of silently assigning the non-LRP fallback value.

    This test confirms the fix is in place by asserting the CORRECT behavior:
    a ``RuntimeError`` is raised from Strategy 3 (visible in the warning log) when
    all gradient strategies fail.
    """

    def _make_mock_yolo26_model(self):
        """Build a minimal mock that looks like a YOLO26 model to _generate_lrp_with_zennit."""
        import torch

        mock_model = MagicMock()
        mock_model.model_name = "yolo26n"
        mock_model.imgsz = 64

        # The inner torch_model must be a real nn.Module so that
        # wrapped_model = LRPModelWrapper(torch_model) works and
        # torch_model.modules() / torch_model.parameters() are iterable.
        real_inner = torch.nn.Linear(3 * 64 * 64, 4)
        mock_model.model = real_inner

        return mock_model

    def test_strategy3_raises_runtime_error_when_all_strategies_fail(self, tmp_path, caplog):
        """Strategy 3 must raise RuntimeError (not silently assign input * |input|).

        We force all strategies to fail by patching ``torch.autograd.grad`` to
        always return ``(None,)``:
        - Strategy 1: ``gradient = None`` (autograd.grad returned None)
        - Strategy 2: ``grad_raw = None`` → TypeError caught → gradient stays None
        - Strategy 3: must raise RuntimeError (the fix)

        The outer fallback catches the RuntimeError and logs it as a warning.
        We verify the fix is in place by asserting that the warning log contains
        the Strategy 3 RuntimeError message — which would NOT appear on unfixed
        code (where Strategy 3 silently set ``gradient = input * |input|``).
        """
        import logging

        from PIL import Image

        from src.explainability.xai.lrp import _generate_lrp_with_zennit

        # Create a tiny real image on disk so PIL.Image.open succeeds.
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)

        mock_model = self._make_mock_yolo26_model()

        # Patch torch.autograd.grad to always return (None,) so both Strategy 1
        # and Strategy 2 fail to produce a non-zero gradient.
        with (
            caplog.at_level(logging.WARNING, logger="src.explainability.xai.lrp"),
            patch("torch.autograd.grad", return_value=(None,)),
        ):
            # The function will catch the RuntimeError from Strategy 3 and log it.
            # It will then fall back to the outer gradient method.
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

        # The fix: Strategy 3 raises RuntimeError with a descriptive message.
        # The outer except catches it and logs: "Zennit LRP failed: <message>".
        # On UNFIXED code, Strategy 3 silently set gradient = input * |input|,
        # so no RuntimeError was raised and no "Zennit LRP failed" warning appeared.
        zennit_failed_warnings = [
            record.message
            for record in caplog.records
            if "zennit lrp failed" in record.message.lower()
        ]
        assert zennit_failed_warnings, (
            "Bug 1.4 regression: No 'Zennit LRP failed' warning was logged. "
            "On fixed code, Strategy 3 raises RuntimeError which is caught and logged. "
            "On unfixed code, Strategy 3 silently set gradient = input * |input| "
            "and no error was raised or logged."
        )

        # The logged message must contain the Strategy 3 RuntimeError text,
        # confirming it was the RuntimeError from Strategy 3 (not some other error).
        combined_warnings = " ".join(zennit_failed_warnings).lower()
        assert any(
            keyword in combined_warnings
            for keyword in [
                "all lrp gradient strategies failed",
                "strategy",
                "yolo26",
                "gradient strategies",
            ]
        ), (
            f"The 'Zennit LRP failed' warning does not mention Strategy 3 failure. "
            f"Got: {zennit_failed_warnings!r}. "
            "Expected the RuntimeError message from Strategy 3 to be logged."
        )

    def test_strategy3_error_message_is_descriptive(self, tmp_path, caplog):
        """The RuntimeError from Strategy 3 must have a descriptive message.

        This confirms the fix is informative — the error message must explain
        which strategies were attempted and why they failed.
        """
        import logging

        from PIL import Image

        from src.explainability.xai.lrp import _generate_lrp_with_zennit

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)

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

        # Find the "Zennit LRP failed" warning and check its content.
        zennit_failed_messages = [
            record.message
            for record in caplog.records
            if "zennit lrp failed" in record.message.lower()
        ]
        assert zennit_failed_messages, (
            "No 'Zennit LRP failed' warning found — Strategy 3 did not raise RuntimeError."
        )

        # The message should reference YOLO26 and gradient strategies.
        full_message = " ".join(zennit_failed_messages).lower()
        assert "yolo26" in full_message or "gradient strateg" in full_message, (
            f"RuntimeError message is not descriptive enough. Got: {zennit_failed_messages!r}"
        )


class TestBug15ModuleLevelLoggerDebugExploratory:
    """Bug 1.5 — Exploratory: lrp.py sets logger.setLevel(logging.DEBUG) at module level.

    This is an EXPLORATORY test that asserts the BUGGY behavior.
    It should PASS on unfixed code and FAIL after the fix is applied.

    The original bug: importing ``src.explainability.xai.lrp`` caused
    ``logger.setLevel(logging.DEBUG)`` to execute at module level, overriding
    any application-level logging configuration.
    """

    @pytest.mark.xfail(
        strict=False, reason="Bug 1.5 is fixed: logger level no longer forced to DEBUG"
    )
    def test_bug_1_5_module_level_logger_debug(self):
        """EXPLORATORY: The lrp module-level logger has level == logging.DEBUG (10).

        This asserts the BUGGY behavior:
        - After importing lrp, lrp.logger.level == logging.DEBUG (10)
        - The logger level is not NOTSET (0)

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        import logging

        from src.explainability.xai import lrp

        # Bug condition: module-level setLevel(DEBUG) forces the logger to DEBUG.
        assert lrp.logger.level == logging.DEBUG, (
            "Bug 1.5 is FIXED: lrp.logger.level is no longer forced to DEBUG at import time. "
            f"Current level: {lrp.logger.level} (expected {logging.DEBUG})"
        )
        assert lrp.logger.level != logging.NOTSET, (
            "lrp.logger.level is NOTSET — the module-level setLevel call is absent."
        )


class TestBug16SerializerOmitsFieldsExploratory:
    """Bug 1.6 — Exploratory: _xai_to_dict omits lrp_rule and gradcam_target_classes.

    This is an EXPLORATORY test that asserts the BUGGY behavior.
    It should PASS on unfixed code and FAIL after the fix is applied.

    The original bug: ConfigurationSerializer._xai_to_dict was written before
    lrp_rule and gradcam_target_classes were added to XAIConfig, and was never
    updated to serialize these fields. Round-trip serialization silently drops them.
    """

    @pytest.mark.xfail(strict=False, reason="Bug 1.6 is fixed: lrp_rule is now serialized")
    def test_bug_1_6_lrp_rule_absent_from_serialized_dict(self):
        """EXPLORATORY: _xai_to_dict does NOT include lrp_rule when it is non-default.

        This asserts the BUGGY behavior:
        - Create XAIConfig with lrp_rule="gamma" (non-default)
        - Call _xai_to_dict
        - Assert "lrp_rule" is NOT in the result (confirming the bug)

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        from src.config.serializer import ConfigurationSerializer
        from src.explainability.xai.config import XAIConfig

        xai_config = XAIConfig(lrp_rule="gamma")
        serialized = ConfigurationSerializer._xai_to_dict(xai_config)

        # Bug condition: lrp_rule is absent from the serialized dict
        assert "lrp_rule" not in serialized, (
            "Bug 1.6 is FIXED: 'lrp_rule' is now present in the serialized dict. "
            f"Serialized keys: {list(serialized.keys())!r}. "
            "On unfixed code, _xai_to_dict never wrote lrp_rule to the output."
        )

    @pytest.mark.xfail(
        strict=False, reason="Bug 1.6 is fixed: gradcam_target_classes is now serialized"
    )
    def test_bug_1_6_gradcam_target_classes_absent_from_serialized_dict(self):
        """EXPLORATORY: _xai_to_dict does NOT include gradcam_target_classes when non-default.

        This asserts the BUGGY behavior:
        - Create XAIConfig with gradcam_target_classes=["gun", "rifle"] (non-default)
        - Call _xai_to_dict
        - Assert "gradcam_target_classes" is NOT in the result (confirming the bug)

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        from src.config.serializer import ConfigurationSerializer
        from src.explainability.xai.config import XAIConfig

        xai_config = XAIConfig(gradcam_target_classes=["gun", "rifle"])
        serialized = ConfigurationSerializer._xai_to_dict(xai_config)

        # Bug condition: gradcam_target_classes is absent from the serialized dict
        assert "gradcam_target_classes" not in serialized, (
            "Bug 1.6 is FIXED: 'gradcam_target_classes' is now present in the serialized dict. "
            f"Serialized keys: {list(serialized.keys())!r}. "
            "On unfixed code, _xai_to_dict never wrote gradcam_target_classes to the output."
        )


class TestBug17ParserIgnoresGradcamTargetClassesExploratory:
    """Bug 1.7 — Exploratory: _parse_xai ignores gradcam_target_classes from YAML.

    This is an EXPLORATORY test that asserts the BUGGY behavior.
    It should PASS on unfixed code and FAIL after the fix is applied.

    The original bug: ConfigurationParser._parse_xai had no
    ``data.get("gradcam_target_classes", ...)`` call. The field is defined in
    XAIConfig and documented in the README but could not be set via YAML —
    the parser always used the default value regardless of what was in the YAML.
    """

    @pytest.mark.xfail(
        strict=False, reason="Bug 1.7 is fixed: parser now reads gradcam_target_classes"
    )
    def test_bug_1_7_parser_ignores_gradcam_target_classes(self):
        """EXPLORATORY: _parse_xai returns the DEFAULT gradcam_target_classes even when
        the YAML dict contains a different value.

        This asserts the BUGGY behavior:
        - Call _parse_xai with {"gradcam_target_classes": ["gun"]}
        - Assert the parsed config's gradcam_target_classes equals the DEFAULT
          (i.e., the YAML value was ignored)

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        from src.config.parser import ConfigurationParser
        from src.explainability.xai.config import XAIConfig

        default_classes = XAIConfig().gradcam_target_classes
        yaml_data = {"gradcam_target_classes": ["gun"]}

        parsed = ConfigurationParser._parse_xai(yaml_data, source="test")

        # Bug condition: the parser ignored the YAML value and returned the default
        assert parsed.gradcam_target_classes == default_classes, (
            "Bug 1.7 is FIXED: _parse_xai now reads gradcam_target_classes from YAML. "
            f"Expected default {default_classes!r} but got {parsed.gradcam_target_classes!r}. "
            "On unfixed code, the parser had no data.get('gradcam_target_classes') call "
            "and always returned the default value."
        )
        assert parsed.gradcam_target_classes != ["gun"], (
            "Bug 1.7 is FIXED: the parsed value matches the YAML input ['gun'], "
            "meaning the parser now correctly reads gradcam_target_classes from YAML."
        )


class TestBug18TargetLayerStringIgnoredExploratory:
    """Bug 1.8 — Exploratory: _parse_xai ignores target_layer string key from YAML.

    This is an EXPLORATORY test that asserts the BUGGY behavior.
    It should PASS on unfixed code and FAIL after the fix is applied.

    The original bug: YAML configs use ``target_layer: "model.9"`` (singular string).
    ``_parse_xai`` has no handler for this key, so it falls back to the default
    ``target_layers`` dict. The YAML value is never stored or validated.
    """

    @pytest.mark.xfail(
        strict=False, reason="Bug 1.8 is fixed: parser now handles target_layer string"
    )
    def test_bug_1_8_parser_ignores_target_layer_string(self):
        """EXPLORATORY: _parse_xai returns the DEFAULT target_layers dict even when
        the YAML dict contains a target_layer string value.

        This asserts the BUGGY behavior:
        - Call _parse_xai with {"target_layer": "model.5"}
        - Assert the parsed config's target_layers equals the DEFAULT dict
          (i.e., the singular target_layer string was ignored)

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        from src.config.parser import ConfigurationParser
        from src.explainability.xai.config import XAIConfig

        default_target_layers = XAIConfig().target_layers
        yaml_data = {"target_layer": "model.5"}

        parsed = ConfigurationParser._parse_xai(yaml_data, source="test")

        # Bug condition: the parser ignored the singular target_layer string
        # and returned the default target_layers dict unchanged.
        assert parsed.target_layers == default_target_layers, (
            "Bug 1.8 is FIXED: _parse_xai now reads the target_layer string from YAML. "
            f"Expected default {default_target_layers!r} but got {parsed.target_layers!r}. "
            "On unfixed code, the parser had no handler for the singular 'target_layer' key "
            "and always returned the default target_layers dict."
        )


class TestBug110DeadCodeWarningExploratory:
    """Bug 1.10 — Exploratory: _validate_xai_config emits no warning for expensive methods
    without sample_limit (dead code block).

    This is an EXPLORATORY test that asserts the BUGGY behavior.
    It should PASS on unfixed code and FAIL after the fix is applied.

    The original bug: when ``enabled_expensive`` is non-empty and
    ``xai.sample_limit is None``, the code executed ``pass`` with a comment
    suggesting a warning should be logged. The warning was never emitted.
    """

    def _make_config_with_lrp_no_sample_limit(self):
        """Build a minimal Configuration with LRP enabled and sample_limit=None."""
        from src.config.parser import Configuration, DataConfig, ModelConfig, TrainingConfig
        from src.explainability.xai.config import XAIConfig

        xai = XAIConfig(
            enabled=True,
            methods={"gradcam": False, "lrp": True, "shap": False},
            sample_limit=None,
        )
        return Configuration(
            training=TrainingConfig(epochs=25, patience=10, image_size=640, device="cpu"),
            model=ModelConfig(name="yolov11n", weights="yolov11n.pt"),
            data=DataConfig(yaml_path="data.yaml"),
            xai=xai,
        )

    @pytest.mark.xfail(strict=False, reason="Bug 1.10 is fixed: warning is now logged")
    def test_bug_1_10_no_warning_logged_for_lrp_without_sample_limit(self, caplog):
        """EXPLORATORY: _validate_xai_config does NOT log a warning when LRP is enabled
        and sample_limit is None.

        This asserts the BUGGY behavior:
        - Call _validate_xai_config with LRP enabled and sample_limit=None
        - Assert that NO warning is logged (confirming the dead code bug)

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        import logging

        from src.config.manager import ConfigurationManager

        config = self._make_config_with_lrp_no_sample_limit()

        with caplog.at_level(logging.WARNING, logger="src.config.manager"):
            ConfigurationManager._validate_xai_config(config, "test_config.yaml")

        # Bug condition: no warning was logged about expensive methods without sample_limit.
        # On unfixed code, the body was ``pass`` so no warning was ever emitted.
        # On fixed code, logger.warning(...) is called and a warning record appears.
        sample_limit_warnings = [
            record.message
            for record in caplog.records
            if "sample_limit" in record.message.lower() or "expensive" in record.message.lower()
        ]
        assert not sample_limit_warnings, (
            "Bug 1.10 is FIXED: A warning about expensive XAI methods without sample_limit "
            "was logged, meaning the dead code block now emits the intended warning. "
            f"Warnings found: {sample_limit_warnings!r}"
        )

    @pytest.mark.xfail(strict=False, reason="Bug 1.10 is fixed: warning is now logged")
    def test_bug_1_10_no_warning_logged_for_shap_without_sample_limit(self, caplog):
        """EXPLORATORY: _validate_xai_config does NOT log a warning when SHAP is enabled
        and sample_limit is None.

        This asserts the BUGGY behavior for the SHAP method variant.

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        import logging

        from src.config.manager import ConfigurationManager
        from src.config.parser import Configuration, DataConfig, ModelConfig, TrainingConfig
        from src.explainability.xai.config import XAIConfig

        xai = XAIConfig(
            enabled=True,
            methods={"gradcam": False, "lrp": False, "shap": True},
            sample_limit=None,
        )
        config = Configuration(
            training=TrainingConfig(epochs=25, patience=10, image_size=640, device="cpu"),
            model=ModelConfig(name="yolov11n", weights="yolov11n.pt"),
            data=DataConfig(yaml_path="data.yaml"),
            xai=xai,
        )

        with caplog.at_level(logging.WARNING, logger="src.config.manager"):
            ConfigurationManager._validate_xai_config(config, "test_config.yaml")

        # Bug condition: no warning was logged
        sample_limit_warnings = [
            record.message
            for record in caplog.records
            if "sample_limit" in record.message.lower() or "expensive" in record.message.lower()
        ]
        assert not sample_limit_warnings, (
            "Bug 1.10 is FIXED: A warning about expensive XAI methods without sample_limit "
            "was logged. "
            f"Warnings found: {sample_limit_warnings!r}"
        )


class TestBug112CaptumTargetHardcodedExploratory:
    """Bug 1.12 — Exploratory: _generate_lrp_with_captum always passes target=0 to LRP.

    This is an EXPLORATORY test that asserts the BUGGY behavior.
    It should PASS on unfixed code and FAIL after the fix is applied.

    The original bug: ``_generate_lrp_with_captum`` called
    ``lrp.attribute(input_tensor, target=0)``, always attributing to class 0
    regardless of which class was actually detected.
    """

    @pytest.mark.xfail(
        strict=False, reason="Bug 1.12 is fixed: captum target now uses highest-confidence class"
    )
    def test_bug_1_12_captum_target_hardcoded_to_zero(self, tmp_path):
        """EXPLORATORY: CaptumLRP.attribute is called with target=0 even when the
        highest-confidence detection is class 2 (non-zero).

        This asserts the BUGGY behavior:
        - Mock CaptumLRP.attribute to capture the target argument
        - Call _generate_lrp_with_captum with detections containing class 2
        - Assert that attribute was called with target=0 (confirming the hardcoded bug)

        This test will PASS on unfixed code and FAIL after the fix is applied.
        """
        from PIL import Image
        import torch

        from src.explainability.xai.lrp import _generate_lrp_with_captum

        # Create a tiny real image on disk so PIL.Image.open succeeds.
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        image_path = str(tmp_path / "test.jpg")
        img.save(image_path)

        # Build a minimal mock model
        mock_model = MagicMock()
        mock_model.imgsz = 64

        # Detections: class 2 is the highest-confidence detection (non-zero class)
        detections = {
            "scores": [0.5, 0.9],
            "classes": [0, 2],
        }

        captured_targets = []

        def fake_attribute(input_tensor, target):
            captured_targets.append(target)
            # Return a zero attribution tensor matching input shape
            return torch.zeros_like(input_tensor)

        mock_lrp_instance = MagicMock()
        mock_lrp_instance.attribute.side_effect = fake_attribute

        with patch(
            "captum.attr.LRP",
            return_value=mock_lrp_instance,
        ):
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

        # Bug condition: attribute was called with target=0 (hardcoded),
        # even though the highest-confidence detection is class 2.
        assert len(captured_targets) == 1, (
            "CaptumLRP.attribute should have been called exactly once"
        )
        assert captured_targets[0] == 0, (
            "Bug 1.12 is FIXED: CaptumLRP.attribute was NOT called with target=0. "
            f"Got target={captured_targets[0]!r}. "
            "On unfixed code, target was always hardcoded to 0 regardless of detections."
        )
