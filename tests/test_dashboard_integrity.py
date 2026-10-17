"""Regression test for MetricsDashboard scientific reporting and data integrity."""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_metrics_dashboard_text_corrections():
    """Verify that false scratch-training claims and inaccurate subtitles are eliminated."""
    dashboard_file = PROJECT_ROOT / "frontend" / "src" / "components" / "MetricsDashboard.tsx"
    assert dashboard_file.exists(), f"MetricsDashboard.tsx not found at {dashboard_file}"

    content = dashboard_file.read_text(encoding="utf-8")

    # 1. Obsolete / incorrect claims must NOT be present
    assert "learned from scratch" not in content.lower(), (
        "Found obsolete 'learned from scratch' claim in MetricsDashboard.tsx"
    )
    assert "Normalized class confusion on 214 test samples" not in content, (
        "Found incorrect 'Normalized class confusion' subtitle in MetricsDashboard.tsx"
    )

    # 2. Correct scientific descriptions MUST be present
    expected_caption = (
        "Strong diagonal values indicate correct predictions for several classes. "
        "MobileNetV3-Small uses ImageNet-pretrained features adapted through transfer "
        "learning to classify crop-leaf conditions."
    )
    assert expected_caption in content, (
        f"Expected caption not found in MetricsDashboard.tsx:\n{expected_caption}"
    )

    expected_subtitle = "Confusion matrix counts on 214 held-out test images"
    assert expected_subtitle in content, (
        f"Expected subtitle not found in MetricsDashboard.tsx:\n{expected_subtitle}"
    )


def test_confusion_matrix_raw_counts_integrity():
    """Verify that the underlying confusion matrix contains raw counts summing to 214."""
    metrics_path = PROJECT_ROOT / "reports" / "mobilenetv3_test_metrics.json"
    assert metrics_path.exists(), f"mobilenetv3_test_metrics.json not found at {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])

    # Must be 11x11 matrix
    assert cm.shape == (11, 11), f"Expected (11, 11) matrix shape, got {cm.shape}"

    # Must be integers (raw counts)
    assert np.issubdtype(cm.dtype, np.integer), f"Matrix elements must be integer counts, got {cm.dtype}"

    # Must sum to 214 test images
    assert cm.sum() == 214, f"Expected 214 total test images, got {cm.sum()}"
    assert data["test_samples_count"] == 214

    # Diagonal sum (correct predictions) = 144
    assert np.trace(cm) == 144
    expected_acc = 144 / 214
    assert abs(data["test_accuracy"] - round(expected_acc, 4)) < 1e-4


def test_evaluation_script_colorbar_label():
    """Verify that the evaluation plotting code provides a descriptive colorbar label."""
    eval_script = PROJECT_ROOT / "src" / "training" / "evaluate_mobilenetv3.py"
    assert eval_script.exists()

    content = eval_script.read_text(encoding="utf-8")
    assert 'plt.colorbar(label="Number of images")' in content, (
        "Expected colorbar to have label='Number of images'"
    )
