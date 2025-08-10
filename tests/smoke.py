from pathlib import Path

from helios.cli import run_end_to_end


def test_dry_run():
    # Expect to create an image and skip publish when DRY_RUN=true
    run_end_to_end(seed="cats", geo="US", num_ideas=3, draft=True, margin=0.5, blueprint_id=482, print_provider_id=1)
    out_dir = Path(__file__).resolve().parents[1] / "output"
    assert any(f.suffix == ".png" for f in out_dir.glob("*.png"))
