from __future__ import annotations

from src.linear_algebra_demo import main


def test_linear_algebra_demo_main_runs(capsys) -> None:
    main()
    captured = capsys.readouterr().out
    assert "det(A)" in captured
    assert "Eigenvalues" in captured
    assert "A @ x" in captured
