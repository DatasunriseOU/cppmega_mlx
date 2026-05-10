from __future__ import annotations

import subprocess
import sys
import textwrap


def test_tilelang_experimental_import_does_not_load_torch_or_dlpack() -> None:
    script = textwrap.dedent(
        """
        import sys

        import cppmega_mlx.nn._tilelang._experimental as experimental

        assert experimental.__all__
        loaded = {
            name
            for name in sys.modules
            if name == "torch" or name.startswith("torch.") or "dlpack" in name.lower()
        }
        assert not loaded, sorted(loaded)[:20]
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
