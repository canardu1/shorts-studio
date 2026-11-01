"""Entry point for the bundled app (PyInstaller)."""
import multiprocessing

from face_reframe.app.window import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
