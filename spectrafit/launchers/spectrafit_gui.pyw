"""Desktop build entry point; worker dispatch must precede package imports."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    from spectrafit.launcher import gui_main

    raise SystemExit(gui_main())
