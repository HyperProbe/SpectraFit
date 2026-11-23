"""Console build entry point."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    from spectrafit.launcher import console_main

    raise SystemExit(console_main())
