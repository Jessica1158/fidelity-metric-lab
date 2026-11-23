from pathlib import Path


def find_project_root(start=None):
    """
    Find the project root by searching upward for
    directories that identify the project structure.

    The project root is expected to contain:
        - src/
        - data/
        - results/

    Parameters
    ----------
    start : str or Path, optional
        Directory from which to start searching.
        Defaults to the current working directory.

    Returns
    -------
    Path
        Absolute path to the project root.
    """

    start = (
        Path.cwd()
        if start is None
        else Path(start).resolve()
    )

    for path in [
        start,
        *start.parents
    ]:

        if (
            (path / "src").exists()
            and
            (path / "data").exists()
            and
            (path / "results").exists()
        ):
            return path

    raise FileNotFoundError(
        "Could not locate project root. "
        "Expected src/, data/, and results/."
    )


def get_project_paths(start=None):
    """
    Return the main project directories.
    """

    root = find_project_root(start)

    return {
        "root": root,
        "src": root / "src",
        "data": root / "data",
        "processed": root / "data" / "processed",
        "results": root / "results",
        "notebooks": root / "notebooks"
    }