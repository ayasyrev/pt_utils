import nox


@nox.session(python=["3.8", "3.9", "3.7"])
def tests(session):
    args = session.posargs or ["--cov"]
    session.install(".", "pytest", "pytest-cov", "coverage[toml]")
    session.run("pytest", *args)


locations = "pt_utils", "tests", "noxfile.py"


@nox.session(python=["3.8", "3.9", "3.7"])
def lint(session):
    args = session.posargs or locations
    session.install("flake8")
    # session.install("flake8-import-order")
    session.run("flake8", *args)
