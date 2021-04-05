import nox


@nox.session(python=["3.8"])
def tests(session):
    args = session.posargs or ["--cov"]
    session.install("-r", "requirements_img.txt")
    session.install(".", "pytest", "pytest-cov", "coverage[toml]")
    session.run("pytest", *args)


conda_img_packages = ['accimage', 'pyvips']


@nox.session(python=["3.8"], venv_backend='conda')
@nox.parametrize('img_lib', conda_img_packages)
def tests_conda(session, img_lib):
    args = session.posargs or ["--cov"]
    session.install(".", "pytest", "pytest-cov", "coverage[toml]")
    session.conda_install(img_lib, '--channel=conda-forge')
    session.conda_install('numpy', 'pillow')
    session.run("pytest", '--img_lib', img_lib, *args)


@nox.session(python="3.8")
def coverage(session) -> None:
    """Upload coverage data."""
    session.install("coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)
