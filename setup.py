from setuptools import setup, find_packages
import re
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).parent

    # Read package version
    version = re.search(
        r'__version__ = "(.+?)"',
        (here / "spectral_entropy_regularization" / "__init__.py").read_text("utf8"),
    ).group(1)

    # Read requirements from requirements.txt
    requirements = (
        (here / "requirements.txt").read_text("utf8").strip().split("\n")
    )

    setup(
        name="spectral_entropy_regularization",
        description="",
        version=version,
        author="Christopher N. P. Hassoe",
        author_email="christopher@hassoe.dk",
        license="MIT",
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,
    )