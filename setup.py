from setuptools import setup, find_packages

from fastPTA import __author__, __version__, __url__


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="fastPTA",
    description="Code for fast PTA forecasts",
    keywords="PTAs, GWs",
    version=__version__,
    author=__author__,
    url=__url__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=required_packages,
)
