import setuptools

from fastPTA import __author__, __version__, __url__


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastPTA",
    description="Code for fast PTA forecasts",
    keywords="PTAs, GWs",
    version=__version__,
    author=__author__,
    url=__url__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "matplotlib>=3.7.4",
        "pandas>=2.0.3",
        "tqdm>=4.64.1",
        "pyyaml>=6.0",
        "corner>=2.2.1",
        "emcee>=3.1.2",
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
    ],
    classifiers=[],
)
