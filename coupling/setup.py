"""
Configuração do pacote HAC Coupling.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hac-coupling",
    version="1.0.0",
    author="HAC Team",
    author_email="hac@example.com",
    description="Geometric Coupling Hypothesis implementation for solar-terrestrial physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/hac",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "dataclasses-json>=0.5.0",
        "pytest>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "hac-coupling-validate=validate_coupling:run_validation",
        ],
    },
)
