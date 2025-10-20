"""
Setup script for C-LIGHT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = ""
readme_file = this_directory.parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()

setup(
    name="c-light-opensource",
    version="0.1.0",
    author="C-LIGHT Team",
    description="Open-source RAG system for cognitive and behavioral science research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/c-light",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=3.0",
        "arxiv>=2.0.0",
    ],
    extras_require={
        "rocksdb": ["python-rocksdb>=0.7.0"],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.4",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clight=c_light_opensource.cli:main",
        ],
    },
)
