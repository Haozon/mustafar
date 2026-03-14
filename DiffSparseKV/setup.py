"""
Setup script for DiffSparseKV library
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "diffsparsekv", "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "DiffSparseKV: Differential Sparse Key-Value Cache for Efficient LLM Inference"

setup(
    name="diffsparsekv",
    version="0.1.0",
    author="DiffSparseKV Team",
    author_email="your.email@example.com",
    description="Differential Sparse Key-Value Cache for Efficient LLM Inference",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/diffsparsekv",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "benchmark": [
            "datasets>=2.0.0",
            "tqdm>=4.60.0",
        ],
    },
    keywords="llm, kv-cache, sparsity, efficient-inference, long-context",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/diffsparsekv/issues",
        "Source": "https://github.com/yourusername/diffsparsekv",
    },
)
