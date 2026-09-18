"""Setup script for MoEInfra — INT4 Mixtral-8x7B inference engine with expert offloading."""
from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="moeinfra",
    version="0.1.0",
    description="INT4 Mixtral-8x7B inference engine with expert offloading for a single T4 16GB GPU",
    author="Parshwa Patel",
    python_requires=">=3.10",
    packages=["cache", "engine", "metrics", "model", "transfer"],
    install_requires=[
        "torch==2.13.0",
        "transformers==5.0.0",
        "bitsandbytes==0.50.2",
        "vllm==0.29.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
