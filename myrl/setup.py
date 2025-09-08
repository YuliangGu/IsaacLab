import io
import os
from setuptools import setup, find_packages


def read_readme() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # Prefer the package local README; fall back to repo root README if present.
    for candidate in (os.path.join(here, "README.md"), os.path.join(here, "..", "README.md")):
        try:
            with io.open(candidate, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
    return "BYOL-augmented PPO components for Isaac Lab."


setup(
    name="myrl",
    version="0.1.0",
    description="BYOL-augmented PPO components and runner for Isaac Lab (RSL-RL)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Yuliang Gu",
    url="https://github.com/YuliangGu/myrl",
    packages=find_packages(include=["myrl", "myrl.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "torch>=2.0",
        "tensordict>=0.2",
        "rsl-rl-lib>=3.0.1",
    ],
    extras_require={
        "dev": ["black", "isort", "flake8", "pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
