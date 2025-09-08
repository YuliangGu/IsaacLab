import os
from setuptools import setup, find_packages

# Load the long description from the project root README.md
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "..", "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="myrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, for example:
        # "numpy",
        # "gym",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom reinforcement learning utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isaac-sim/IsaacLab/tree/main/myrl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
