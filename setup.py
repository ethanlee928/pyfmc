import os
from setuptools import setup, find_packages


setup(
    name="pyfmc",
    version=os.getenv("VER"),
    author="Ethan Lee",
    author_email="ethan2000.el@gmail.com",
    description="Finance Monte-Carlo Simulation using PyTorch",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("."),
    package_dir={"": "."},
    url="https://github.com/ethanlee928/pyfmc",
    install_requires=["torch==2.0.0", "tqdm==4.65.0", "numpy==1.24.3", "pandas==2.0.0"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
