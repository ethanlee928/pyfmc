from setuptools import find_packages, setup

setup(
    name="pyfmc",
    version="0.1.6",
    author="Ethan Lee",
    author_email="ethan2000.el@gmail.com",
    description="Finance Monte-Carlo Simulation using PyTorch",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("."),
    package_dir={"": "."},
    url="https://github.com/ethanlee928/pyfmc",
    install_requires=["torch==2.4.1", "tqdm==4.66.5", "numpy==1.24.4", "pandas==2.0.3", "matplotlib==3.7.1", "seaborn==0.12.2"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
