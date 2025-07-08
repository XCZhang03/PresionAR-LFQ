from setuptools import setup, find_packages

setup(
    name="maskbit",
    version="0.1.0",
    packages=find_packages(),
    description="",
    install_requires=[
        "torch",
        "einops",
        # Add other dependencies
    ],
)