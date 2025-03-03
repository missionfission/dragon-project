from setuptools import setup, find_packages

setup(
    name="dragonx-optimizer",
    version="0.1.1",
    description="Hardware-software co-design framework for AI accelerators",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
        "matplotlib",
        "torch"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 