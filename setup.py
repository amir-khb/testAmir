from setuptools import setup, find_packages

setup(
    name="hyperspectral-classification",
    version="0.1.0",
    description="Hyperspectral Image Classification with Open Set Recognition",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "rasterio>=1.2.0",
        "pandas>=1.3.0",
        "tabulate>=0.8.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)