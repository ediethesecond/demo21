from setuptools import setup, find_packages

setup(
    name="medical_vision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",  # for experiment tracking
        "matplotlib>=3.5.0",
        "albumentations>=1.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A medical imaging model for classification and segmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medical_vision",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
