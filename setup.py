from setuptools import setup, find_packages

setup(
    name="image-classifier",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "classify=classify.cli:main",
        ],
    },
    package_data={
        "classify": ["config.json"],
    },
)