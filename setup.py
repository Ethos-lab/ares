from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "numpy",
    "scipy",
    "Pillow",
    "tqdm",
    "torch",
    "torchvision",
    "gym",
    "adversarial-robustness-toolbox",
]

setup(
    name="ares",
    version="0.0.1",
    description="A System-Oriented Wargame Framework for Adversarial ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=install_requires,
    extras_require={"dev": ["pytest", "tox"]},
    packages=find_packages(),
)
