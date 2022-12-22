from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "numpy",
    "adversarial-robustness-toolbox",
    "gymnasium",
]
torch_requires = ["torch", "torchvision"]
tf_requires = ["tensorflow", "h5py"]
dev_requires = ["pytest", "tox"]
all_requires = torch_requires + tf_requires + dev_requires

setup(
    name="ares",
    version="0.1.0",
    description="A System-Oriented Wargame Framework for Adversarial ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Ethos-lab/ares",
    install_requires=install_requires,
    extras_require={
        "pytorch": torch_requires,
        "tensorflow": tf_requires,
        "dev": dev_requires,
        "all": all_requires,
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ares=ares:main",
        ]
    },
)
