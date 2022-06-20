from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "numpy",
    "scipy",
    "gym",
]
dev_requires = ["pytest", "tox"]
sklearn_requires = ["scikit-learn"]
torch_requires = ["torch", "torchvision"]
tf_requires = ["tensorflow", "h5py"]
keras_requires = ["keras", "h5py"]
art_requires = ["adversarial-robustness-toolbox"]
all_requires = dev_requires + sklearn_requires + torch_requires + tf_requires + art_requires

setup(
    name="ares",
    version="1.0.0",
    description="A System-Oriented Wargame Framework for Adversarial ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "scikit-learn": sklearn_requires,
        "sklearn": sklearn_requires,
        "pytorch": torch_requires,
        "torch": torch_requires,
        "tensorflow": tf_requires,
        "tf": tf_requires,
        "keras": keras_requires,
        "adversarial-robustness-toolbox": art_requires,
        "art": art_requires,
        "all": all_requires,
    },
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ares=ares:main",
        ]
    },
)
