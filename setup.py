from setuptools import setup
import re
from pathlib import Path


def get_version() -> str:
    with open(Path(__file__).parent / "pytorch_nn" / "__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return re.split(r"['\"]", line)[1]


setup(
    name="pytorch_nn",
    version=get_version(),
    description="Remaking PyTorch Layers, Modules and Architectures from scratch for educational purposes",
    author="Ruben Branco",
    author_email="ruben.branco@outlook.pt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    keywords="pytorch",
    packages=[
        "pytorch_nn",
        "pytorch_nn.layers",
    ],
    install_requires=[
        "torch",
    ],
    extras_require={"test": ["pytest", "pytest-cov"]},
    url="https://github.com/RubenBranco/pytorch_nn",
)
