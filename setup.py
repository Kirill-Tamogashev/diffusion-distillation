from setuptools import setup

requirements = ["blobfile>=1.0.5", "torch", "tqdm"]
with open("./train_with_stylegan/requirements.txt") as f:
    additional_requirements = f.readlines()
    requirements.extend(additional_requirements)


setup(
    name="improved-diffusion",
    py_modules=["improved_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
