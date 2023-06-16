from setuptools import setup


with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="improved-diffusion",
    py_modules=["improved_diffusion"],
    install_requires=requirements,
)
