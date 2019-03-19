import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlib",
    version="0.0.2",
    author="Márcio Porto",
    author_email="mflporto@gmail.com",
    description="Implementations of popular Deep Reinforcement Learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarcioPorto/rlib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
