import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AlgoGen-Cyril-Grl", # Replace with your own username
    version="0.0.1",
    author="Cyril-Grl",
#    author_email="Cyril-Grl",
    description="Genetic Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cyril-Grl/AlgoGen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

