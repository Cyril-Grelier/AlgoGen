import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="algo_gen",  # Replace with your own username
    version="0.16",
    author="Cyril-Grl",
    #    author_email="Cyril-Grl",
    description="Genetic Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cyril-Grl/AlgoGen",
    packages=setuptools.find_packages(),
    install_requires=[
        'seaborn',
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
