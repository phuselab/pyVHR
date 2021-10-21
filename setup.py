import setuptools
import os
from shutil import copyfile

requirementPath = 'requirements.txt'
reqs = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        reqs = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyVHR",
    version="1.0.0",
    author="Vittorio Cuculo <vittorio.cuculo@unimi.it>, Donatello Conte <donatello.conte@univ-tours.fr>, Alessandro D'Amelio <alessandro.damelio@unimi.it>, Giuliano Grossi <giuliano.grossi@unimi.it>, Edoardo Mortara <edoardo.mortara@studenti.unimi.it>",
    author_email="",
    description="Package for rPPG methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phuselab/pyVHR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=reqs,
)
