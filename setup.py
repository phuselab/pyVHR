import setuptools
import os

requirementPath = 'requirements.txt'
reqs = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        reqs = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyVHR",
    version="0.0.11",
    author="V. Cuculo, D. Conte, A. D'Amelio, G. Grossi",
    author_email="grossi@di.unimi.it",
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
    install_requires=reqs
)
