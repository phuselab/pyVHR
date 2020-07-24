import setuptools
import os

requirementPath = 'requirements.txt'
reqs = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        reqs = f.read().splitlines()

setuptools.setup(
    name="pyVHR",
    version="0.1",
    author="V. Cuculo, D. Conte, A. D'Amelio, G. Grossi",
    author_email="grossi@di.unimi.it",
    description="Package for rPPG methods",
    url="https://github.com/phuselab/pyVHR",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=reqs
)
