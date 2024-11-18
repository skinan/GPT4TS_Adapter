from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GPT4TSAdapter",  # Replace with your package name
    # version="1",  # Replace with your package version
    packages=find_packages(),  # Automatically find packages in your directory
    install_requires=requirements,  # Include the dependencies from requirements.txt
    include_package_data=True,  # Include package data specified in MANIFEST.in
)
