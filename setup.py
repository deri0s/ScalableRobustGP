from setuptools import setup, find_packages

setup(
    name="ScalableRobustGP",
    version="0.1.0",
    long_description=open("README.md").read(),
    author="Diego Echeverria Rios (Derios)",
    author_email="derios8@outlook.com",
    url="https://github.com/deri0s/ScalableRobustGP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Tell setuptools to look in `src` for packages
    python_requires='>=3.6',  # Specify the Python versions you support
)
