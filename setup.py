from setuptools import setup, find_namespace_packages

setup(
    name='precipitation',
    packages=find_namespace_packages(include=["precipitation.*"]),
    version='0.1',
    description='Package for experiments on precipitation forecasting.',
    author='Eva Walz, Gregor Koehler (Medical Image Computing Group, DKFZ)'
)