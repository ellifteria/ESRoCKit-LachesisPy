from setuptools import setup, find_packages

setup(
    author='Eleftheria Beres <elli.beres@u.northwestern.edu>',
    description='LachesisPy, an ESRoCKit package for simulated robot control.',
    name='LachesisPy',
    version='0.0.1',
    packages=find_packages(include=['lachesis','lachesis.*']),
    install_requires=[
         "numpy == 1.25.1",
        #  "pandas == 2.0.3",
         "pybullet == 3.2.5",
         "mypy == 1.4.1"
    ],
    python_requires='>=3.11'
)
