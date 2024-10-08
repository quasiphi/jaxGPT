from setuptools import setup

install_requires = [line for line in open('requirements.txt').read().splitlines() if len(line) > 0]

setup(
    name='jaxgpt',
    version='0.0.1',
    author='Piotrek Rybiec, Bartosz Sasal',
    packages=['jaxgpt'],
    description='Karpathy gpt-2 port to jax',
    install_requires=install_requires,
)