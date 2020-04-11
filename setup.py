from setuptools import setup
import setuptools_rust as rust

import os
import re
import setuptools
from pathlib import Path

p = Path(__file__)

setup_requires = [
    'setuptools',
    'setuptools_rust',
    'pytest-runner'
]

install_requires = [
]

test_require = [
    'pytest-cov',
    'pytest-html',
    'pytest'
]

setuptools.setup(
    name="frontier_graph",
    version="0.1.0",
    python_requires='>3.5',
    author="Koji Ono",
    author_email="koji.ono@exwzd.com",
    description="Graph Proximity Search Library based on Frontiaer Algorithm.",
    url='https://github.com/0h-n0/frontier_subgraph',
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=["src"],
    install_requires=install_requires,
    setup_requires=setup_requires,
    rust_extensions=[rust.RustExtension("frontier_graph.frontier")],
    tests_require=test_require,
    extras_require={
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    zip_safe=False,
)
