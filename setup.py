"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("docs/release-notes.md") as history_file:
    history = history_file.read()

requirements = ["bodo"]
dev_requirements = [
    # lint and tools
    "black",
    "flake8",
    "isort",
    "mypy",
    "pre-commit",
    "seed-isort-config",
    # publishing
    "re-ver",
    "twine",
    # docs
    "jupyter-book",
    "Sphinx>=2.0,<3",
    # tests
    "pytest",
    # devops
    "docker-compose",
]

extra_requires = {"dev": requirements + dev_requirements}

setup(
    author="Ivan Ogasawara",
    author_email="ivan.ogasawara@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="bodoai-examples",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="bodoai-examples",
    name="bodoai-examples",
    packages=find_packages(include=["bodoai_example"]),
    test_suite="tests",
    extras_require=extra_requires,
    url="https://github.com/xmnlab/bodoai-tests",
    version="0.0.1",
    zip_safe=False,
)
