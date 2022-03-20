from setuptools import setup, find_packages


def read_install_requires():
    print("Reading install requirements")
    return [r.split('\n')[0] for r in open('requirements.txt', 'r').readlines()]


def get_log_description():
    print("Reading README File")
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description


setup_obj = setup(name='normflowpy',
                  long_description=get_log_description(),
                  long_description_content_type="text/markdown",
                  description='A Normalizing flow package using PyTorch',
                  packages=find_packages(
                      exclude=["tests", "tests.*",
                               "requirements", "requirements.*",
                               "examples", "examples.*",
                               ".github.*", "github"]),
                  classifiers=[
                      "Programming Language :: Python :: 3",
                      "License :: OSI Approved :: MIT License",
                      "Operating System :: OS Independent",
                      "Topic :: Scientific/Engineering :: Artificial Intelligence"
                  ],
                  install_requires=read_install_requires(),
                  python_requires='>=3.6'
                  )
