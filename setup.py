from setuptools import find_packages, setup

setup(
    name='mapper',
    version='0.2.0',
    author='Viacheslav Sharunov',
    author_email='viacheslav.sharunov@gmail.com',
    packages=find_packages(exclude=['tests']),
    description='There is some nice text here',
    long_description=__doc__,
    classifiers=[
        'Programming Lanugage :: Python',
        'Programming Lanugage :: Python :: 3'
    ],
    extras_require={
        'testing': ['pytest',
                    'pytest-cov'
        ],
    },
    install_requires=['click',
                      'networkx'],
    include_package_date=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'mapper = mapper.mapper:cli',
        ],
    },
)
