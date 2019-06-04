from setuptools import setup

setup(name='dstoolkit',
      version='0.1',
      description='Collection of code to ease ml and dl.',
      url="https://github.com/Smitharry/ds_toolkit",
      author='Maria Kuznetsova',
      author_email='kuznetsovamaria1996@gmail.com',
      packages=['deep_learning'],
      install_requires=[
            'keras',
            'tensorflow',
            'numpy'
      ],
      zip_safe=False)
