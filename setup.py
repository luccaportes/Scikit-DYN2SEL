from distutils.core import setup

setup(
    name="dyn2sel",  # How you named your package folder (MyLib)
    packages=[
        "dyn2sel",
        "dyn2sel.ensemble",
        "dyn2sel.dcs_techniques",
        "dyn2sel.apply_dcs",
        "dyn2sel.dcs_techniques.extras",
        "dyn2sel.dcs_techniques.from_deslib",
        "dyn2sel.utils",
        "dyn2sel.utils.evaluators",
    ],  # Chose the same as "name"
    version="0.1.5",  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="A framework for applying and implementing DCS techniques in the data stream mining context.",  # Give a short description about your library
    author="Lucca Portes",  # Type in your name
    author_email="lucca54@gmail.com",  # Type in your E-Mail
    url="https://github.com/luccaportes/Scikit-DYN2SEL",  # Provide either the link to your github or to your website
    download_url="https://github.com/luccaportes/Scikit-DYN2SEL/archive/v_015.tar.gz",  # I explain this later on
    keywords=[
        "Dynamic Selection",
        "Machine Learning",
        "Data Streams",
    ],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        "numpy==1.18.1",
        "scikit-learn==0.23.1",
        "scikit-multiflow==0.5.3",
        "scipy==1.4.1",
        "deslib==0.3",
        "imblearn"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3",  # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
