from setuptools import find_packages, setup

setup(
    name="yutalk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph==0.3.25",
        "langchain==0.3.23",
        "langchain-openai==0.3.12",
        "python-dotenv==1.1.0",
        "openai==1.70.0",
    ],
) 