# Product Category Prediction

This project aims to predict product categories based on their descriptions. It utilizes various Python libraries and machine learning techniques to process and analyze product description data, ultimately classifying them into predefined categories.

## Project Structure

The project is structured into Jupyter Notebooks, each serving a specific purpose in the data processing and model training pipeline.

### Notebooks

- **PRODUCT_CATEGORY_PREDICTION.ipynb**: The main notebook that includes all the steps from data loading, preprocessing, model training, and evaluation.

### Data

The data used in this project consists of product descriptions and their corresponding categories. It is divided into two main files:

- **Product_Explanation.txt**: Contains product IDs and their descriptions.
- **Product_Categories.txt**: Contains product IDs and their corresponding categories.

### Libraries Used

- `json`: For parsing JSON files.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `nltk`: For natural language processing tasks.
- `tensorflow`: For building and training machine learning models.
- `sklearn`: For model evaluation and additional machine learning tasks.
- `joblib`: For saving and loading models.
- `os`: For interacting with the file system.

### Installation

To run this project, you need to have Python installed on your system. After cloning the repository, install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

```markdown
## Prerequisites

- Docker installed on your machine

## Build the Docker Image

Run the following command in your terminal to build the Docker image. Replace `product-category-prediction` with your desired image name:

```bash
docker build -t product-category-prediction .
```

## Run the Docker Container

After the image has been built, you can run the container using:

```bash
docker run -p 8888:8888 product-category-prediction
```

This command runs the Docker container and maps port `8888` of the container to port `8888` on your host, allowing you to access the Jupyter notebook server by navigating to [http://localhost:8888](http://localhost:8888) in your web browser.

**Note:** The first time you run the `docker build` command, it might take a while as it needs to download the base image and install the required packages. Subsequent builds will be faster due to Docker's caching mechanism.

## Accessing the Jupyter Notebook

Once the container is running, you can access the Jupyter notebook server by opening your web browser and navigating to:

[http://localhost:8888](http://localhost:8888)

## Conclusion

This setup allows you to easily run and test the neural network model in a consistent and reproducible environment. Happy coding!
```

Save this content to a file named `README.md` in the root directory of your project. This will provide clear instructions for building and running the Docker container for your neural network model.