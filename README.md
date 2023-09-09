# Codsoft-Task-2

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# SALES PREDICTION\n",
        "\n",
        ">Sales prediction involves forecasting the amount of a product that\n",
        ">customers will purchase, taking into account various factors such as\n",
        ">advertising expenditure, target audience segmentation, and\n",
        ">advertising platform selection."
      ],
      "metadata": {
        "id": "Anoaq3ENWp61"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "IMPORTING IMPORTANT LIBRARIES"
      ],
      "metadata": {
        "id": "EaTrrCP7W3ht"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "PxHL5IcCWgu_"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "LOADING THE DATASET"
      ],
      "metadata": {
        "id": "1OYC9EeaXHll"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\"/content/advertising.csv\")\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "bW5BtmCoXD9H",
        "outputId": "796fd3bf-cfb4-4d4d-de6c-60e77d41232d"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      TV  Radio  Newspaper  Sales\n",
              "0  230.1   37.8       69.2   22.1\n",
              "1   44.5   39.3       45.1   10.4\n",
              "2   17.2   45.9       69.3   12.0\n",
              "3  151.5   41.3       58.5   16.5\n",
              "4  180.8   10.8       58.4   17.9"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-06e4a93c-de20-4d73-854b-78f17dba1bff\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>TV</th>\n",
              "      <th>Radio</th>\n",
              "      <th>Newspaper</th>\n",
              "      <th>Sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>230.1</td>\n",
              "      <td>37.8</td>\n",
              "      <td>69.2</td>\n",
              "      <td>22.1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>44.5</td>\n",
              "      <td>39.3</td>\n",
              "      <td>45.1</td>\n",
              "      <td>10.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>17.2</td>\n",
              "      <td>45.9</td>\n",
              "      <td>69.3</td>\n",
              "      <td>12.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>151.5</td>\n",
              "      <td>41.3</td>\n",
              "      <td>58.5</td>\n",
              "      <td>16.5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>180.8</td>\n",
              "      <td>10.8</td>\n",
              "      <td>58.4</td>\n",
              "      <td>17.9</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-06e4a93c-de20-4d73-854b-78f17dba1bff')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-06e4a93c-de20-4d73-854b-78f17dba1bff button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-06e4a93c-de20-4d73-854b-78f17dba1bff');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-e5caa507-b295-42ee-a869-56f851e20f44\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-e5caa507-b295-42ee-a869-56f851e20f44')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-e5caa507-b295-42ee-a869-56f851e20f44 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nCFpnkznYzjD",
        "outputId": "53a861f8-76cb-433a-e227-bd2ed6103720"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(200, 4)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "yy560mxWXUuL",
        "outputId": "0bb91614-139b-4143-b101-11ca7425e926"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "               TV       Radio   Newspaper       Sales\n",
              "count  200.000000  200.000000  200.000000  200.000000\n",
              "mean   147.042500   23.264000   30.554000   15.130500\n",
              "std     85.854236   14.846809   21.778621    5.283892\n",
              "min      0.700000    0.000000    0.300000    1.600000\n",
              "25%     74.375000    9.975000   12.750000   11.000000\n",
              "50%    149.750000   22.900000   25.750000   16.000000\n",
              "75%    218.825000   36.525000   45.100000   19.050000\n",
              "max    296.400000   49.600000  114.000000   27.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-08dad05a-fc03-4156-a921-40949d299ab2\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>TV</th>\n",
              "      <th>Radio</th>\n",
              "      <th>Newspaper</th>\n",
              "      <th>Sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>200.000000</td>\n",
              "      <td>200.000000</td>\n",
              "      <td>200.000000</td>\n",
              "      <td>200.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>147.042500</td>\n",
              "      <td>23.264000</td>\n",
              "      <td>30.554000</td>\n",
              "      <td>15.130500</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>85.854236</td>\n",
              "      <td>14.846809</td>\n",
              "      <td>21.778621</td>\n",
              "      <td>5.283892</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.700000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.300000</td>\n",
              "      <td>1.600000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>74.375000</td>\n",
              "      <td>9.975000</td>\n",
              "      <td>12.750000</td>\n",
              "      <td>11.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>149.750000</td>\n",
              "      <td>22.900000</td>\n",
              "      <td>25.750000</td>\n",
              "      <td>16.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>218.825000</td>\n",
              "      <td>36.525000</td>\n",
              "      <td>45.100000</td>\n",
              "      <td>19.050000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>296.400000</td>\n",
              "      <td>49.600000</td>\n",
              "      <td>114.000000</td>\n",
              "      <td>27.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-08dad05a-fc03-4156-a921-40949d299ab2')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-08dad05a-fc03-4156-a921-40949d299ab2 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-08dad05a-fc03-4156-a921-40949d299ab2');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-677aa786-950a-41f7-bb23-566e62ec71e2\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-677aa786-950a-41f7-bb23-566e62ec71e2')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-677aa786-950a-41f7-bb23-566e62ec71e2 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "**Basic Observation**\n",
        "\n",
        "---\n",
        "Avg expense spend is highest on TV\n",
        "\n",
        "\n",
        "Avg expense spend is lowest on Radio\n",
        "\n",
        "\n",
        "Max sale is 27 and min is 1.6\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "YfUIQ8j_ZKas"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 268
        },
        "id": "CoHHKCzNbCBz",
        "outputId": "ef945815-7458-453c-f7af-478e8cf70cf7"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 750x250 with 3 Axes>"
            ],
            "image/png": "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w4dx2223oVOnTkhMTMQtt9yC/fv3A2hbF5OTk7F582bk5ORAr9dj4sSJuHDhAvddFRUVmDZtGrp27YqEhAQMHz4cX3/9tUt7e/bsiRUrVmDWrFmIj4/HDTfcgDfffNPlGpPJhAceeABdunRBYmIixo8fj8OHD3PvL126FEOGDME777yD7Oxs6PV6b7uWuAYp5UTEMHv2bNhsNnz00UfcawzDYMOGDSgqKoJarQ5j6wgi+ikrK8PevXuh0+kAAFarFbfccgs+//xzlJWV4aGHHsJ///d/4/vvv+c+84c//AE7d+7Eli1b8J///Ac7duzAwYMHRX/D6XRi2rRpuHr1Knbu3ImvvvoKZ86cwa9//eug3x9BBBO1Wo0XX3wRb7zxBn766SeP9ysqKnDnnXdi5syZOHLkCP7xj39gz549KC4uBgCMHTsWx48fx5UrVwAAO3fuRFpaGnbs2AEAaGlpwbfffotx48YBaFsTb7zxRvzwww84cOAAFi5cCK1Wy/2exWLBH//4R7z77rsoKSmByWTCfffdx73f2NiIu+66C9u2bUNpaSnuvPNOTJ06FZWVlS7tfvXVVzF48GCUlpZi4cKFePTRR7mNOwDcc889qK6uxr/+9S8cOHAAQ4cOxe23346rV69y15SXl+Pjjz/GJ598gkOHDvnVzx0ahiBCzLp165ikpCTB9+677z5m7Nix3N/btm1jADCnT58OTeMIoh1RWFjIqNVqJj4+nomNjWUAMDExMcxHH30k+pnJkyczTzzxBMMwDNPQ0MDodDrmn//8J/d+bW0tExcXxzz66KPca1lZWcxf/vIXhmEY5j//+Q+jVquZyspK7v1jx44xAJjvv/8+sDdIECGisLCQmTZtGsMwDHPrrbcyc+fOZRiGYTZt2sSwqtS8efOYhx56yOVzu3fvZmJiYpjm5mbG6XQyqampzIcffsgwDMMMGTKEWblyJdOtWzeGYRhmz549jFarZZqamhiGYZhOnTox69evF2zPunXrGADMd999x7124sQJBgCzb98+0fvo378/88Ybb3B/Z2VlMXfeeafLNb/+9a+ZSZMmce1PTExkrFaryzW9e/dm/vd//5dhGIZZsmQJo9VqmerqatHfJZRBlnIiopg7dy527dqFiooKAMDatWsxduxYGI3GMLeMIKKT2267DYcOHcK+fftQWFiIOXPmYObM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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Pair Plot Observation\n",
        "\n",
        "\n",
        "---\n",
        "TV Ad costs rising reliably boost sales, but for newspapers and radio, the impact is less predictable.\n"
      ],
      "metadata": {
        "id": "hgdzg55_gZGj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df['TV'].plot.hist(bins=10, color=\"green\", xlabel=\"TV\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 447
        },
        "id": "t8zkZdwfbsAz",
        "outputId": "d42eaf6f-84a7-4d97-c7b4-219a3f724c40"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: ylabel='Frequency'>"
            ]
          },
          "metadata": {},
          "execution_count": 33
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df['Radio'].plot.hist(bins=10, color=\"orange\", xlabel=\"Radio\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 447
        },
        "id": "NkJQ5JdKeBVa",
        "outputId": "0167a83e-4c7e-422d-ddef-d45c7bb3ef58"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: ylabel='Frequency'>"
            ]
          },
          "metadata": {},
          "execution_count": 30
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df['Newspaper'].plot.hist(bins=10,color=\"black\", xlabel=\"newspaper\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 447
        },
        "id": "cl91UmVwfXdu",
        "outputId": "fd4b3824-a8be-4cfc-d3bf-369298cf604e"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: ylabel='Frequency'>"
            ]
          },
          "metadata": {},
          "execution_count": 29
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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vHmzNm/erKamJl1wwQWSpB07dqhdu3YaMGCA/3XcMA4AAJxKQYXM9ddfr9jYWL3yyiv+u/x+//33mjJliq666irdfffdIR0SAADgWIL6iIJzzjlH77//vi6++OKA7du2bdN1110XVveS4SMKgJbjIwoAhItT+hEFXq9X+/fvP2r7/v37VVdXF8wuAQAAWi2okBk3bpymTJmipUuXau/evdq7d6/+/ve/a+rUqRo/fnyoZwQAADimoM6RmTdvnu655x79+te/VlNT05EdRUZq6tSpevrpp0M6IAAAwPEEdY7MTw4ePKivvvpKknTeeeepU6dOIRssVDhHBmg5zpEBEC5O6TkyP6msrFRlZaV69+6tTp068UMQAACcVkGFzHfffadhw4bp/PPP16hRo1RZWSlJmjp1KpdeAwCA0yaokJk5c6aioqJUXl6ujh07+rffeOONeu+990I2HAAAwIkEdbLv+++/rxUrVqh79+4B23v37q2vv/46JIMBAAD8nKCOyBw8eDDgSMxPDhw4oJiYmJMeCgAAoCWCCpmrrrpKCxcu9H/tcrnk8/n01FNP6eqrrw7ZcAAAACcS1FtLTz31lIYNG6aNGzfq0KFDuvfee/X555/rwIED+te//hXqGQEAAI4pqCMyffv21Y4dOzR06FCNGTNGBw8e1Pjx47V582add955oZ4RAADgmFp9RKapqUkjRozQvHnz9OCDD56KmQAAAFqk1SETFRWlrVu3nopZADjMxrtVcyNO4MwW1FtLkyZN0ssvvxzqWQAAAFolqJN9Dx8+rPnz5+vDDz9Uenr6UZ+xVFBQEJLhAAAATqRVR2R27doln8+nbdu2acCAAYqNjdWOHTu0efNm/6O0tLTF+8vPz9fll1+u2NhYJSYmauzYsSorKwt4TUNDg3JycpSQkKDOnTsrKytL1dXVrRkbAAC0Ua369Ot27dqpsrJSiYmJko58JMHcuXOVlJQU1B8+YsQITZgwQZdffrkOHz6sBx54QNu2bdMXX3zhP8pzxx136B//+IcWLFggj8ejadOmKSIiosWXefPp10DbxjkyQNvU0t/frQqZiIgIVVVV+UPG7XartLRU55577slPLGn//v1KTEzU6tWr9X//93+qra3V2WefrUWLFumGG26QJG3fvl0XXnihiouLNXjw4J/dJyEDtG2EDNA2tfT3d1An+/4k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O6/Xq3Xr1p3xaydJP/zwgyIiAn8ttGvXTj6fTxLr11ItWaeMjAzV1NSopKTE/5qVK1fK5/Np0KBBp31mqzl9trFtFi9ebGJiYsyCBQvMF198YW677TYTFxdnqqqqnB4trNxxxx3G4/GYjz76yFRWVvofP/zwg/81t99+u0lNTTUrV640GzduNBkZGSYjI8PBqcPXf1+1ZAxrdzzr1683kZGR5vHHHzc7d+40r732munYsaN59dVX/a+ZPXu2iYuLM2+++abZunWrGTNmzBl5+fCxZGdnm3POOcd/+fXSpUtNly5dzL333ut/Det3RF1dndm8ebPZvHmzkWQKCgrM5s2bzddff22Madk6jRgxwlx22WVm3bp1Zu3ataZ3795cfh0EQiYIf/rTn0xqaqqJjo42V1xxhfn000+dHinsSDrmo7Cw0P+aH3/80dx5553mrLPOMh07djTjxo0zlZWVzg0dxv43ZFi743v77bdN3759TUxMjOnTp4/5y1/+EvC8z+czDz30kElKSjIxMTFm2LBhpqyszKFpw4vX6zW5ubkmNTXVtG/f3px77rnmwQcfNI2Njf7XsH5HrFq16pg/47Kzs40xLVun7777ztx0002mc+fOxu12mylTppi6ujoH/jZ2cxnzX7dsBAAAsAjnyAAAAGsRMgAAwFqEDAAAsBYhAwAArEXIAAAAaxEyAADAWoQMAACwFiEDAACsRcgAAABrETIAAMBahAwAALAWIQMAAKz1/wCc+8oVMQ7vMgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Histogram Observation\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "Most sales result from minimal advertising expenses in newspapers."
      ],
      "metadata": {
        "id": "NBa3U32-9Y5I"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.heatmap(df.corr(),annot = True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 435
        },
        "id": "wEYFjKEGfYl_",
        "outputId": "fc3408fb-43fd-4fff-bf3f-297b5fde312b"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TV has a strong correlation with high sales.\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "We should train our model using linear regression since it's correlated with just one variable: TV."
      ],
      "metadata": {
        "id": "sJKDuzoQ-u34"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)"
      ],
      "metadata": {
        "id": "HGLBXmXO9KsN"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nGbr1cGSEhki",
        "outputId": "c1e7029c-5477-4da9-8cbd-3b841db3b7b9"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "        TV\n",
            "131  265.2\n",
            "96   197.6\n",
            "181  218.5\n",
            "19   147.3\n",
            "153  171.3\n",
            "..     ...\n",
            "67   139.3\n",
            "192   17.2\n",
            "117   76.4\n",
            "47   239.9\n",
            "172   19.6\n",
            "\n",
            "[140 rows x 1 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TyQ6XE31ElXE",
        "outputId": "91290793-f80c-47ff-c4f3-41ac27df7213"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     Sales\n",
            "131   17.7\n",
            "96    16.7\n",
            "181   17.2\n",
            "19    14.6\n",
            "153   16.0\n",
            "..     ...\n",
            "67    13.4\n",
            "192    5.9\n",
            "117    9.4\n",
            "47    23.2\n",
            "172    7.6\n",
            "\n",
            "[140 rows x 1 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ppl1EpTQEq-B",
        "outputId": "6dfa4ed5-e3fd-40f2-f87f-12bc0353b1e9"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "        TV\n",
            "18    69.2\n",
            "170   50.0\n",
            "107   90.4\n",
            "98   289.7\n",
            "177  170.2\n",
            "182   56.2\n",
            "5      8.7\n",
            "146  240.1\n",
            "12    23.8\n",
            "152  197.6\n",
            "61   261.3\n",
            "125   87.2\n",
            "180  156.6\n",
            "154  187.8\n",
            "80    76.4\n",
            "7    120.2\n",
            "33   265.6\n",
            "130    0.7\n",
            "37    74.7\n",
            "74   213.4\n",
            "183  287.6\n",
            "145  140.3\n",
            "45   175.1\n",
            "159  131.7\n",
            "60    53.5\n",
            "123  123.1\n",
            "179  165.6\n",
            "185  205.0\n",
            "122  224.0\n",
            "44    25.1\n",
            "16    67.8\n",
            "55   198.9\n",
            "150  280.7\n",
            "111  241.7\n",
            "22    13.2\n",
            "189   18.7\n",
            "129   59.6\n",
            "4    180.8\n",
            "83    68.4\n",
            "106   25.0\n",
            "134   36.9\n",
            "66    31.5\n",
            "26   142.9\n",
            "113  209.6\n",
            "168  215.4\n",
            "63   102.7\n",
            "8      8.6\n",
            "75    16.9\n",
            "118  125.7\n",
            "143  104.6\n",
            "71   109.8\n",
            "124  229.5\n",
            "184  253.8\n",
            "97   184.9\n",
            "149   44.7\n",
            "24    62.3\n",
            "30   292.9\n",
            "160  172.5\n",
            "40   202.5\n",
            "56     7.3\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SeMmfXxIEuN6",
        "outputId": "53d78c41-13ad-4554-de09-3b2aca65e9cf"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     Sales\n",
            "18    11.3\n",
            "170    8.4\n",
            "107   12.0\n",
            "98    25.4\n",
            "177   16.7\n",
            "182    8.7\n",
            "5      7.2\n",
            "146   18.2\n",
            "12     9.2\n",
            "152   16.6\n",
            "61    24.2\n",
            "125   10.6\n",
            "180   15.5\n",
            "154   20.6\n",
            "80    11.8\n",
            "7     13.2\n",
            "33    17.4\n",
            "130    1.6\n",
            "37    14.7\n",
            "74    17.0\n",
            "183   26.2\n",
            "145   10.3\n",
            "45    16.1\n",
            "159   12.9\n",
            "60     8.1\n",
            "123   15.2\n",
            "179   17.6\n",
            "185   22.6\n",
            "122   16.6\n",
            "44     8.5\n",
            "16    12.5\n",
            "55    23.7\n",
            "150   16.1\n",
            "111   21.8\n",
            "22     5.6\n",
            "189    6.7\n",
            "129    9.7\n",
            "4     17.9\n",
            "83    13.6\n",
            "106    7.2\n",
            "134   10.8\n",
            "66    11.0\n",
            "26    15.0\n",
            "113   20.9\n",
            "168   17.1\n",
            "63    14.0\n",
            "8      4.8\n",
            "75     8.7\n",
            "118   15.9\n",
            "143   10.4\n",
            "71    12.4\n",
            "124   19.7\n",
            "184   17.6\n",
            "97    20.5\n",
            "149   10.1\n",
            "24     9.7\n",
            "30    21.4\n",
            "160   16.4\n",
            "40    16.6\n",
            "56     5.5\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "model = LinearRegression()\n",
        "model.fit(X_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "kauh3hlh_Tkp",
        "outputId": "1fd12076-759e-4bf5-cee8-2423f6aa866b"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "res= model.predict(X_test)\n",
        "print(res)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gi239bVB_rkS",
        "outputId": "e02c0a5c-f3b0-455f-e189-74b17899fed1"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[10.93127621]\n",
            " [ 9.88042193]\n",
            " [12.09159447]\n",
            " [22.99968079]\n",
            " [16.45920756]\n",
            " [10.21976029]\n",
            " [ 7.6199906 ]\n",
            " [20.28497391]\n",
            " [ 8.4464437 ]\n",
            " [17.95886418]\n",
            " [21.44529217]\n",
            " [11.91645209]\n",
            " [15.71485245]\n",
            " [17.42249065]\n",
            " [11.32534656]\n",
            " [13.72260788]\n",
            " [21.68063975]\n",
            " [ 7.18213465]\n",
            " [11.23230217]\n",
            " [18.82362968]\n",
            " [22.88474361]\n",
            " [14.82272095]\n",
            " [16.72739433]\n",
            " [14.35202581]\n",
            " [10.07198391]\n",
            " [13.88133066]\n",
            " [16.20744039]\n",
            " [18.36388094]\n",
            " [19.40378881]\n",
            " [ 8.51759529]\n",
            " [10.85465142]\n",
            " [18.03001578]\n",
            " [22.50709285]\n",
            " [20.3725451 ]\n",
            " [ 7.86628457]\n",
            " [ 8.16731053]\n",
            " [10.40584907]\n",
            " [17.03936669]\n",
            " [10.88749061]\n",
            " [ 8.51212209]\n",
            " [ 9.16343282]\n",
            " [ 8.86788005]\n",
            " [14.96502414]\n",
            " [18.61564811]\n",
            " [18.93309367]\n",
            " [12.76479799]\n",
            " [ 7.6145174 ]\n",
            " [ 8.06879294]\n",
            " [14.02363385]\n",
            " [12.86878878]\n",
            " [13.15339515]\n",
            " [19.70481478]\n",
            " [21.03480222]\n",
            " [17.26376787]\n",
            " [ 9.59034237]\n",
            " [10.55362545]\n",
            " [23.17482317]\n",
            " [16.58509115]\n",
            " [18.22705095]\n",
            " [ 7.54336581]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Accuracy Score: \", model.score(X_test,y_test)*100)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EtJmc1_6Gawh",
        "outputId": "59dcbb33-55a9-4925-8720-197a161f256d"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy Score:  81.50168765722069\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.coef_"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AdzwVEYZGE_E",
        "outputId": "4ced9780-b732-4e8b-80ee-6787824661b2"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.05473199]])"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.intercept_"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wcCfZKIZGqEH",
        "outputId": "4339d7e1-3fd4-455d-e8fc-6a4561d5a55f"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([7.14382225])"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "0.05473199* 69.2 + 7.14382225"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qx0VONugG4js",
        "outputId": "30abe34b-00d5-4318-cba3-87587a4a0506"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "10.931275958"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.style.use('dark_background')\n",
        "plt.grid()\n",
        "plt.plot(res)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 447
        },
        "id": "ESYUfBzMHCRF",
        "outputId": "8ad9d48d-c927-4dc9-bfa4-19bf0478f6ba"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7cad365487f0>]"
            ]
          },
          "metadata": {},
          "execution_count": 44
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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a9WqVdK7776b82fiWOX+uuOOO6SXXnqp4DY8r/2Touzi8XiwfPlyPPvss/R7kiTh2WefxVFHHVXCldmf2bNnY+rUqRnHbnh4GK+//ro4djK1tbUAgP7+fgDA8uXL4fV6M47Z5s2b0dHRIY4ZAKfTic9+9rMIBAJ47bXXxPEqwP3334+///3veO655zK+L47ZRObPn4/Ozk5s27YNv/vd7zB9+nQA4ljl46yzzsJbb72FP/3pT+ju7sY777yDSy+9lP6c97V/UgQfTU1NcLvd6O7uzvh+d3c3pkyZUqJVlQfk+IhjlxuHw4F77rkHL7/8MjZs2AAgfcyi0SiGhoYytp3sx2zJkiUIh8OIRqP4xS9+gXPPPRcffvihOF55+OxnP4tly5bhW9/61oSfiWOWyeuvv45LLrkEp512Gq644grMnj0b//rXv1BTUyOOVR7mzJmDK664Alu2bMGpp56Kn//857j33ntx0UUXAeB/7bflVFuBoFy4//77sWTJEhxzzDGlXort2bx5Mw455BDU1tbi05/+NB5++GEcd9xxpV6WLZk2bRp+8pOf4JRTThG24BpYs2YN/f/3338fr7/+Ojo6OnD++ecjEomUcGX2xel04q233sItt9wCAFi3bh2WLFmCyy+/HL/97W/5vz/3d7ABvb29SCQSaG1tzfh+a2srurq6SrSq8oAcH3HsJnLffffhjDPOwAknnIDOzk76/a6uLvh8PlqOIUz2YxaPx7Ft2za88847+Pa3v43169fj2muvFccrB8uXL0drayveeecdxONxxONxHH/88bjmmmsQj8fR3d0tjlkBhoaG8NFHH2HevHni/MrDvn37sHHjxozvffjhh5gxYwYA/tf+SRF8xONxvP322zjppJPo9xwOB0466SS89tprJVyZ/dmxYwf27duXceyCwSCOOOKISX3s7rvvPpx77rk48cQTsXPnzoyfvf3224jFYhnHbMGCBZg5c+akPmbZOJ1O+Hw+cbxy8Nxzz2HJkiU45JBD6Nebb76J3//+9zjkkEPw1ltviWNWgEAggLlz52Lfvn3i/MrDK6+8goULF2Z8b8GCBejo6ABgzbW/5KpbK77OP/98KRKJSBdddJF0wAEHSL/4xS+k/v5+qaWlpeRrK/VXIBCQli5dKi1dulSSJEm67rrrpKVLl0rTp0+XgHS7VX9/v3TmmWdKS5Yskf7yl79M6lbb+++/XxoYGJCOPfbYjNY+v99Pt/nZz34m7dy5Uzr++OOlZcuWSa+88or0yiuvlHztpfr6wQ9+IK1cuVKaOXOmtGTJEukHP/iBlEwmpZNPPlkcL41f6m4Xccwyv+68807p2GOPlWbOnCkdddRR0j//+U9p//79UlNTkzhWeb5WrFghxWIx6Vvf+pY0d+5c6XOf+5w0MjIiXXjhhXQbztf+0h8Eq76uvPJKaefOndL4+Lj073//Wzr88MNLviY7fB133HFSLh566CG6zXe/+11p3759UiQSkZ555hlp/vz5JV93qb7ycfHFF9NtfD6f9NOf/lTq6+uTRkZGpMcff1xqbW0t+dpL9fXrX/9a2rFjhzQ+Pi51d3dLzzzzDA08xPHS9pUdfIhjpnw9+uijUmdnpzQ+Pi7t3r1bevTRRzP8KsSxyv31yU9+UnrvvfekSCQibdy4Ubr00ksnbMPr2u+Q/0cgEAgEAoHAEiaF5kMgEAgEAoF9EMGHQCAQCAQCSxHBh0AgEAgEAksRwYdAIBAIBAJLEcGHQCAQCAQCSxHBh0AgEAgEAksRwYdAIBAIBAJLEcGHQCAQCAQCSxHBh0AgEAgEAksRwYdAIBAIBAJLEcGHQCAQCAQCSxHBh0AgEAgEAkv5/8LSsKDm1vJUAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.style.use('default')\n",
        "plt.grid()\n",
        "plt.scatter(X_test, y_test)\n",
        "plt.plot(X_test, 7.14382225 + 0.05473199 * X_test, 'r')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 430
        },
        "id": "M_Q7hzagHLN9",
        "outputId": "03e0dd94-534b-40ff-9531-7cef968d4dd7"
      },
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### The solution mentioned above effectively predicts sales using advertising platform datasets."
      ],
      "metadata": {
        "id": "KJQKzWJkHbZP"
      }
    }
  ]
}