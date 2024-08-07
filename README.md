#  Stock Analysis Assistant

## Overview
This project is a simple demonstration of Amazon Bedrock and the Anthropic Claude 3 Sonnet model integrated with Langchain and Streamlit. For more details, please refer to the following links:
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- [Claude 3](https://www.anthropic.com/news/claude-3-family)

## Directory Structure
```
Stock-Analysis-Assistant/
├── img/
├── pages/
│   ├── base.py
│   ├── company.json
│   ├── Home.py
│   ├── libs.py
│   ├── main.py
├── nohup.out
├── README.md
├── requirements.txt
├── SP500.csv
├── tickers.csv
```

## To View Demo and Sample Data
- Access the `demo` folder for the demo video.
- Access the `samples` folder for sample videos.

## Setup Instructions
1. **Install Python:**
   - Follow the [Python installation guide](https://docs.python-guide.org/starting/install3/linux/).

2. **Set Up Python Environment:**
   - Follow the [Python virtual environment setup guide](https://docs.python-guide.org/dev/virtualenvs/#virtualenvironments-ref).

3. **Set Up AWS CLI:**
   - Follow the [AWS CLI quickstart guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html).

4. **Clone the Repository:**
   ```sh
   git clone https://github.com/nguyendinhthi0705/Study-Assistant.git
   cd Stock-Analysis-Assistant
   pip3 install -r requirements.txt
   streamlit run Home.py --server.port 8080
   ```

## Architecture
![Architecture](./img/Architecture.png)

## Learn More About Prompt and Claude 3
- [Introduction to Prompt Design](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design)
- [Claude 3 Model Card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)

