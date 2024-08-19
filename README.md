### Product Description: GenAI Market Sage

**GenAI Market Sage** is a cutting-edge application that leverages the power of Amazon Bedrock and the Anthropic Claude 3 Sonnet model to provide advanced market analysis and insights. It is integrated with Langchain, a versatile framework for building language models, and is deployed using Streamlit for a seamless user interface. The project serves as a demonstration of how modern AI/ML technologies can be combined to offer powerful data-driven solutions in the financial market analysis domain.

### Features:
- **Integration with Amazon Bedrock**: Utilizes AWS Bedrock for scalable and managed AI/ML model hosting.
- **Claude 3 Sonnet Model**: Leverages the Anthropic Claude 3 Sonnet model for natural language processing and generating insights.
- **Langchain Integration**: Uses Langchain for managing and chaining together different language models and functions.
- **Streamlit Interface**: Provides a user-friendly web interface for interacting with the model and viewing analysis results.

### Step-by-Step Deployment Guide

#### Prerequisites
- An AWS account with access to Amazon Bedrock.
- A machine with Python 3.7+ installed.
- AWS CLI configured with appropriate permissions.

#### Step 1: Install Python
1. Follow the [Python installation guide](https://docs.python-guide.org/starting/install3/linux/) to install Python 3.7+ on your system.

#### Step 2: Set Up Python Virtual Environment
1. Create a virtual environment to isolate dependencies:
   ```sh
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   - On Linux/MacOS:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```

#### Step 3: Set Up AWS CLI
1. Install and configure AWS CLI by following the [AWS CLI quickstart guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html).
   ```sh
   aws configure
   ```
   Provide your AWS Access Key, Secret Key, and default region.

#### Step 4: Clone the Repository
1. Clone the project repository from GitHub:
   ```sh
   git clone https://github.com/awsstudygroup/Stock-Analysis-Assistant
   cd Stock-Analysis-Assistant
   ```

#### Step 5: Install Required Packages
1. Install all necessary Python packages listed in the `requirements.txt` file:
   ```sh
   pip3 install -r requirements.txt
   ```

#### Step 6: Run the Application
1. Start the Streamlit application:
   ```sh
   streamlit run Home.py --server.port 8080
   ```
2. Access the application by navigating to `http://localhost:8080` in your web browser.

### Additional Resources
- **Prompt Design**: Learn how to effectively design prompts for Claude 3 by visiting the [Introduction to Prompt Design](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design).
- **Claude 3 Model Card**: Understand the capabilities and limitations of the Claude 3 model by reviewing the [Claude 3 Model Card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf).
