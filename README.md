# Bedrock Connected to the Real World

Practical examples of using Amazon Bedrock with real-world applications and use cases.

## About

This repository demonstrates practical applications of Amazon Bedrock, focusing on real-world scenarios and integrations. It showcases how to combine generative AI with external data sources, vector databases, and APIs to create meaningful applications.

Each example is documented with detailed explanations and demonstrates best practices for building production-ready AI applications.

## Prerequisites

Before running these examples, you'll need:

- An AWS account with access to Amazon Bedrock
- AWS credentials configured (via AWS CLI or environment variables)
- Python 3.8 or later
- Basic understanding of generative AI concepts

## Structure

The repository is organized by increasing complexity:

### 1. Basic Integration
- Simple LLM invocation [01_simple_prompt.py](./python/01_basic_llm/01_simple_prompt.py)
- Context-aware responses [02_system_prompt.py](./python/01_basic_llm/02_system_prompt.py)
- Maintaining conversation history [03_conversation_history.py](./python/01_basic_llm/03_conversation_history.py)

### 2. RAG (Retrieval-Augmented Generation)
- Real-time data augmentation [01_basic_rag.py](./python/02_rag/01_basic_rag.py)
- Vector store integration [02_rag_with_vector_store.py](./python/02_rag/02_rag_with_vector_store.py)

### 3. Tool Use (Function Calling)
- Connecting the AI to external functions [01_tool_use.py](./python/03_tool_use/01_tool_use.py)

## Getting Started

1. Clone this repository
2. Install the required dependencies:
  ```bash
  cd python
  pip install -r requirements.txt
  ```
3. Navigate to your preferred language directory
4. Configure your AWS credentials
5. Run the examples

## Usage

This repository is for educational purposes only. The code samples are designed to be:

- Easy to understand
- Ready to run
- Simple to modify
- Adaptable for your own projects

## Contributing

As an educational repository, it does not accept Pull Requests or Issues. For the latest information on Amazon Bedrock, please refer to the [official documentation](https://aws.amazon.com/developer/generative-ai/bedrock/?trk=2483aad2-15a6-4b7a-a1c5-189851586b67&sc_channel=el).

## Additional Resources

- [Amazon Bedrock Documentation](https://aws.amazon.com/developer/generative-ai/bedrock/?trk=2483aad2-15a6-4b7a-a1c5-189851586b67&sc_channel=el)
- [AWS SDK Documentation](https://aws.amazon.com/developer/tools/?trk=2483aad2-15a6-4b7a-a1c5-189851586b67&sc_channel=el)
- [AWS Free Tier](https://aws.amazon.com/free/?trk=2483aad2-15a6-4b7a-a1c5-189851586b67&sc_channel=el)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.