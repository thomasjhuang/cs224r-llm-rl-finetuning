curl --location 'http://2080.local:8002/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
        "frequency_penalty": 0,
        "max_completion_tokens": 2048,
        "messages": [
{
    "content": "Your response should contain exactly 4 bullet points. Use the markdown bullet points such as: * This is point 1.\n\nI need advice on how to organize my study schedule for the upcoming semester. What are some effective strategies to ensure I cover all my subjects without feeling overwhelmed?\n",
    "role": "user"
  }],
        "model": "anatal/qwen2_05_smol-smoltalk",
        "presence_penalty": 0,
        "stream": false,
        "temperature": 0,
        "top_p": 1
}'