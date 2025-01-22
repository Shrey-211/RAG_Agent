import requests
import json
import time

OLLAMA_BASE = "https://ollama-devsecops.flyt.link/api"
DEFAULT_MODEL = "llama3.2-vision:latest"

class OllamaClient:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.chat_endpoint = f"{OLLAMA_BASE}/chat"

    def generate(self, prompt, system="", context=None, temperature=0.2, top_p=0.9):
        """
        Generate a response using Ollama Chat API with streaming support
        
        Args:
            prompt (str): The prompt to send to the model
            system (str): System message to set context/behavior
            context (list): Previous context from the conversation
            temperature (float): Controls randomness (0.0 to 1.0)
            top_p (float): Controls diversity via nucleus sampling
        
        Returns:
            str: The generated response
        """
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})
        
        # Add context messages if provided
        if context:
            for i, msg in enumerate(context):
                role = "assistant" if i % 2 == 1 else "user"
                messages.append({"role": role, "content": msg})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # print("\n=== LLM Request ===")
        # print(f"Messages: {json.dumps(messages, indent=2)}")
        
        try:
            start_time = time.time()
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
            response = requests.post(self.chat_endpoint, json=payload, stream=True)
            response.raise_for_status()

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
            
            # Handle streaming response
            full_response = ""
            print("\n=== LLM Response Stream ===")
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        
                        if 'message' in json_response:
                            chunk = json_response['message'].get('content', '')
                            if chunk:
                                full_response += chunk
                                print(chunk, end='', flush=True)
                        
                        if json_response.get('done', False):
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"\nError parsing JSON line: {e}")
                        continue
            
            end_time = time.time()
            print(f"\nRequest took {end_time - start_time:.2f} seconds")
            
            return full_response.strip()
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"Error generating response: {str(e)}"

def main():
    client = OllamaClient()
    
    while True:
        prompt = input("Enter a prompt: ")
        response = client.generate(prompt)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
