import json
import requests
import yaml
import os

class OllamaClient:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'ollama_config.yml')
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)

        self.model = data["DEFAULT_MODEL"]
        self.chat_endpoint = data['OLLAMA_BASE']
        self.headers = {"Content-Type": "application/json"}

    def generate(self, prompt, system="", context=None, temperature=0.7, top_p=0.9):
        """
        Generate a response using the Ollama API with streaming.
        
        Args:
            prompt (str): The input text for generation.
            system (str, optional): System message for model behavior.
            context (list, optional): List of previous messages (each a dict with "role" and "content").
            temperature (float, optional): Controls randomness (default 0.7).
            top_p (float, optional): Controls diversity (default 0.9).
        
        Returns:
            str: The generated response.
        """
        # Build the full prompt string.
        if system or context:
            full_prompt = ""
            if system:
                full_prompt += f"System: {system}\n"
            if context:
                for msg in context:
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    full_prompt += f"{role}: {content}\n"
            full_prompt += f"User: {prompt}"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p
        }

        try:
            with requests.post(self.chat_endpoint, headers=self.headers, data=json.dumps(payload), stream=True, timeout=30) as response:
                response.raise_for_status()
                full_response = ""
                print("\n=== LLM Response Stream ===")
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            # Try multiple keys to extract the generated text.
                            chunk = (
                                data.get("message", {}).get("content", "")
                                or data.get("token", "")
                                or data.get("response", "")
                            )
                            if chunk:
                                full_response += chunk
                                print(chunk, end="", flush=True)
                            # Break if the stream signals completion.
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            print(f"\nError parsing JSON: {e}")
                            continue
                print("")  # For newline after stream ends.
                return full_response.strip()
        except requests.Timeout:
            print("Error: Request timed out")
            return None
        except requests.RequestException as e:
            print(f"Error: {e}")
            return None


# Testing purpose only
def main():
    client = OllamaClient()
    while True:
        prompt = input("\nEnter a prompt (or type 'exit' to quit): ").strip()
        if prompt.lower() == "exit":
            print("Exiting...")
            break
        response = client.generate(prompt)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()
