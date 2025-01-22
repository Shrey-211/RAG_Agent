import sys
import os
import logging
from logging_config import setup_logging

# Initialize logging
logger = setup_logging()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_agent.agent import RAGAgent, WebKnowledgeBase
import argparse
from typing import Optional

class ChatInterface:
    def __init__(self, knowledge_base_url: Optional[str] = None, knowledge_dir: Optional[str] = None):
        """
        Initialize chat interface
        
        Args:
            knowledge_base_url: URL of the knowledge base website
            knowledge_dir: Path to local knowledge directory
        """
        self.logger = logging.getLogger(__name__ + ".ChatInterface")
        self.logger.info("Initializing chat interface")
        
        if knowledge_base_url:
            self.logger.info(f"Using web knowledge base: {knowledge_base_url}")
            knowledge_source = WebKnowledgeBase(knowledge_base_url)
        elif knowledge_dir:
            self.logger.info(f"Using local knowledge directory: {knowledge_dir}")
            knowledge_source = knowledge_dir
        else:
            self.logger.error("No knowledge source provided")
            raise ValueError("Either knowledge_base_url or knowledge_dir must be provided")
            
        print("Initializing RAG agent... This may take a moment.")
        try:
            self.agent = RAGAgent(knowledge_source)
            self.logger.info("RAG agent initialized successfully")
            print("RAG agent initialized successfully!")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG agent: {str(e)}")
            raise
    
    def print_welcome(self):
        """Print welcome message and instructions"""
        welcome_message = """
=== RAG Agent Chat Interface ===
Type your questions and press Enter to get responses.
Type 'exit', 'quit', or press Ctrl+C to end the chat.
=====================================
"""
        print(welcome_message)
        self.logger.info("Welcome message displayed")
    
    def format_response(self, response: str) -> str:
        """Format the agent's response for display"""
        formatted = f"\n🤖 Agent Response:\n{response}\n"
        self.logger.debug(f"Formatted response: {formatted}")
        return formatted
    
    def run(self):
        """Run the chat interface"""
        self.logger.info("Starting chat interface")
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    self.logger.info("User requested to exit")
                    print("\nGoodbye! 👋")
                    break
                
                # Skip empty inputs
                if not user_input:
                    self.logger.debug("Skipping empty input")
                    continue
                
                # Process query and display response
                self.logger.info(f"Processing user input: {user_input}")
                response = self.agent.process_query(user_input)
                print(self.format_response(response))
                
            except KeyboardInterrupt:
                self.logger.info("User interrupted the program")
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                self.logger.error(f"Error in chat loop: {str(e)}")
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'exit' to quit.")

def main():
    logger.info("Starting application")
    parser = argparse.ArgumentParser(description="RAG Agent Chat Interface")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        help="URL of the knowledge base website",
        type=str
    )
    group.add_argument(
        "--dir",
        help="Path to local knowledge directory",
        type=str
    )
    
    args = parser.parse_args()
    logger.info(f"Command line arguments: {args}")
    
    try:
        chat = ChatInterface(
            knowledge_base_url=args.url,
            knowledge_dir=args.dir
        )
        chat.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
