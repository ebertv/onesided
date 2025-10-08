import tiktoken
import anthropic
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import logging

class TokenCounterUtils:
    def __init__(self):
        """Initialize token counters for both Claude and ChatGPT models"""
        # Load environment variables
        load_dotenv('.env')
        
        # Get API key for Claude token counting
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        if not self.claude_api_key:
            raise ValueError("Please set CLAUDE_API_KEY in .env file for token counting")
        
        # Initialize Claude client for token counting
        self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
        
        # Initialize tiktoken encoders for ChatGPT mode        
        self.gpt4o_encoder = tiktoken.get_encoding("cl100k_base")
        
        # Token counters
        self.claude_input_tokens = 0
        self.claude_output_tokens = 0
        self.chatgpt_input_tokens = 0
        self.chatgpt_output_tokens = 0
        
        # Detailed breakdowns
        self.claude_input_breakdown = {
            'prediction_generation': 0,
            'summary_generation': 0,
        }
        
        self.claude_output_breakdown = {
            'prediction_generation': 0,
            'summary_generation': 0,
        }
        
        self.chatgpt_input_breakdown = {
            'turn_prediction_rubric': 0,
            'summary_rubric': 0,
            'summary_precision_recall': 0,
            'conversation_rubric': 0,
        }
        
        self.chatgpt_output_breakdown = {
            'turn_prediction_rubric': 0,
            'summary_rubric': 0,
            'summary_precision_recall': 0,
            'conversation_rubric': 0,
        }
        
        # Prompt storage for verification
        self.saved_prompts = {
            'claude': [],
            'chatgpt': []
        }
        self.prompt_counter = 0
    
    def count_claude_tokens(self, prompt: str, system_prompt: str = None, category: str = 'prediction_generation', estimated_output_tokens: int = None) -> int:
        """Count tokens for Claude prompt using the official API and add estimated output tokens"""
        try:
            messages = [{"role": "user", "content": prompt}]
            system_used = system_prompt if system_prompt else "You are a helpful assistant"
            
            # Use the official Claude token counting API
            response = self.claude_client.messages.count_tokens(
                model="claude-sonnet-4-20250514",
                system=system_used,
                messages=messages,
            )
            
            input_token_count = response.input_tokens
            
            # Add estimated output tokens based on category
            if estimated_output_tokens is None:
                if category == 'summary_generation':
                    estimated_output_tokens = 124  # 124 tokens per summary as requested
                elif category == 'prediction_generation':
                    estimated_output_tokens = 50   # Typical prediction response
                else:
                    estimated_output_tokens = 100  # Default estimate
            
            total_token_count = input_token_count + estimated_output_tokens
            self.claude_input_tokens += input_token_count
            self.claude_output_tokens += estimated_output_tokens
            self.claude_input_breakdown[category] += input_token_count
            self.claude_output_breakdown[category] += estimated_output_tokens
            
            # Save prompt for verification
            self.prompt_counter += 1
            prompt_info = {
                'id': self.prompt_counter,
                'category': category,
                'input_tokens': input_token_count,
                'estimated_output_tokens': estimated_output_tokens,
                'total_tokens': total_token_count,
                'system_prompt': system_used,
                'user_prompt': prompt,
                'model': 'claude-sonnet-4-20250514'
            }
            self.saved_prompts['claude'].append(prompt_info)
            
            logging.info(f"Claude tokens for {category}: {total_token_count} (input: {input_token_count}, estimated output: {estimated_output_tokens})")
            return total_token_count
            
        except Exception as e:
            logging.error(f"Error counting Claude tokens: {e}")
            return 0
    
    def count_chatgpt_tokens(self, prompt: str, system_prompt: str = None, category: str = 'turn_prediction_rubric', estimated_output_tokens: int = None) -> int:
        """Count tokens for ChatGPT prompt using tiktoken and add estimated output tokens"""
        try:
            # Count system prompt tokens if provided
            system_tokens = 0
            system_used = system_prompt if system_prompt else ""
            if system_prompt:
                system_tokens = len(self.gpt4o_encoder.encode(system_prompt))
            
            # Count user prompt tokens
            user_tokens = len(self.gpt4o_encoder.encode(prompt))
            input_tokens = system_tokens + user_tokens
            
            # Add estimated output tokens based on category
            if estimated_output_tokens is None:
                if category == 'summary_rubric':
                    estimated_output_tokens = 800  # Complex blind evaluation with detailed reasoning
                elif category == 'summary_precision_recall':
                    estimated_output_tokens = 300  # Precision/recall evaluation
                elif category == 'conversation_rubric':
                    estimated_output_tokens = 500  # Conversation evaluation
                elif category == 'turn_prediction_rubric':
                    estimated_output_tokens = 400  # Turn prediction evaluation
                else:
                    estimated_output_tokens = 200  # Default estimate
            
            total_tokens = input_tokens + estimated_output_tokens
            self.chatgpt_input_tokens += input_tokens
            self.chatgpt_output_tokens += estimated_output_tokens
            self.chatgpt_input_breakdown[category] += input_tokens
            self.chatgpt_output_breakdown[category] += estimated_output_tokens
            
            # Save prompt for verification
            self.prompt_counter += 1
            prompt_info = {
                'id': self.prompt_counter,
                'category': category,
                'input_tokens': input_tokens,
                'system_tokens': system_tokens,
                'user_tokens': user_tokens,
                'estimated_output_tokens': estimated_output_tokens,
                'total_tokens': total_tokens,
                'system_prompt': system_used,
                'user_prompt': prompt,
                'model': "gpt-4o"
            }
            self.saved_prompts['chatgpt'].append(prompt_info)
            
            logging.info(f"ChatGPT tokens for {category}: {total_tokens} (input: {input_tokens} [system: {system_tokens}, user: {user_tokens}], estimated output: {estimated_output_tokens})")
            return total_tokens
            
        except Exception as e:
            logging.error(f"Error counting ChatGPT tokens: {e}")
            return 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive token usage summary"""
        claude_total = self.claude_input_tokens + self.claude_output_tokens
        chatgpt_total = self.chatgpt_input_tokens + self.chatgpt_output_tokens
        return {
            "total_tokens": {
                "claude_input": self.claude_input_tokens,
                "claude_output": self.claude_output_tokens,
                "claude_total": claude_total,
                "chatgpt_input": self.chatgpt_input_tokens,
                "chatgpt_output": self.chatgpt_output_tokens,
                "chatgpt_total": chatgpt_total,
                "combined": claude_total + chatgpt_total
            },
            "claude_input_breakdown": self.claude_input_breakdown.copy(),
            "claude_output_breakdown": self.claude_output_breakdown.copy(),
            "chatgpt_input_breakdown": self.chatgpt_input_breakdown.copy(),
            "chatgpt_output_breakdown": self.chatgpt_output_breakdown.copy(),
            "detailed_breakdown": {
                "claude_input_prediction_generation": self.claude_input_breakdown['prediction_generation'],
                "claude_input_summary_generation": self.claude_input_breakdown['summary_generation'],
                "claude_output_prediction_generation": self.claude_output_breakdown['prediction_generation'],
                "claude_output_summary_generation": self.claude_output_breakdown['summary_generation'],
                "chatgpt_input_turn_prediction_rubric": self.chatgpt_input_breakdown['turn_prediction_rubric'],
                "chatgpt_input_summary_rubric": self.chatgpt_input_breakdown['summary_rubric'],
                "chatgpt_input_summary_precision_recall": self.chatgpt_input_breakdown['summary_precision_recall'],
                "chatgpt_input_conversation_rubric": self.chatgpt_input_breakdown['conversation_rubric'],
                "chatgpt_output_turn_prediction_rubric": self.chatgpt_output_breakdown['turn_prediction_rubric'],
                "chatgpt_output_summary_rubric": self.chatgpt_output_breakdown['summary_rubric'],
                "chatgpt_output_summary_precision_recall": self.chatgpt_output_breakdown['summary_precision_recall'],
                "chatgpt_output_conversation_rubric": self.chatgpt_output_breakdown['conversation_rubric']
            },
            "prompt_counts": {
                "claude_prompts": len(self.saved_prompts['claude']),
                "chatgpt_prompts": len(self.saved_prompts['chatgpt']),
                "total_prompts": len(self.saved_prompts['claude']) + len(self.saved_prompts['chatgpt'])
            }
        }
    
    def reset_counters(self):
        """Reset all token counters"""
        self.claude_input_tokens = 0
        self.claude_output_tokens = 0
        self.chatgpt_input_tokens = 0
        self.chatgpt_output_tokens = 0
        for key in self.claude_input_breakdown:
            self.claude_input_breakdown[key] = 0
        for key in self.claude_output_breakdown:
            self.claude_output_breakdown[key] = 0
        for key in self.chatgpt_input_breakdown:
            self.chatgpt_input_breakdown[key] = 0
        for key in self.chatgpt_output_breakdown:
            self.chatgpt_output_breakdown[key] = 0
        self.saved_prompts = {'claude': [], 'chatgpt': []}
        self.prompt_counter = 0
    
    def log_summary(self):
        """Log comprehensive token usage summary"""
        summary = self.get_summary()
        
        logging.info("\n" + "="*50)
        logging.info("TOKEN USAGE SUMMARY")
        logging.info("="*50)
        logging.info(f"Claude Input tokens: {summary['total_tokens']['claude_input']:,}")
        logging.info(f"Claude Output tokens: {summary['total_tokens']['claude_output']:,}")
        logging.info(f"Claude Total tokens: {summary['total_tokens']['claude_total']:,}")
        logging.info(f"ChatGPT Input tokens: {summary['total_tokens']['chatgpt_input']:,}")
        logging.info(f"ChatGPT Output tokens: {summary['total_tokens']['chatgpt_output']:,}")
        logging.info(f"ChatGPT Total tokens: {summary['total_tokens']['chatgpt_total']:,}")
        logging.info(f"Combined total: {summary['total_tokens']['combined']:,}")
        
        logging.info("\nClaude Input Breakdown:")
        for category, count in summary['claude_input_breakdown'].items():
            logging.info(f"  {category}: {count:,}")
        
        logging.info("\nClaude Output Breakdown:")
        for category, count in summary['claude_output_breakdown'].items():
            logging.info(f"  {category}: {count:,}")
        
        logging.info("\nChatGPT Input Breakdown:")
        for category, count in summary['chatgpt_input_breakdown'].items():
            logging.info(f"  {category}: {count:,}")
        
        logging.info("\nChatGPT Output Breakdown:")
        for category, count in summary['chatgpt_output_breakdown'].items():
            logging.info(f"  {category}: {count:,}")
        
        logging.info("\nPrompt Counts:")
        logging.info(f"  Claude prompts: {summary['prompt_counts']['claude_prompts']:,}")
        logging.info(f"  ChatGPT prompts: {summary['prompt_counts']['chatgpt_prompts']:,}")
        logging.info(f"  Total prompts: {summary['prompt_counts']['total_prompts']:,}")
        
        logging.info("="*50)
    
    def save_prompts_to_file(self, output_file: str):
        """Save all counted prompts to a text file for verification"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PROMPT VERIFICATION FILE\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n")
                f.write(f"Total Claude prompts: {len(self.saved_prompts['claude'])}\n")
                f.write(f"Total ChatGPT prompts: {len(self.saved_prompts['chatgpt'])}\n")
                f.write("=" * 80 + "\n\n")
                
                # Save Claude prompts
                f.write("CLAUDE PROMPTS\n")
                f.write("=" * 80 + "\n")
                for i, prompt_info in enumerate(self.saved_prompts['claude'], 1):
                    f.write(f"\n--- CLAUDE PROMPT #{prompt_info['id']} ---\n")
                    f.write(f"Category: {prompt_info['category']}\n")
                    f.write(f"Model: {prompt_info['model']}\n")
                    f.write(f"Input Tokens: {prompt_info['input_tokens']:,}\n")
                    f.write(f"Estimated Output Tokens: {prompt_info['estimated_output_tokens']:,}\n")
                    f.write(f"Total Tokens: {prompt_info['total_tokens']:,}\n")
                    f.write(f"\nSYSTEM PROMPT:\n{'-' * 40}\n")
                    f.write(prompt_info['system_prompt'])
                    f.write(f"\n{'-' * 40}\n")
                    f.write(f"\nUSER PROMPT:\n{'-' * 40}\n")
                    f.write(prompt_info['user_prompt'])
                    f.write(f"\n{'-' * 40}\n")
                    f.write("\n" + "=" * 80 + "\n")
                
                # Save ChatGPT prompts
                f.write("\nCHATGPT PROMPTS\n")
                f.write("=" * 80 + "\n")
                for i, prompt_info in enumerate(self.saved_prompts['chatgpt'], 1):
                    f.write(f"\n--- CHATGPT PROMPT #{prompt_info['id']} ---\n")
                    f.write(f"Category: {prompt_info['category']}\n")
                    f.write(f"Model: {prompt_info['model']}\n")
                    f.write(f"Input Tokens: {prompt_info['input_tokens']:,}\n")
                    f.write(f"System Tokens: {prompt_info['system_tokens']:,}\n")
                    f.write(f"User Tokens: {prompt_info['user_tokens']:,}\n")
                    f.write(f"Estimated Output Tokens: {prompt_info['estimated_output_tokens']:,}\n")
                    f.write(f"Total Tokens: {prompt_info['total_tokens']:,}\n")
                    f.write(f"\nSYSTEM PROMPT:\n{'-' * 40}\n")
                    f.write(prompt_info['system_prompt'])
                    f.write(f"\n{'-' * 40}\n")
                    f.write(f"\nUSER PROMPT:\n{'-' * 40}\n")
                    f.write(prompt_info['user_prompt'])
                    f.write(f"\n{'-' * 40}\n")
                    f.write("\n" + "=" * 80 + "\n")
                
                logging.info(f"Saved {len(self.saved_prompts['claude']) + len(self.saved_prompts['chatgpt'])} prompts to {output_file}")
                
        except Exception as e:
            logging.error(f"Error saving prompts to file: {e}") 