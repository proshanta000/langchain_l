import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"
headers = {'Content-Type': 'application/json'}

# NOTE: The 'history' list is no longer needed globally,
# as gr.ChatInterface manages the conversation history.

# 1. UPDATED function signature for gr.ChatInterface
def generate_response_chat(prompt, history):
    """
    Takes the user's prompt and the full conversation history.
    History format: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
    """
    
    # Reconstruct the full prompt including history
    # The current prompt is the last message in the history.
    # We join all previous messages to maintain context for Ollama.
    
    full_conversation = []
    for human_msg, bot_msg in history[:-1]: # Exclude the current, un-answered prompt
        full_conversation.append(f"User: {human_msg}")
        full_conversation.append(f"Assistant: {bot_msg}")

    # Add the current prompt
    full_conversation.append(f"User: {prompt}")
    
    final_prompt = "\n".join(full_conversation)

    data = {
        "model": "codehelper",
        "prompt": final_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is it running at http://localhost:11434?"
    
    if response.status_code == 200:
        data = response.json()
        if 'response' in data:
            # gr.ChatInterface expects the function to return only the new response text
            return data['response']
        else:
             return f"Error: No 'response' field found in the API output. Full output: {response.text}"
    else:
        print(f"Error Status Code: {response.status_code}. Response: {response.text}")
        return f"API Error: Status Code {response.status_code}. See console for details."


# --- MODERN GRADIO INTERFACE SETUP ---

# 2. Use gr.ChatInterface instead of gr.Interface
demo = gr.ChatInterface(
    fn=generate_response_chat,
    title="Ollama Chat Interface (codehelper)",
    description="Ask codehelper a question about code!",
    # You can customize the textbox label and submit button if desired:
    textbox=gr.Textbox(placeholder="Enter Your Code/Prompt Here...", container=False, scale=7),
    submit_btn="Send to Ollama"
)

if __name__ == "__main__":
    demo.launch()