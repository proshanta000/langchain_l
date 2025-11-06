import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"

headers={
    'Content-Type':'application/json'
}

history=[]

def generate_response(prompt):
    history.append(prompt)
    final_prompt="\n".join(history)

    data = {
        "model": "codehelper",
        "prompt":final_prompt,
        "stream":False
    }

    response=requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code==200:
        response=response.text
        data = json.loads(response)
        actual_response=data['response']
        return actual_response
    else:
        print("error:", response.text)
    
interface = gr.Interface(
    fn=generate_response,
    title="Ollama Chat Interface (codehelper)",
    description="Ask codehelper a question about code!",
    inputs=gr.Textbox(lines=4, placeholder="Enter Your Prompt"),
    outputs=gr.Textbox(lines=4,
                       max_lines=None,
                       label="Ollama Response",
                       interactive=False,
                       autoscroll=True)
)


interface.launch()
