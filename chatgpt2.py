from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize FastAPI app
app = FastAPI()

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Track conversation history (limited to last 3 exchanges)
user_messages = []
bot_responses = []

# Define the assistant persona (used initially to establish context)
assistant_persona = (
    "You are a helpful AI assistant for ServiceNow. Your job is to help users "
    "with ServiceNow tasks like managing tickets, resetting passwords, resolving IT issues, "
    "and providing general technical support. You only respond as 'Bot' to ServiceNow or IT-related questions. "
    "Here are some examples:\n"
    "User: How do I reset my password?\n"
    "Bot: You can reset your password by clicking on the 'Forgot Password' link on the login page.\n"
    "User: What is ServiceNow?\n"
    "Bot: ServiceNow is a cloud-based platform that provides IT service management and automates common business processes.\n"
    "Make sure to only respond with clear, concise answers, and never speak on behalf of the user."
    "And don't extend the aswers to much, it's better if it is concise"
)

# Define the data model for incoming requests
class UserInput(BaseModel):
    message: str

def build_conversation_history(user_messages, bot_responses, new_message):
    # Build a limited conversation history (last 3 exchanges)
    conversation = ""
    for user_msg, bot_resp in zip(user_messages[-3:], bot_responses[-3:]):
        conversation += f"User: {user_msg}\nBot: {bot_resp}\n"
    conversation += f"User: {new_message}\nBot: "
    return conversation

def generate_bot_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Create an attention mask with the same shape as the input tensors
    attention_mask = torch.ones(inputs.shape, device=device)
    outputs = model.generate(
        inputs, max_new_tokens=100, attention_mask=attention_mask, 
        num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# API endpoint to get a response from the chatbot
@app.post("/chat/")
async def chat(user_input: UserInput):
    global user_messages, bot_responses

    # Use the assistant persona for the first interaction or every 5 interactions
    if len(user_messages) == 0 or len(user_messages) % 5 == 0:
        conversation_history = f"{assistant_persona}\nUser: {user_input.message}\nBot: "
    else:
        conversation_history = build_conversation_history(user_messages, bot_responses, user_input.message)

    # Generate bot response
    bot_response = generate_bot_response(conversation_history)

    # Store the message and bot response in history
    user_messages.append(user_input.message)
    bot_responses.append(bot_response)

    # Return the bot response as JSON
    return {"response": bot_response}

# Root endpoint to display the HTML page with input form
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>ServiceNow Assistant</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                input[type="text"] { width: 300px; padding: 10px; }
                input[type="submit"] { padding: 10px; }
                #response { margin-top: 20px; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>ServiceNow AI Assistant</h1>
            <p>Type your message below and get a response from the AI assistant:</p>
            <form id="chatForm">
                <input type="text" id="message" placeholder="Type your message here..." required />
                <input type="submit" value="Send" />
            </form>
            <div id="response"></div>

            <script>
                document.getElementById("chatForm").addEventListener("submit", async function(event) {
                    event.preventDefault();
                    let message = document.getElementById("message").value;

                    // Send the message to the API
                    let response = await fetch("/chat/", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({ message: message })
                    });

                    let data = await response.json();
                    document.getElementById("response").innerText = "Bot: " + data.response;
                });
            </script>
        </body>
    </html>
    """




