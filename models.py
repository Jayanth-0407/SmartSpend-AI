import os
import re  #for pattern matching
from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph_agent import run_financial_coach
from langgraph_agent import graph

load_dotenv()

class Transaction(BaseModel):
    date: str = Field(description="The date of the transaction in YYYY-MM-DD format.")
    merchant: str = Field(description="The normalized, clean name of the merchant (e.g., 'Swiggy' instead of 'UPI/9876543210/SWIGGY').")
    category: str = Field(description="The assigned category. Must be one of: Food, Transport, Housing, Utilities, Subscriptions, Entertainment, Income, Other.")
    amount: float = Field(description="The monetary value of the transaction. Use positive numbers for expenses and negative for income.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating how confident you are in the merchant normalization and category.")

class TransactionList(BaseModel):
    transactions: List[Transaction]

app=FastAPI(title="Smart Spend Coach API")

app.add_middleware( 
    CORSMiddleware,      
    allow_origins=["http://localhost:3000","https://smart-spend-ai-front-end.vercel.app"],  
    allow_credentials=True,
    allow_methods=['*'], 
    allow_headers=['*']  
)

def scrub_personal(text):
    # Mask Indian Phone Numbers
    text = re.sub(r"(?:\+?91[\-\s]?)?[6789]\d{9}", '[PHONE_REDACTED]', text)
    # Mask standard Bank Account Numbers
    text = re.sub(r"\b\d{9,18}\b", '[ACCT_REDACTED]', text)
    return text

llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", #Fast and cheap for simple parsing
    api_key=os.environ.get("GROQ_API_KEY")
)
structured_llm = llm.with_structured_output(TransactionList) #return data in pydantic object

@app.post("/upload")  #sends a large payload(our file) to server to process our file

async def process_statement(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Only CSV or TXT files are supported.")
    
    try:
        # 2. Read into memory
        contents = await file.read()
        
        #Decoding the file
        try:
            raw_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = contents.decode("latin-1")
            
        #Scrubbing private info
        safe_text = scrub_personal(raw_text)
        
        #Groq Extraction
        prompt = f"""
        Extract transactions into JSON from the following text. 
        Clean merchant names and categorize them.

        CRITICAL RULES FOR CSV DATA:
        - If you see a value in the 'Debit (DR)' column, it is an expense (make amount positive).
        - If you see a value in the 'Credit (CR)' column, it is income (make amount negative).
        - Clean up the merchant names (e.g., "UPI/ZOMATO/123" -> "Zomato")
        
        Data:
        {safe_text}
        """
        
        print(f"Processing file: {file.filename}...")
        extracted_data = structured_llm.invoke(prompt)
        transaction_dicts = [t.model_dump() for t in extracted_data.transactions]

        print("Running AI Financial Coach analysis...")

        calculated_income = sum(abs(t.get('amount', 0)) for t in transaction_dicts if t.get('amount', 0) < 0)
        final_income = calculated_income if calculated_income > 0 else 85000.0
        
        coach_results = run_financial_coach(
            transactions=transaction_dicts,
            user_id="test_user_1",
            total_income=final_income
        )
        
        #Return to frontend
        return {
            "transactions": transaction_dicts, 
            "insights": coach_results          
        }

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
class ChatRequest(BaseModel):
    history: List[Dict[str,str]]
    user_id: str

@app.post("/chat")

def chat_with_coach(request: ChatRequest):
    try:

        for i, msg in enumerate(request.history):
            # Prints the first 50 characters of each message
            print(f"  [{i}] {msg['role'].upper()}: {msg['content'][:50]}...")
            
        config = {"configurable": {"thread_id": request.user_id}}
        state = graph.get_state(config)
        
        context = "No financial data found. Tell the user to upload a statement first."
        if state and state.values:
            data = state.values
            context = f"""
            USER'S ACTUAL FINANCIAL DATA:
            - Income: ₹{data.get('total_income', 0)}
            - Category Breakdown: {data.get('category_totals', {})}
            - Top Merchants: {data.get('top_merchants', [])}
            """

        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        groq_messages = [
            {
                "role": "system", 
                "content": f"You are the Smart Spend Coach. Base your answers strictly on this data: {context}. Do not ask for the user's name. Answer their financial questions directly without fluff."
            }
        ]
        
        for msg in request.history:
            groq_messages.append({"role": msg["role"], "content": msg["content"]})

        chat_completion = groq_client.chat.completions.create(
            messages=groq_messages,
            model="llama-3.3-70b-versatile",
        )
        
        return {"response": chat_completion.choices[0].message.content}
        
    except Exception as e:
        print(f"CHAT ERROR: {str(e)}")

@app.get("/dashboard/{user_id}")

def get_dashboard_data(user_id: str):
    try:
        #Ask LangGraph for the current memory state of this user
        config = {"configurable": {"thread_id": user_id}}
        state = graph.get_state(config)
        
        if state and state.values:
            return {
                "insights": state.values,
                "transactions": state.values.get("transactions", [])
            }
        return {"error": "No data found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
