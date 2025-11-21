"""
Load and manage Qwen2.5 models for demand forecasting analysis.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class ModelLoader:
    """Load and manage Qwen2.5 models as recommended in the document."""
    
    # Model options from the document
    MODELS = {
        'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct',
        'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
        'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    }
    
    def __init__(self, model_name='qwen2.5-7b', device=None):
        """
        Initialize model loader.
        
        Args:
            model_name: Name of the model to load (from MODELS dict)
            device: Device to load model on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the specified model and tokenizer."""
        if self.model_name not in self.MODELS:
            raise ValueError(f"Model {self.model_name} not found. Available: {list(self.MODELS.keys())}")
        
        model_id = self.MODELS[self.model_name]
        
        print(f"Loading {model_id} on {self.device}...")
        print("Note: First-time download may take several minutes.")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            trust_remote_code=True
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
        return self
    
    def generate_analysis(self, prompt, max_length=1024, temperature=0.7):
        """
        Generate analysis using the loaded model.
        
        Args:
            prompt: Input prompt for the model
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare input
        messages = [
            {"role": "system", "content": "You are an expert AI agent for demand forecasting and inventory management. Analyze data and provide actionable insights."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "").strip()
        else:
            # Fallback: return everything after the prompt
            response = generated_text[len(text):].strip()
        
        return response
    
    def analyze_forecast(self, historical_data, forecast_data, external_factors):
        """
        Generate comprehensive forecast analysis.
        
        Args:
            historical_data: Dict with historical sales information
            forecast_data: Dict with forecast predictions
            external_factors: Dict with external factor information
            
        Returns:
            Analysis text
        """
        prompt = f"""
Analyze the following demand forecasting scenario and provide actionable insights:

HISTORICAL PERFORMANCE:
- Average Daily Sales: {historical_data.get('avg_sales', 'N/A')}
- Sales Trend: {historical_data.get('trend', 'N/A')}
- Seasonality Pattern: {historical_data.get('seasonality', 'N/A')}
- Stockout Rate: {historical_data.get('stockout_rate', 'N/A')}%

FORECAST (Next 30 Days):
- Predicted Average Daily Sales: {forecast_data.get('predicted_avg', 'N/A')}
- Expected Peak Demand: {forecast_data.get('peak_demand', 'N/A')} (Day {forecast_data.get('peak_day', 'N/A')})
- Confidence Interval: {forecast_data.get('confidence', 'N/A')}

EXTERNAL FACTORS:
- Economic Index: {external_factors.get('economic_index', 'N/A')}
- Seasonal Factor: {external_factors.get('seasonal', 'N/A')}
- Promotion Activity: {external_factors.get('promotion', 'N/A')}

CURRENT INVENTORY:
- Current Stock: {historical_data.get('current_stock', 'N/A')} units
- Reorder Point: {historical_data.get('reorder_point', 'N/A')} units
- Lead Time: {historical_data.get('lead_time', 'N/A')} days

Please provide:
1. Risk Assessment: Evaluate stockout and overstock risks
2. Recommended Actions: Specific reorder quantities and timing
3. Key Insights: Important patterns or factors to monitor
4. Optimization Opportunities: Ways to improve inventory efficiency

Keep the analysis concise and actionable.
"""
        
        return self.generate_analysis(prompt, max_length=800)
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Model unloaded successfully")