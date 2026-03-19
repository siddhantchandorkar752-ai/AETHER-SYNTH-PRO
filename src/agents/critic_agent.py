import os
from litellm import completion
from loguru import logger
from config import cfg

class CriticAgent:
    """
    The Cognitive Auditor of AETHER-SYNTH.
    Uses LLM reasoning to evaluate GAN performance.
    """
    def __init__(self):
        self.model = cfg.LLM_MODEL
        self.api_key = cfg.LLM_API_KEY
        
        if not self.api_key or "REPLACE" in self.api_key:
            logger.error("❌ API KEY MISSING: Please update your .env file!")
        else:
            logger.success(f"🤖 Critic Agent initialized with model: {self.model}")

    async def audit_generation(self, epoch, loss_g, loss_d):
        """
        Evaluates training metrics and provides strategic feedback.
        """
        prompt = f"""
        System Role: Senior AI Architect
        Task: Audit GAN Training Progress
        
        Current Stats:
        - Epoch: {epoch}
        - Generator Loss: {loss_g:.4f}
        - Discriminator Loss: {loss_d:.4f}
        
        Analysis Request: 
        1. Is the training stable? 
        2. Suggest one hyperparameter tweak if loss is diverging.
        Keep it technical and concise.
        """
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key
            )
            feedback = response.choices[0].message.content
            logger.info(f"📝 Agent Feedback (Epoch {epoch}): {feedback}")
            return feedback
        except Exception as e:
            logger.error(f"❌ Agent Reasoning Failed: {str(e)}")
            return "Fallback: Maintain current training parameters."
